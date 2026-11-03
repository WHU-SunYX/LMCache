# SPDX-License-Identifier: Apache-2.0
"""Opt-in coarse-selector capture and an offline NumPy reference.

Runtime imports this module only when coarse_mode=shadow and diag_enabled=true.
Capture copies K before page release; offline analysis never reads freed pages.
Run directly with --input DIR --output report.json on the HOST after Smoke.
"""

import argparse
from dataclasses import dataclass
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import sys
import uuid

logger = logging.getLogger(__name__)
SCHEMA = "aissd-gpu-coarse-diagnostic-v1"
BUCKETS = (8, 16, 32, 64, 96, 128, 256, 384, 512)


def _json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _get(extra, key, default):
    return extra.get("lmcache.aissd_gpu_coarse_diag_" + key,
                     os.environ.get("AISSD_GPU_COARSE_DIAG_" + key.upper(), default))


def _positive_list(value, name, minimum):
    values = value.split(",") if isinstance(value, str) else value
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"Diagnostic {name} must be a nonempty integer list")
    result = []
    for item in values:
        if isinstance(item, bool) or str(item).strip() != str(int(item)) or int(item) < minimum:
            raise ValueError(f"Invalid diagnostic {name}: {value}")
        result.append(int(item))
    if len(set(result)) != len(result):
        raise ValueError(f"Duplicate diagnostic {name}")
    return tuple(sorted(result))


@dataclass(frozen=True)
class DiagnosticOptions:
    directory: str
    abi_path: str
    layers: tuple
    steps: tuple
    max_requests: int


def _identity(path):
    st = os.stat(path)
    return {"dev": st.st_dev, "ino": st.st_ino,
            "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def _check_abi(abi):
    canonical = dict(abi)
    declared = canonical.pop("abi_identity_sha256", None)
    actual = _sha(json.dumps(canonical, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False).encode("utf-8"))
    if declared and declared != actual:
        raise ValueError("Diagnostic ABI identity mismatch")
    if int(abi.get("schema_version", 1)) >= 2 and not declared:
        raise ValueError("Diagnostic ABI schema >=2 requires abi_identity_sha256")
    return actual


def _tensor_bytes(tensor, dtype):
    import torch
    t = tensor.detach().contiguous().to(device="cpu")
    if dtype == "bfloat16":
        if t.dtype != torch.bfloat16:
            raise ValueError("Diagnostic live K/summary must be BF16")
        data = t.view(torch.uint16).numpy().astype("<u2", copy=False).tobytes()
    elif dtype == "float32":
        data = t.float().numpy().astype("<f4", copy=False).tobytes()
    else:
        raise ValueError(f"Unsupported capture dtype {dtype}")
    return data


def _tensor_write(root, name, tensor, dtype):
    data = _tensor_bytes(tensor, dtype)
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"file": name, "shape": list(tensor.shape), "dtype": dtype,
            "nbytes": len(data), "sha256": _sha(data)}


class CoarseDiagnostic:
    @classmethod
    def from_extra(cls, extra, coarse_options):
        directory = str(_get(extra, "dir", "")).strip()
        abi_path = str(_get(extra, "abi_path", "")).strip()
        if not directory or not abi_path:
            raise ValueError("GPU coarse diagnostic requires diag_dir and diag_abi_path")
        maximum = _positive_list([_get(extra, "max_requests", 1)], "max_requests", 1)[0]
        options = DiagnosticOptions(
            str(Path(directory).expanduser().absolute()), str(Path(abi_path).expanduser().absolute()),
            _positive_list(_get(extra, "layers", "9,11"), "layers", 0),
            _positive_list(_get(extra, "steps", "1,2"), "steps", 1), maximum)
        abi_bytes = Path(options.abi_path).read_bytes()
        abi = json.loads(abi_bytes)
        _check_abi(abi)
        if any(layer >= int(abi.get("num_layers", 0)) for layer in options.layers):
            raise ValueError("Diagnostic layer is outside the ABI model")
        return cls(options, coarse_options, abi, abi_bytes)

    def __init__(self, options, coarse_options, abi, abi_bytes):
        self.options, self.coarse_options = options, coarse_options
        self.abi, self.abi_bytes = abi, abi_bytes
        self.requests = {}
        self.selected_count = 0
        self.run_dir = None
        logger.info("[aissd-gpu-coarse-diag] enabled layers=%s steps=%s max_requests=%d dir=%s",
                    options.layers, options.steps, options.max_requests, options.directory)

    def wants_request(self, req_id):
        return str(req_id) not in self.requests and self.selected_count < self.options.max_requests

    def start_request(self, req_id, source, raw, sidecars, layout_fn, **geometry):
        if not self.wants_request(req_id):
            return None
        if any(layer >= geometry["num_layers"] for layer in self.options.layers):
            raise ValueError("Diagnostic layer is outside the model")
        for key in ("num_layers", "chunk_size", "head_dim"):
            if int(self.abi.get(key, 0)) != geometry[key]:
                raise ValueError(f"Diagnostic ABI/model mismatch: {key}")
        if int(self.abi.get("num_kv_heads", 0)) != geometry["kv_heads"]:
            raise ValueError("Diagnostic ABI/model KV head mismatch")
        if self.run_dir is None:
            self.run_dir = Path(self.options.directory) / ("run-" + uuid.uuid4().hex)
            self.run_dir.mkdir(parents=True, exist_ok=False)
            (self.run_dir / "abi.json").write_bytes(self.abi_bytes)
            _json_write(self.run_dir / "run.json", {
                "schema": SCHEMA, "layers": list(self.options.layers),
                "steps": list(self.options.steps), "max_requests": self.options.max_requests,
                "abi_sha256": _sha(self.abi_bytes), "abi_source_path": self.options.abi_path,
                "diagnostic_code_sha256": _sha(Path(__file__).read_bytes()),
                "coarse": {key: getattr(self.coarse_options, key) for key in
                           ("mode", "candidates", "segments", "keep_first", "keep_recent")}})
            logger.info("[aissd-gpu-coarse-diag] run_dir=%s", self.run_dir)
        self.selected_count += 1
        capture = RequestCapture(self, str(req_id), self.selected_count,
                                 source, raw, sidecars, layout_fn, geometry)
        self.requests[str(req_id)] = capture
        return capture

    def begin_case(self, req_id, step, layer, context, logical, starts, ends,
                   num_heads, attention_scale):
        capture = self.requests.get(str(req_id))
        if step not in self.options.steps or layer not in self.options.layers:
            return None
        if capture is None:
            if self.selected_count < self.options.max_requests:
                raise RuntimeError("Diagnostic K was not captured before page release; copy the matching aissd_resident.py")
            return None
        return capture.begin_case(step, layer, context, logical, starts, ends,
                                  num_heads, attention_scale)

    def abort_request(self, req_id):
        capture = self.requests.pop(str(req_id), None)
        if capture is not None:
            capture.metadata["status"] = "capture_failed"
            _json_write(capture.root / "request.json", capture.metadata)

    def close_request(self, req_id):
        capture = self.requests.pop(str(req_id), None)
        if capture is not None:
            capture.metadata["closed"] = True
            _json_write(capture.root / "request.json", capture.metadata)


class RequestCapture:
    def __init__(self, owner, req_id, ordinal, source, raw, sidecars, layout_fn, geometry):
        self.owner = owner
        self.root = owner.run_dir / f"request-{ordinal:04d}"
        self.root.mkdir(exist_ok=False)
        self.next_chunks, self.hashers, self.cases = {}, {}, set()
        self.geometry = dict(geometry)
        self.geometry["num_chunks"] = int(source["prefix_tokens"]) // geometry["chunk_size"]
        if set(raw) != set(range(self.geometry["num_chunks"])):
            raise ValueError("Diagnostic raw archive must map every logical chunk")
        pages = int(source["prefix_tokens"]) // geometry["block_size"]
        block_ids = [int(x) for x in source["coarse_block_ids"][:pages]]
        if len(block_ids) != pages or len(set(block_ids)) != pages or any(x <= 0 for x in block_ids):
            raise ValueError("Diagnostic source pages are incomplete or aliased")
        chunks = []
        for logical in sorted(raw):
            ch = raw[logical]
            path = str(Path(ch["path"]).absolute())
            item = {"logical_chunk": logical, "token_start": int(ch["token_start"]),
                    "token_end": int(ch["token_end"]), "raw_path": path,
                    "raw_identity": _identity(path), "layers": {}}
            if (item["token_start"] != logical * geometry["chunk_size"] or
                    item["token_end"] != (logical + 1) * geometry["chunk_size"]):
                raise ValueError("Diagnostic raw archive token ranges are not logical chunks")
            for layer in owner.options.layers:
                offset, _, size = layout_fn(ch, layer, geometry["num_layers"],
                                            geometry["chunk_size"],
                                            geometry["kv_heads"] * geometry["head_dim"])
                sel = sidecars[layer][logical]
                if (int(sel["token_start"]) != item["token_start"] or
                        int(sel["token_end"]) != item["token_end"]):
                    raise ValueError("Diagnostic sidecar/range mapping mismatch")
                sel_path = str(Path(sel["path"]).absolute())
                with open(sel_path, "rb") as handle:
                    header_bytes = handle.read(4096)
                header = json.loads(header_bytes.split(b"\0", 1)[0].strip())
                if header.get("magic") != "AISSDQKPACK":
                    raise ValueError("Diagnostic sidecar header is not AISSDQKPACK")
                item["layers"][str(layer)] = {
                    "raw_offset": int(offset), "raw_nbytes": int(size),
                    "sidecar_path": sel_path, "sidecar_identity": _identity(sel_path),
                    "sidecar_offset": int(sel["file_offset"]),
                    "sidecar_nbytes": int(sel["nbytes"]),
                    "sidecar_header": header, "sidecar_header_sha256": _sha(header_bytes)}
            chunks.append(item)
        tail = source.get("recomputed_tail")
        if tail is not None and (int(tail["token_index"]) != int(source["prefix_tokens"]) - 1 or
                                 int(tail["block_id"]) != block_ids[-1]):
            raise ValueError("Diagnostic recomputed tail does not match the last live prefix page")
        self.metadata = {"schema": SCHEMA, "req_id": req_id, "ordinal": ordinal,
                         "status": "building", "closed": False,
                         "geometry": self.geometry, "segments": owner.coarse_options.segments,
                         "block_ids": block_ids, "chunks": chunks, "layers": {},
                         "expected_steps": list(owner.options.steps),
                         "expected_layers": list(owner.options.layers),
                         "recomputed_tail": None if tail is None else
                         {k: int(v) for k, v in tail.items()
                          if k in ("token_index", "block_id", "offset")}}
        _json_write(self.root / "request.json", self.metadata)

    def capture_k(self, layer, first, last, tensor):
        if layer not in self.owner.options.layers:
            return
        if first != self.next_chunks.get(layer, 0) or last <= first:
            raise RuntimeError("Diagnostic K batches are out of logical order")
        data = _tensor_bytes(tensor, "bfloat16")
        g = self.geometry
        if len(data) != (last - first) * g["chunk_size"] * g["kv_heads"] * g["head_dim"] * 2:
            raise RuntimeError("Diagnostic K batch geometry mismatch")
        name = f"layer-{layer:04d}/live-k.bf16.bin"
        path = self.root / name
        path.parent.mkdir(exist_ok=True)
        with path.open("wb" if first == 0 else "ab") as handle:
            handle.write(data)
        self.hashers.setdefault(layer, hashlib.sha256()).update(data)
        self.next_chunks[layer] = last

    def capture_summary(self, layer, lo, hi):
        if layer not in self.owner.options.layers:
            return
        g = self.geometry
        if self.next_chunks.get(layer) != g["num_chunks"]:
            raise RuntimeError("Diagnostic live K is incomplete")
        shape = [g["num_chunks"], g["chunk_size"], g["kv_heads"], g["head_dim"]]
        self.metadata["layers"][str(layer)] = {
            "live_k": {"file": f"layer-{layer:04d}/live-k.bf16.bin", "shape": shape,
                       "dtype": "bfloat16", "nbytes": math.prod(shape) * 2,
                       "sha256": self.hashers[layer].hexdigest()},
            "min": _tensor_write(self.root, f"layer-{layer:04d}/min.bf16.bin", lo, "bfloat16"),
            "max": _tensor_write(self.root, f"layer-{layer:04d}/max.bf16.bin", hi, "bfloat16")}

    def complete_summary(self, summary):
        if set(map(int, self.metadata["layers"])) != set(self.owner.options.layers):
            raise RuntimeError("Diagnostic summary is missing requested layers")
        if summary.num_chunks != self.geometry["num_chunks"]:
            raise RuntimeError("Diagnostic and production summary disagree")
        self.metadata["status"] = "ready"
        self.metadata["summary_bytes"] = summary.nbytes
        self.hashers.clear()
        _json_write(self.root / "request.json", self.metadata)

    def begin_case(self, step, layer, context, logical, starts, ends, num_heads, scale):
        if self.metadata["status"] != "ready" or (step, layer) in self.cases:
            raise RuntimeError("Duplicate diagnostic case or summary not ready")
        if int(self.owner.abi.get("num_q_heads", 0)) != num_heads:
            raise RuntimeError("Diagnostic Q-head count does not match captured ABI")
        self.cases.add((step, layer))
        return CaseCapture(self, step, layer, context, logical, starts, ends, num_heads, scale)


class CaseCapture:
    def __init__(self, request, step, layer, context, logical, starts, ends, num_heads, scale):
        self.request = request
        self.prefix = f"cases/step-{step:04d}-layer-{layer:04d}"
        count = len(logical)
        native = {}
        for key in ("aissd_candidate_chunk_ids", "aissd_candidate_dtype", "aissd_candidate_fmt",
                    "aissd_candidate_shape", "aissd_candidate_extent_count",
                    "aissd_candidate_extent_lba", "aissd_candidate_extent_bytes"):
            if key in context:
                native[key] = context[key][0, :count].tolist()
        self.metadata = {"schema": SCHEMA, "req_id": request.metadata["req_id"],
                         "step": int(step), "layer_id": int(layer), "status": "pending",
                         "logical_chunks": list(logical), "token_starts": list(starts),
                         "token_ends": list(ends), "native": native, "num_heads": int(num_heads),
                         "attention_scale": float(scale), "top_n": int(context["top_n_chunks"]),
                         "top_m": int(context.get("aissd_top_m", 8)),
                         "score_mode_code": int(context.get("aissd_score_mode_code", 1)),
                         "selector_algorithm": os.environ.get("AISSD_SPARSE_KV_SELECTOR_ALGO",
                                                               "global_token_head_topm"),
                         "candidate_count": count, "bucket": next(b for b in BUCKETS if count <= b),
                         "coarse_budget": request.owner.coarse_options.candidates,
                         "keep_first": request.owner.coarse_options.keep_first,
                         "keep_recent": request.owner.coarse_options.keep_recent}

    def capture_scores(self, query, upper, scores, ranking):
        self.metadata["query_source_dtype"] = str(query.dtype)
        self.metadata["query_source_shape"] = list(query.shape)
        self.metadata["query_source_stride"] = list(query.stride())
        for name, tensor in (("q", query), ("upper", upper), ("scores", scores)):
            self.metadata[name] = _tensor_write(self.request.root,
                                               self.prefix + f"-{name}.f32.bin", tensor, "float32")
        self.metadata["gpu_ranking_positions"] = list(ranking)

    def set_positions(self, positions):
        self.metadata["coarse_positions"] = list(positions)
        _json_write(self.request.root / (self.prefix + ".pending.json"), self.metadata)

    def finish(self, selected, native_ids):
        if "scores" not in self.metadata or "coarse_positions" not in self.metadata:
            raise RuntimeError("Diagnostic score capture is missing")
        self.metadata.update(status="complete", ssd_selected_logical_chunks=list(selected),
                             ssd_selected_native_ids=list(native_ids))
        _json_write(self.request.root / (self.prefix + ".json"), self.metadata)
        (self.request.root / (self.prefix + ".pending.json")).unlink()


def _array(root, descriptor):
    import numpy as np
    path = (root / descriptor["file"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Capture tensor path escapes its request")
    expected = math.prod(descriptor["shape"]) * (2 if descriptor["dtype"] == "bfloat16" else 4)
    if expected != descriptor["nbytes"] or path.stat().st_size != expected:
        raise ValueError(f"Truncated capture tensor: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != descriptor["sha256"]:
        raise ValueError(f"Capture tensor checksum mismatch: {path}")
    if descriptor["dtype"] == "bfloat16":
        bits = np.fromfile(path, dtype="<u2").astype(np.uint32)
        return (bits << 16).view(np.float32).reshape(descriptor["shape"])
    if descriptor["dtype"] != "float32":
        raise ValueError("Unsupported captured dtype")
    return np.fromfile(path, dtype="<f4").reshape(descriptor["shape"])


def _read_at(path, identity, offset, size):
    if _identity(path) != identity:
        raise ValueError(f"Source file changed since capture: {path}")
    fd = os.open(path, os.O_RDONLY)
    try:
        if offset < 0 or offset + size > identity["size"]:
            raise ValueError(f"Invalid diagnostic source range: {path}")
        parts = []
        while size:
            part = os.pread(fd, size, offset)
            if not part:
                raise OSError(f"Short diagnostic source read: {path}")
            parts.append(part)
            offset += len(part)
            size -= len(part)
        if _identity(path) != identity:
            raise ValueError(f"Source changed during diagnostic read: {path}")
        return b"".join(parts)
    finally:
        os.close(fd)


def _load_archive(metadata, layer, abi, bucket):
    import numpy as np
    g = metadata["geometry"]
    shape = (g["num_chunks"], g["chunk_size"], g["kv_heads"], g["head_dim"])
    raw = np.empty(shape, np.float32)
    packed = np.empty(shape, np.int16)
    info = abi["buckets"][str(bucket)]
    kinfo = info["k_flat"]
    if int(kinfo["tensor_precision"]) != 16 or int(info["q_block"]["tensor_precision"]) != 16:
        raise ValueError("Offline reference requires INT16 K and Q")
    if int(info["output"]["tensor_precision"]) != 32 or info["output"]["tensor_type"].lower() != "float":
        raise ValueError("Offline reference requires FP32 QK output")
    hidden = g["kv_heads"] * g["head_dim"]
    if (int(kinfo["tensor_width"]) != hidden or
            int(info["q_block"]["tensor_width"]) != int(abi["num_q_heads"]) or
            int(info["q_block"]["tensor_ori_channels"]) != hidden):
        raise ValueError("Diagnostic ABI Q/K geometry does not match the capture")
    abi_identity = _check_abi(abi)
    for chunk in metadata["chunks"]:
        i = chunk["logical_chunk"]
        item = chunk["layers"][str(layer)]
        data = _read_at(chunk["raw_path"], chunk["raw_identity"], item["raw_offset"], item["raw_nbytes"])
        raw[i] = (np.frombuffer(data, "<u2").astype(np.uint32) << 16).view(np.float32).reshape(shape[1:])
        header = item["sidecar_header"]
        current_header = _read_at(item["sidecar_path"], item["sidecar_identity"], 0, 4096)
        if _sha(current_header) != item["sidecar_header_sha256"]:
            raise ValueError("Sidecar header changed after capture")
        if (int(header["layer_id"]) != layer or int(header["bucket"]) != bucket or
                str(header.get("layout", "token_major_hkv_dim")) != "token_major_hkv_dim" or
                str(header.get("packed_dtype", "int16")) != "int16" or
                not math.isclose(float(header["scale"]), float(kinfo["tensor_scale_factor"]), rel_tol=1e-12) or
                int(header.get("zero_point", 0)) != int(kinfo.get("tensor_zero_point", 0))):
            raise ValueError("Sidecar header does not match the current layer/bucket/ABI")
        if header.get("abi_identity_sha256") and header["abi_identity_sha256"] != abi_identity:
            raise ValueError("Sidecar ABI identity does not match captured ABI")
        stride = int(header["row_stride_bytes"])
        if stride < hidden * 2 or stride % 2 or stride != int(kinfo["tensor_row_ddr_step"]):
            raise ValueError("Invalid sidecar row stride")
        if (int(header.get("packed_offset", 4096)) != item["sidecar_offset"] or
                int(header["packed_bytes"]) != item["sidecar_nbytes"] or
                item["sidecar_nbytes"] != g["chunk_size"] * stride):
            raise ValueError("Invalid sidecar payload geometry")
        data = _read_at(item["sidecar_path"], item["sidecar_identity"],
                        item["sidecar_offset"], item["sidecar_nbytes"])
        values = np.frombuffer(data, "<i2").reshape(g["chunk_size"], stride // 2)
        packed[i] = values[:, :g["kv_heads"] * g["head_dim"]].reshape(shape[1:])
    return raw, packed


def _quantize(value, scale, zero_point, *, query=False):
    import numpy as np
    if not math.isfinite(scale) or scale <= 0 or not -32768 <= zero_point <= 32767:
        raise ValueError("Invalid diagnostic quantization scale/zero point")
    value = np.asarray(value, np.float64)
    rounded = np.rint(value / scale) + zero_point if query else np.rint(value / scale + zero_point)
    return np.clip(rounded, -32768, 32767).astype(np.int16), {
        "clamped_low": int((rounded < -32768).sum()),
        "clamped_high": int((rounded > 32767).sum())}


def _qk(k, q):
    """Independent explicit GQA mapping; no reuse of the runtime reshape/einsum."""
    import numpy as np
    n, tokens, kv_heads, dim = k.shape
    q = np.asarray(q, np.float64).reshape(-1, dim)
    if len(q) % kv_heads:
        raise ValueError("Invalid GQA geometry")
    out = np.empty((n, tokens, len(q)), np.float32)
    group = len(q) // kv_heads
    for head in range(len(q)):
        out[:, :, head] = np.einsum("ntd,d->nt", k[:, :, head // group].astype(np.float64),
                                   q[head], optimize=False)
    return out


def _aggregate(qk, top_m, mode, algorithm):
    import numpy as np
    if algorithm in ("token_headmax_topm", "token_max_head_topm"):
        flat = qk.max(axis=2)
    elif algorithm in ("global_token_head_topm", "flat_token_head_topm"):
        flat = qk.reshape(len(qk), -1)
    else:
        raise ValueError(f"Unsupported selector reference algorithm: {algorithm}")
    if mode == 0:
        return flat.max(axis=1).astype(np.float32)
    m = top_m if 0 < top_m <= flat.shape[1] else flat.shape[1]
    values = np.partition(flat, flat.shape[1] - m, axis=1)[:, -m:]
    return values.astype(np.float64).mean(axis=1).astype(np.float32)


def _ranking(scores, native_ids):
    import numpy as np
    return np.lexsort((np.asarray(native_ids), -np.asarray(scores)))


def _top_compare(scores, native_ids, logical, selected, top_n, atol, rtol):
    import numpy as np
    order = _ranking(scores, native_ids)
    n = min(top_n, len(scores))
    expected = [logical[i] for i in order[:n]]
    pos = {c: i for i, c in enumerate(logical)}
    if len(selected) != n or len(set(selected)) != n or any(c not in pos for c in selected):
        raise ValueError("Invalid SSD Top-N diagnostic result")
    cutoff = float(scores[order[n - 1]])
    tolerance = atol + rtol * abs(cutoff)
    required = {logical[i] for i, s in enumerate(scores) if float(s) > cutoff + tolerance}
    compatible = required.issubset(selected) and all(float(scores[pos[c]]) >= cutoff - tolerance for c in selected)
    return {"expected_topn": expected, "ssd_topn": selected,
            "set_equal": set(expected) == set(selected), "order_equal": expected == selected,
            "overlap_fraction": len(set(expected).intersection(selected)) / n,
            "score_compatible_with_topn": compatible, "cutoff_score": cutoff,
            "cutoff_tolerance": tolerance}


def _error(reference, actual, atol, rtol):
    import numpy as np
    delta = np.abs(np.asarray(reference, np.float64) - np.asarray(actual, np.float64))
    if delta.shape != np.asarray(reference).shape or not np.isfinite(delta).all():
        raise ValueError("Invalid/non-finite diagnostic score arrays")
    return {"max_abs": float(delta.max(initial=0)),
            "max_rel": float((delta / np.maximum(np.abs(reference), 1e-20)).max(initial=0)),
            "within_tolerance": bool(np.all(delta <= atol + rtol * np.abs(reference)))}


def analyze_case(root, metadata, case, abi, live, lo, hi, archive, sidecar, atol, rtol):
    import numpy as np
    g = metadata["geometry"]
    logical = case["logical_chunks"]
    count = len(logical)
    if len(set(logical)) != count or any(c < 0 or c >= g["num_chunks"] for c in logical):
        raise ValueError("Invalid logical diagnostic candidate mapping")
    if case["status"] != "complete" or case["candidate_count"] != count:
        raise ValueError("Incomplete diagnostic case")
    if case["schema"] != SCHEMA or case["req_id"] != metadata["req_id"]:
        raise ValueError("Diagnostic case belongs to another request/schema")
    if case["bucket"] != next(b for b in BUCKETS if count <= b):
        raise ValueError("Diagnostic candidate bucket mismatch")
    if case["token_starts"] != [c * g["chunk_size"] for c in logical] or case["token_ends"] != [(c + 1) * g["chunk_size"] for c in logical]:
        raise ValueError("Diagnostic token/chunk mapping mismatch")
    native_ids = case["native"]["aissd_candidate_chunk_ids"]
    if len(native_ids) != count or len(set(native_ids)) != count:
        raise ValueError("Invalid native candidate IDs")
    if len(case["ssd_selected_native_ids"]) != len(case["ssd_selected_logical_chunks"]) or [dict(zip(native_ids, logical))[i] for i in case["ssd_selected_native_ids"]] != case["ssd_selected_logical_chunks"]:
        raise ValueError("Native/logical SSD result mapping mismatch")
    heads, segments, scale = case["num_heads"], metadata["segments"], case["attention_scale"]
    q = _array(root, case["q"]).reshape(heads, g["head_dim"])
    upper_gpu = _array(root, case["upper"]).reshape(count, segments, heads)
    scores_gpu = _array(root, case["scores"]).reshape(count)
    if heads % g["kv_heads"] or heads != int(abi["num_q_heads"]) or scale <= 0:
        raise ValueError("Diagnostic Q/head/attention scale mismatch")
    if not all(np.isfinite(a).all() for a in (q, upper_gpu, scores_gpu, live, lo, hi, archive)):
        raise ValueError("Non-finite diagnostic Q/K/summary")
    # An explicit head loop independently checks the production GQA grouping.
    upper_cpu = np.empty_like(upper_gpu, dtype=np.float64)
    group = heads // g["kv_heads"]
    for h in range(heads):
        kvh = h // group
        lower, higher = lo[logical, :, kvh].astype(np.float64), hi[logical, :, kvh].astype(np.float64)
        upper_cpu[:, :, h] = (np.maximum(q[h], 0) * higher + np.minimum(q[h], 0) * lower).sum(axis=-1)
    scores_cpu = upper_cpu.max(axis=(1, 2)) * scale
    # GPU ranking is checked against its own captured FP32 scores, including ties.
    rank_gpu_expected = np.argsort(-scores_gpu, kind="stable").tolist()
    ranking_ok = case["gpu_ranking_positions"] == rank_gpu_expected
    protected = set(sorted(range(count), key=lambda i: case["token_starts"][i])[:case["keep_first"]])
    if case["keep_recent"]:
        protected.update(sorted(range(count), key=lambda i: case["token_starts"][i])[-case["keep_recent"]:])
    chosen = set(protected)
    for i in rank_gpu_expected:
        if len(chosen) >= min(count, case["coarse_budget"]):
            break
        chosen.add(i)
    routing_ok = case["coarse_positions"] == sorted(chosen)
    references = {}
    raw_max = np.empty(count, np.float32)
    kinfo = abi["buckets"][str(case["bucket"])]
    qscale = float(kinfo["q_block"]["tensor_scale_factor"])
    kscale = float(kinfo["k_flat"]["tensor_scale_factor"])
    qzp, kzp = (int(kinfo[key].get("tensor_zero_point", 0)) for key in ("q_block", "k_flat"))
    qi, q_clamp = _quantize(q, qscale, qzp, query=True)
    k_clamp = {"clamped_low": 0, "clamped_high": 0}
    k_mismatch = 0
    for name in ("fp32_live", "fp32_archive", "int16_expected", "int16_sidecar"):
        references[name] = np.empty(count, np.float32)
    # Limit transient FP64 K and QK buffers independently of prefix length.
    for first in range(0, count, 16):
        last = min(count, first + 16)
        ids = logical[first:last]
        live_qk = _qk(live[ids], q) * scale
        archive_qk = _qk(archive[ids], q) * scale
        raw_max[first:last] = live_qk.max(axis=(1, 2))
        quantized, clamped = _quantize(archive[ids], kscale, kzp)
        for key in k_clamp:
            k_clamp[key] += clamped[key]
        k_mismatch += int(np.count_nonzero(quantized != sidecar[ids]))
        q_center = qi.astype(np.float64) - qzp
        quant_scale = qscale * kscale / math.sqrt(g["head_dim"])
        # _qk normally emits FP32, but INT16 products need scaling after exact
        # FP64 accumulation; calculate each head directly before the FP32 cast.
        int_qks = {}
        for name, values in (("int16_expected", quantized), ("int16_sidecar", sidecar[ids])):
            out = np.empty((last - first, g["chunk_size"], heads), np.float32)
            for h in range(heads):
                k_center = values[:, :, h // group].astype(np.float64) - kzp
                out[:, :, h] = (np.einsum("ntd,d->nt", k_center, q_center[h], optimize=False) * quant_scale).astype(np.float32)
            int_qks[name] = out
        for name, values in (("fp32_live", live_qk), ("fp32_archive", archive_qk), *int_qks.items()):
            references[name][first:last] = _aggregate(values, case["top_m"], case["score_mode_code"], case["selector_algorithm"])
    ssd = case["ssd_selected_logical_chunks"]
    comparisons = {name: _top_compare(values, native_ids, logical, ssd,
                                      case["top_n"], atol, rtol)
                   for name, values in references.items()}
    coarse_ranks = {logical[i]: rank + 1 for rank, i in enumerate(rank_gpu_expected)}
    pos = {c: i for i, c in enumerate(logical)}
    kept = {logical[i] for i in chosen}
    missed = [c for c in ssd if c not in kept]
    score_error = _error(scores_cpu, scores_gpu, atol, rtol)
    upper_error = _error(upper_cpu, upper_gpu, atol, rtol)
    under = np.maximum(raw_max - scores_cpu, 0)
    bound_ok = bool(np.all(under <= atol + rtol * np.abs(raw_max)))
    checks = {"upper_cpu_gpu": upper_error, "score_cpu_gpu": score_error,
              "gpu_stable_ranking_correct": ranking_ok, "protected_selection_correct": routing_ok,
              "live_qk_below_box_bound": bound_ok, "max_bound_underestimate": float(under.max(initial=0))}
    return {"req_id": case["req_id"], "step": case["step"], "layer_id": case["layer_id"],
            "candidate_count": count, "bucket": case["bucket"], "coarse_budget": case["coarse_budget"],
            "checks": checks, "q_clamping": q_clamp, "k_clamping": k_clamp,
            "sidecar_vs_requantized_archive_mismatched_elements": k_mismatch,
            "reference_vs_ssd": comparisons,
            "attention_scale_matches_compiled": math.isclose(scale, 1 / math.sqrt(g["head_dim"]), rel_tol=1e-6),
            "coarse_recall": len(kept.intersection(ssd)) / len(ssd),
            "missed_ssd_chunks": [{"logical_chunk": c, "coarse_rank": coarse_ranks[c],
                                   "coarse_score": float(scores_gpu[pos[c]]),
                                   **{name + "_score": float(values[pos[c]]) for name, values in references.items()}}
                                  for c in missed],
            "coarse_score_exact_ties": count - len(set(map(float, scores_gpu))),
            "candidate_scores": [{"logical_chunk": c, "native_id": int(native_ids[i]),
                                  "coarse_rank": coarse_ranks[c], "kept": c in kept,
                                  "coarse_gpu": float(scores_gpu[i]), "coarse_cpu": float(scores_cpu[i]),
                                  "live_qk_max": float(raw_max[i]),
                                  **{name: float(values[i]) for name, values in references.items()}}
                                 for i, c in enumerate(logical)]}


def analyze(input_dir, *, atol=1e-4, rtol=2e-4):
    import numpy as np
    root = Path(input_dir).resolve()
    if not (root / "run.json").is_file():
        runs = sorted(p.parent for p in root.glob("run-*/run.json"))
        if len(runs) != 1:
            raise ValueError(f"Specify one run directory; found {len(runs)} under {root}")
        root = runs[0]
    run = json.loads((root / "run.json").read_text())
    abi_bytes = (root / "abi.json").read_bytes()
    if run["schema"] != SCHEMA or _sha(abi_bytes) != run["abi_sha256"]:
        raise ValueError("Invalid diagnostic run/ABI snapshot")
    abi = json.loads(abi_bytes)
    _check_abi(abi)
    results, layer_checks, coverage_errors = [], [], []
    request_dirs = sorted(root.glob("request-*/request.json"))
    if not request_dirs:
        raise ValueError("No diagnostic request was captured")
    for request_path in request_dirs:
        request_root = request_path.parent
        metadata = json.loads(request_path.read_text())
        if metadata["schema"] != SCHEMA or metadata["status"] != "ready":
            raise ValueError(f"Incomplete pre-release K capture: {request_root}")
        cases = [json.loads(p.read_text()) for p in sorted(request_root.glob("cases/*.json")) if not p.name.endswith(".pending.json")]
        keys = [(int(c["step"]), int(c["layer_id"])) for c in cases]
        expected = {(s, l) for s in metadata["expected_steps"] for l in metadata["expected_layers"]}
        if len(keys) != len(set(keys)) or set(keys) != expected or list(request_root.glob("cases/*.pending.json")):
            coverage_errors.append({"request": metadata["req_id"], "expected": sorted(expected), "actual": keys})
        g, segments = metadata["geometry"], metadata["segments"]
        for layer in metadata["expected_layers"]:
            arrays = metadata["layers"][str(layer)]
            live, lo, hi = (_array(request_root, arrays[k]) for k in ("live_k", "min", "max"))
            expected_shape = (g["num_chunks"], g["chunk_size"], g["kv_heads"], g["head_dim"])
            if live.shape != expected_shape or lo.shape != (g["num_chunks"], segments, g["kv_heads"], g["head_dim"]) or hi.shape != lo.shape:
                raise ValueError("Diagnostic K/summary shape mismatch")
            segmented = live.reshape(g["num_chunks"], segments, g["chunk_size"] // segments,
                                     g["kv_heads"], g["head_dim"])
            summary_ok = bool(np.array_equal(lo, segmented.min(axis=2)) and np.array_equal(hi, segmented.max(axis=2)))
            layer_cases = [c for c in cases if c["layer_id"] == layer]
            if not layer_cases:
                continue
            buckets = {c["bucket"] for c in layer_cases}
            if len(buckets) != 1:
                raise ValueError("One diagnostic request changed sidecar bucket")
            archive, sidecar = _load_archive(metadata, layer, abi, buckets.pop())
            unequal = live != archive
            token_diff = unequal.any(axis=(2, 3))
            changed_tokens = np.flatnonzero(token_diff.reshape(-1)).tolist()
            tail = metadata.get("recomputed_tail")
            allowed = set() if tail is None else {tail["token_index"]}
            unexpected = [t for t in changed_tokens if t not in allowed]
            layer_checks.append({"req_id": metadata["req_id"], "layer_id": layer,
                                 "summary_exact": summary_ok,
                                 "live_archive_different_elements": int(unequal.sum()),
                                 "live_archive_changed_tokens": changed_tokens,
                                 "unexpected_changed_tokens": unexpected,
                                 "recomputed_tail_token": None if tail is None else tail["token_index"]})
            for case in layer_cases:
                results.append(analyze_case(request_root, metadata, case, abi,
                                            live, lo, hi, archive, sidecar, atol, rtol))
            del live, archive, sidecar
    results.sort(key=lambda c: (c["req_id"], c["step"], c["layer_id"]))
    mechanical_ok = (not coverage_errors and all(x["summary_exact"] and not x["unexpected_changed_tokens"] for x in layer_checks) and
                     all(c["checks"]["upper_cpu_gpu"]["within_tolerance"] and c["checks"]["score_cpu_gpu"]["within_tolerance"] and
                         c["checks"]["gpu_stable_ranking_correct"] and c["checks"]["protected_selection_correct"] and
                         c["checks"]["live_qk_below_box_bound"] for c in results))
    sidecar_ok = all(c["sidecar_vs_requantized_archive_mismatched_elements"] == 0 for c in results)
    selectors_ok = all(c["reference_vs_ssd"]["int16_sidecar"]["score_compatible_with_topn"] for c in results)
    findings = []
    if not mechanical_ok:
        findings.append("Inspect summary/page mapping, independent CPU envelope scores, ranking and capture coverage before changing the algorithm.")
    if not sidecar_ok:
        findings.append("Actual sidecar INT16 values differ from quantizing the captured immutable archive with this ABI.")
    if not selectors_ok:
        findings.append("CPU reference from actual sidecar K and expected quantized Q disagrees with SSD; inspect actual packed Q, NPU output and tie handling.")
    if mechanical_ok and sidecar_ok and selectors_ok and any(c["coarse_recall"] < 1 for c in results):
        findings.append("Evidence points to coarse approximation/ranking quality; this does not establish task accuracy or exclude every hardware issue.")
    return {"schema": SCHEMA + "-report", "status": "ANALYZED" if mechanical_ok and sidecar_ok and selectors_ok else "CHECK_FAILED",
            "run_dir": str(root), "requests": len(request_dirs), "cases": len(results),
            "coverage_errors": coverage_errors, "mechanical_checks_pass": mechanical_ok,
            "sidecar_requantization_matches": sidecar_ok, "int16_reference_score_compatible_with_ssd": selectors_ok,
            "atol": atol, "rtol": rtol, "findings": findings,
            "limitations": ["Expected Q quantization is reconstructed from the captured ABI; actual CMB Q bytes and NPU raw QK output are not captured.",
                            "Diagnostic capture adds synchronization and HOST memory/IO; do not use these timings as performance results.",
                            "ANALYZED means the reference checks passed; coarse recall and task accuracy have separate acceptance criteria."],
            "layer_checks": layer_checks, "case_results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="One diagnostic run directory or its parent containing exactly one run")
    parser.add_argument("--output", required=True)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=2e-4)
    args = parser.parse_args()
    if not all(math.isfinite(x) and x >= 0 for x in (args.atol, args.rtol)):
        parser.error("Tolerances must be finite and nonnegative")
    try:
        report = analyze(args.input, atol=args.atol, rtol=args.rtol)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError) as exc:
        report = {"schema": SCHEMA + "-report", "status": "ERROR", "error": str(exc)}
    _json_write(args.output, report)
    print(json.dumps({k: report[k] for k in ("status", "requests", "cases", "mechanical_checks_pass",
                                            "sidecar_requantization_matches", "int16_reference_score_compatible_with_ssd",
                                            "findings", "error") if k in report}, ensure_ascii=False, indent=2))
    print(f"Saved {args.output}")
    return int(report["status"] != "ANALYZED")


if __name__ == "__main__":
    sys.exit(main())
