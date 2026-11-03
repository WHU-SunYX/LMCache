# SPDX-License-Identifier: Apache-2.0
"""Fixed-size post-RoPE K clusters and outliers; no import-time Torch/NumPy.

Runtime uses Torch on the source GPU. The standalone CLI replays saved raw
K/Q captures, without initializing vLLM, LMCache, GDS or the SSD service.
Capture reading is self-contained; no diagnostic module is imported.
Centroid populations preserve token multiplicity in the Top-M estimator.
This is an approximate selector, not a bound or a recall guarantee.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

ALGORITHM = "cluster_centroid_outlier_topm"
CAPTURE_SCHEMA = "aissd-gpu-coarse-diagnostic-v1"
GLOBAL_ALGORITHMS = ("global_token_head_topm", "flat_token_head_topm")
BATCH_CHUNKS = 16


def validate_parameters(tokens, centroids, outliers, iterations):
    values = (tokens, centroids, outliers, iterations)
    if any(isinstance(v, bool) or not isinstance(v, int) for v in values):
        raise ValueError("Cluster sizes/iterations must be integers")
    if not (1 <= centroids <= 8 and 0 <= outliers <= 8 and
            centroids + outliers <= 8 and centroids + outliers <= tokens and
            1 <= iterations <= 16 and tokens <= 32767):
        raise ValueError("Cluster requires 1..8 centroids, 0..8 outliers, at most "
                         "8 total representatives <= chunk_size, 1..16 iterations")


def summary_bytes(chunks, layers, heads, dim, centroids=2, outliers=2):
    # BF16 vectors and INT16 population counts; no assignments/full K retained.
    return int(chunks * layers * heads * (centroids + outliers) * (dim * 2 + 2))


def _bf16_numpy(value):
    import numpy as np
    x = np.asarray(value, dtype=np.float32).copy()
    bits = x.view(np.uint32)
    bits = (bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)
    return bits.view(np.float32)


def summarize_numpy(keys, centroids=2, outliers=2, iterations=4):
    """CPU reference. Input [chunk, token, KV-head, dim], output [chunk, head, rep]."""
    import numpy as np
    k = np.asarray(keys, dtype=np.float32)
    if k.ndim != 4 or any(n <= 0 for n in k.shape) or not np.isfinite(k).all():
        raise ValueError("Cluster K must be finite [chunk, token, KV-head, dim]")
    chunks, tokens, heads, dim = k.shape
    validate_parameters(tokens, centroids, outliers, iterations)
    x = k.transpose(0, 2, 1, 3).reshape(chunks * heads, tokens, dim).copy()
    batch = len(x)
    row = np.arange(batch)
    # Farthest-from-mean, then farthest-point seeds. First-index ties are fixed.
    seed = np.argmax(((x - x.mean(axis=1, keepdims=True)) ** 2).sum(axis=-1), axis=1)
    centers = np.empty((batch, centroids, dim), np.float32)
    nearest = np.full((batch, tokens), np.inf, np.float32)
    for i in range(centroids):
        centers[:, i] = x[row, seed]
        nearest = np.minimum(nearest, ((x - centers[:, i, None]) ** 2).sum(axis=-1))
        seed = np.argmax(nearest, axis=1)
    eye = np.eye(centroids, dtype=np.float32)
    for _ in range(iterations):
        # ||x||^2 cancels in argmin, avoiding [batch, token, centroid, dim].
        metric = (centers ** 2).sum(axis=-1)[:, None] - 2 * np.matmul(x, centers.transpose(0, 2, 1))
        labels = metric.argmin(axis=-1)
        membership = eye[labels]
        count = membership.sum(axis=1)
        sums = np.matmul(membership.transpose(0, 2, 1), x)
        centers = np.where(count[..., None] > 0, sums / np.maximum(count[..., None], 1), centers)
    metric = (centers ** 2).sum(axis=-1)[:, None] - 2 * np.matmul(x, centers.transpose(0, 2, 1))
    labels = metric.argmin(axis=-1)
    residual = ((x - centers[row[:, None], labels]) ** 2).sum(axis=-1)
    ids = np.argsort(-residual, axis=1, kind="stable")[:, :outliers]
    active = np.ones((batch, tokens), np.float32)
    active[row[:, None], ids] = 0
    membership = eye[labels] * active[..., None]
    count = membership.sum(axis=1)
    sums = np.matmul(membership.transpose(0, 2, 1), x)
    # Empty centers have zero population and are never used for scoring.
    means = sums / np.maximum(count[..., None], 1)
    vectors = np.concatenate((means, x[row[:, None], ids]), axis=1)
    weights = np.concatenate((count, np.ones((batch, outliers), np.float32)), axis=1)
    return (_bf16_numpy(vectors).reshape(chunks, heads, centroids + outliers, dim),
            weights.astype(np.int16).reshape(chunks, heads, centroids + outliers))


def summarize_torch(keys, centroids=2, outliers=2, iterations=4):
    """Bounded-batch GPU implementation; input is live post-RoPE BF16 K."""
    import torch
    chunks, tokens, heads, dim = keys.shape
    validate_parameters(tokens, centroids, outliers, iterations)
    x = keys.permute(0, 2, 1, 3).reshape(chunks * heads, tokens, dim).float()
    batch = x.shape[0]
    seed = ((x - x.mean(dim=1, keepdim=True)).square().sum(dim=-1)).argmax(dim=1)
    centers = torch.empty((batch, centroids, dim), dtype=torch.float32, device=x.device)
    nearest = torch.full((batch, tokens), float("inf"), dtype=torch.float32, device=x.device)
    for i in range(centroids):
        point = x.gather(1, seed[:, None, None].expand(-1, 1, dim))
        centers[:, i:i + 1].copy_(point)
        nearest = torch.minimum(nearest, (x - point).square().sum(dim=-1))
        seed = nearest.argmax(dim=1)
    eye = torch.eye(centroids, dtype=torch.float32, device=x.device)
    for _ in range(iterations):
        metric = centers.square().sum(dim=-1)[:, None] - 2 * torch.bmm(x, centers.transpose(1, 2))
        labels = metric.argmin(dim=-1)
        membership = eye[labels]
        count = membership.sum(dim=1)
        sums = torch.bmm(membership.transpose(1, 2), x)
        centers = torch.where(count[..., None] > 0, sums / count[..., None].clamp_min(1), centers)
    metric = centers.square().sum(dim=-1)[:, None] - 2 * torch.bmm(x, centers.transpose(1, 2))
    labels = metric.argmin(dim=-1)
    matched = centers.gather(1, labels[..., None].expand(-1, -1, dim))
    residual = (x - matched).square().sum(dim=-1)
    ids = torch.argsort(residual, dim=1, descending=True, stable=True)[:, :outliers]
    active = torch.ones((batch, tokens), dtype=torch.float32, device=x.device)
    active.scatter_(1, ids, 0)
    membership = eye[labels] * active[..., None]
    count = membership.sum(dim=1)
    sums = torch.bmm(membership.transpose(1, 2), x)
    means = sums / count[..., None].clamp_min(1)
    unusual = x.gather(1, ids[..., None].expand(-1, -1, dim))
    vectors = torch.cat((means, unusual), dim=1).to(dtype=torch.bfloat16)
    weights = torch.cat((count, torch.ones((batch, outliers), dtype=torch.float32, device=x.device)), dim=1)
    return (vectors.reshape(chunks, heads, centroids + outliers, dim),
            weights.to(dtype=torch.int16).reshape(chunks, heads, centroids + outliers))


def _effective_m(top_m, tokens, heads):
    total = tokens * heads
    return top_m if 0 < top_m <= total else total


def score_numpy(vectors, populations, query, scale, top_m=8, score_mode=1):
    """Population-weighted Top-M across tokens x Q-heads within each chunk."""
    import numpy as np
    v, p, q = (np.asarray(x) for x in (vectors, populations, query))
    if (v.ndim != 4 or any(n <= 0 for n in v.shape) or p.shape != v.shape[:-1] or
            q.ndim != 2 or q.shape[0] <= 0 or q.shape[1] != v.shape[-1] or
            not np.issubdtype(p.dtype, np.integer) or not math.isfinite(scale) or scale <= 0):
        raise ValueError("Invalid cluster vectors/populations/Q shape")
    n, kvh, reps, dim = v.shape
    if q.shape[0] % kvh or not np.isfinite(v).all() or not np.isfinite(q).all():
        raise ValueError("Cluster Q must be finite and match GQA groups")
    totals = p.sum(axis=-1)
    if (p < 0).any() or not (totals == totals[0, 0]).all() or totals[0, 0] <= 0:
        raise ValueError("Invalid cluster token populations")
    group = q.shape[0] // kvh
    scores = np.einsum("chrd,hgd->chrg", v.astype(np.float32), q.astype(np.float32).reshape(kvh, group, dim))
    weights = np.broadcast_to(p[..., None], scores.shape).reshape(n, -1).astype(np.int32)
    flat = scores.reshape(n, -1)
    order = np.argsort(-np.where(weights > 0, flat, -np.inf), axis=1, kind="stable")
    vals = np.take_along_axis(flat, order, axis=1)
    w = np.take_along_axis(weights, order, axis=1)
    if score_mode == 0:
        result = vals[:, 0]
    else:
        m = _effective_m(int(top_m), int(totals[0, 0]), q.shape[0])
        end = w.cumsum(axis=1)
        take = np.clip(m - (end - w), 0, None)
        take = np.minimum(take, w)
        result = (vals * take).sum(axis=1, dtype=np.float32) / m
    return result.astype(np.float32) * float(scale)


def score_torch(vectors, populations, query, scale, tokens, top_m=8, score_mode=1):
    import torch
    n, kvh, reps, dim = vectors.shape
    q = query.detach().reshape(kvh, -1, dim).float()
    scores = torch.einsum("chrd,hgd->chrg", vectors.float(), q)
    weights = populations[..., None].expand(-1, -1, -1, q.shape[1]).reshape(n, -1).to(dtype=torch.int32)
    flat = scores.reshape(n, -1)
    masked = torch.where(weights > 0, flat, torch.full_like(flat, -float("inf")))
    order = torch.argsort(masked, dim=1, descending=True, stable=True)
    values, w = flat.gather(1, order), weights.gather(1, order)
    if score_mode == 0:
        result = values[:, 0]
    else:
        m = _effective_m(int(top_m), tokens, kvh * q.shape[1])
        end = w.cumsum(dim=1)
        take = torch.minimum((m - (end - w)).clamp_min(0), w)
        result = (values * take).sum(dim=1) / m
    return result * float(scale)


def build_layers(kv_caches, block_ids, prefix_tokens, chunk_size, block_size,
                 num_layers, kv_heads, head_dim, options):
    import torch
    if prefix_tokens <= 0 or prefix_tokens % chunk_size or chunk_size % block_size:
        raise RuntimeError("Cluster archive must contain complete page-aligned chunks")
    validate_parameters(chunk_size, options.cluster_centroids, options.cluster_outliers, options.cluster_iterations)
    chunks = prefix_tokens // chunk_size
    ids = [int(i) for i in block_ids[:prefix_tokens // block_size]]
    if len(ids) != prefix_tokens // block_size or any(i <= 0 for i in ids) or len(set(ids)) != len(ids):
        raise RuntimeError("Cluster source has missing/null/duplicate GPU pages")
    layers = {}
    with torch.no_grad():
        for name, cache in kv_caches.items():
            match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", str(name))
            if match is None:
                raise RuntimeError(f"Cannot identify cluster KV layer: {name}")
            layer = int(match.group(1))
            if (not 0 <= layer < num_layers or layer in layers or cache.dtype != torch.bfloat16 or
                    cache.ndim != 5 or list(cache.shape[2:]) != [block_size, kv_heads, head_dim] or
                    (cache.shape[0] != 2 and cache.shape[1] != 2)):
                raise RuntimeError(f"Unsupported/duplicate cluster KV layer/layout: {name}")
            paged = cache if cache.shape[1] == 2 else cache.transpose(0, 1)
            if max(ids) >= paged.shape[0]:
                raise RuntimeError("Cluster source GPU page is out of range")
            page_ids = torch.tensor(ids, dtype=torch.long, device=cache.device)
            reps = options.cluster_centroids + options.cluster_outliers
            vectors = torch.empty((chunks, kv_heads, reps, head_dim), dtype=torch.bfloat16, device=cache.device)
            populations = torch.empty((chunks, kv_heads, reps), dtype=torch.int16, device=cache.device)
            per_chunk = chunk_size // block_size
            for first in range(0, chunks, BATCH_CHUNKS):
                last = min(chunks, first + BATCH_CHUNKS)
                selected = page_ids[first * per_chunk:last * per_chunk]
                k = paged[:, 0].index_select(0, selected).reshape(last - first, chunk_size, kv_heads, head_dim)
                v, p = summarize_torch(k, options.cluster_centroids, options.cluster_outliers, options.cluster_iterations)
                vectors[first:last].copy_(v)
                populations[first:last].copy_(p)
            if (not bool(torch.isfinite(vectors).all().item()) or
                    not bool((populations.sum(dim=-1) == chunk_size).all().item())):
                raise RuntimeError("Non-finite cluster summary or invalid token populations")
            layers[layer] = (vectors, populations)
    if set(layers) != set(range(num_layers)):
        raise RuntimeError("Cluster summary is missing model layers")
    torch.cuda.synchronize(next(iter(layers.values()))[0].device)
    return layers, sum(t.numel() * t.element_size() for pair in layers.values() for t in pair)


def _resolve_capture_run(input_dir):
    """Read the capture format without depending on diagnostic private APIs."""
    root = Path(input_dir).resolve()
    if (root / "run.json").is_file():
        return root
    pointer = root / "latest.json"
    if pointer.is_file():
        value = json.loads(pointer.read_text())
        if not isinstance(value, dict):
            raise ValueError("Invalid capture latest.json; refusing to select an older run")
        name = value.get("run_dir")
        if (value.get("schema") != CAPTURE_SCHEMA or not isinstance(name, str) or
                re.fullmatch(r"run-[0-9a-f]{32}", name) is None):
            raise ValueError("Invalid capture latest.json; refusing to select an older run")
        selected = (root / name).resolve()
        if selected.parent != root or not (selected / "run.json").is_file():
            raise ValueError("Latest capture run is missing/invalid; refusing to select an older run")
        return selected
    # Older captures may have a single run and no latest.json pointer.
    runs = sorted(p.parent for p in root.glob("run-*/run.json"))
    if len(runs) != 1:
        raise ValueError(f"No latest.json: specify one legacy run; found {len(runs)} under {root}")
    return runs[0]


def _capture_array(root, descriptor):
    """Load a v1 BF16/FP32 snapshot after path, size and SHA256 validation."""
    import numpy as np
    root = Path(root).resolve()
    shape, dtype = descriptor["shape"], descriptor["dtype"]
    if (not isinstance(shape, list) or not shape or
            any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in shape)):
        raise ValueError("Invalid capture tensor shape")
    if dtype not in ("bfloat16", "float32"):
        raise ValueError("Unsupported captured dtype")
    filename = descriptor["file"]
    if not isinstance(filename, str) or not filename:
        raise ValueError("Invalid capture tensor filename")
    path = (root / filename).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Capture tensor path escapes its request")
    expected = math.prod(shape) * (2 if dtype == "bfloat16" else 4)
    nbytes, checksum = descriptor["nbytes"], descriptor["sha256"]
    if (isinstance(nbytes, bool) or not isinstance(nbytes, int) or
            expected != nbytes or path.stat().st_size != expected):
        raise ValueError(f"Truncated capture tensor: {path}")
    if not isinstance(checksum, str) or re.fullmatch(r"[0-9a-f]{64}", checksum) is None:
        raise ValueError(f"Invalid capture tensor checksum: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != checksum:
        raise ValueError(f"Capture tensor checksum mismatch: {path}")
    values = np.fromfile(path, dtype="<u2" if dtype == "bfloat16" else "<f4")
    if values.size != math.prod(shape):
        raise ValueError(f"Capture tensor changed during read: {path}")
    if dtype == "bfloat16":
        values = (values.astype(np.uint32) << 16).view(np.float32)
    return values.reshape(shape)


def _positions(ranking, logical, budget, first, recent):
    chronological = sorted(range(len(logical)), key=lambda i: logical[i])
    selected = set(chronological[:first])
    if recent:
        selected.update(chronological[-recent:])
    if len(selected) > min(budget, len(logical)):
        raise ValueError("Protected chunks exceed replay budget")
    for i in ranking:
        if len(selected) >= min(budget, len(logical)):
            break
        selected.add(int(i))
    return selected


def replay(input_dir, *, centroids=2, outliers=2, iterations=4,
           budgets=(32, 64, 96, 128, 256), device="numpy",
           min_mean_recall=.99, min_record_recall=.875, checked_budget=128):
    import numpy as np
    root = _resolve_capture_run(input_dir)
    run = json.loads((root / "run.json").read_text())
    if run.get("schema") != CAPTURE_SCHEMA:
        raise ValueError("Unexpected replay capture schema")
    if checked_budget not in budgets or any(b < 1 or b > 512 for b in budgets):
        raise ValueError("Replay budget must be 1..512 and include the checked budget")
    torch = None
    if device != "numpy":
        import torch
        if device != "cpu" and not device.startswith("cuda"):
            raise ValueError("Replay device must be numpy, cpu or cuda[:index]")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA replay requested but CUDA is unavailable")
    rows, scopes, coverage_errors = [], [], []
    for request_path in sorted(root.glob("request-*/request.json")):
        meta = json.loads(request_path.read_text())
        request_root = request_path.parent
        if meta.get("schema") != CAPTURE_SCHEMA or meta.get("status") != "ready":
            raise ValueError("Replay requires a complete pre-release K capture")
        g = meta["geometry"]
        validate_parameters(g["chunk_size"], centroids, outliers, iterations)
        cases = [json.loads(p.read_text()) for p in sorted(request_root.glob("cases/*.json"))
                 if not p.name.endswith(".pending.json")]
        expected = {(s, l) for s in meta["expected_steps"] for l in meta["expected_layers"]}
        actual = [(c["step"], c["layer_id"]) for c in cases]
        if (len(actual) != len(set(actual)) or set(actual) != expected or
                list(request_root.glob("cases/*.pending.json"))):
            coverage_errors.append(meta["req_id"])
        for layer in meta["expected_layers"]:
            keys = _capture_array(request_root, meta["layers"][str(layer)]["live_k"])
            if keys.shape != (g["num_chunks"], g["chunk_size"], g["kv_heads"], g["head_dim"]):
                raise ValueError("Replay K geometry mismatch")
            if not np.isfinite(keys).all():
                raise ValueError("Non-finite replay K")
            reps = centroids + outliers
            if torch is None:
                vectors = np.empty((len(keys), g["kv_heads"], reps, g["head_dim"]), np.float32)
                populations = np.empty(vectors.shape[:-1], np.int16)
            else:
                vectors = torch.empty((len(keys), g["kv_heads"], reps, g["head_dim"]), dtype=torch.bfloat16, device=device)
                populations = torch.empty(vectors.shape[:-1], dtype=torch.int16, device=device)
            started = time.perf_counter()
            for first in range(0, len(keys), BATCH_CHUNKS):
                last = min(len(keys), first + BATCH_CHUNKS)
                if torch is None:
                    v, p = summarize_numpy(keys[first:last], centroids, outliers, iterations)
                else:
                    with torch.no_grad():
                        k = torch.tensor(keys[first:last], dtype=torch.bfloat16, device=device)
                        v, p = summarize_torch(k, centroids, outliers, iterations)
                vectors[first:last], populations[first:last] = v, p
            if torch is not None and device.startswith("cuda"):
                torch.cuda.synchronize(device)
            build_s = time.perf_counter() - started
            for c in (c for c in cases if c["layer_id"] == layer):
                if (c.get("status") != "complete" or c.get("schema") != CAPTURE_SCHEMA or
                        c["req_id"] != meta["req_id"] or c["selector_algorithm"] not in GLOBAL_ALGORITHMS):
                    raise ValueError("Incomplete/unsupported replay case")
                logical = c["logical_chunks"]
                reference = c["ssd_selected_logical_chunks"]
                if (len(set(logical)) != len(logical) or len(logical) != c["candidate_count"] or
                        any(i < 0 or i >= len(keys) for i in logical) or
                        not reference or len(reference) != c["top_n"] or
                        len(set(reference)) != len(reference) or not set(reference).issubset(logical) or
                        c["token_starts"] != [i * g["chunk_size"] for i in logical] or
                        c["token_ends"] != [(i + 1) * g["chunk_size"] for i in logical]):
                    raise ValueError("Invalid replay candidate/token/SSD-result mapping")
                native_ids = c["native"]["aissd_candidate_chunk_ids"]
                mapping = dict(zip(native_ids, logical))
                if len(mapping) != len(logical) or [mapping[i] for i in c["ssd_selected_native_ids"]] != reference:
                    raise ValueError("Replay native/logical IDs disagree")
                q = _capture_array(request_root, c["q"]).reshape(c["num_heads"], g["head_dim"])
                if not math.isfinite(c["attention_scale"]) or c["attention_scale"] <= 0:
                    raise ValueError("Invalid replay attention scale")
                if torch is None:
                    scores = score_numpy(vectors[logical], populations[logical], q, c["attention_scale"], c["top_m"], c["score_mode_code"])
                else:
                    with torch.no_grad():
                        qi = torch.tensor(q, dtype=torch.float32, device=device)
                        ids = torch.tensor(logical, dtype=torch.long, device=device)
                        scores = score_torch(vectors.index_select(0, ids), populations.index_select(0, ids), qi,
                                             c["attention_scale"], g["chunk_size"], c["top_m"], c["score_mode_code"])
                        scores = scores.float().cpu().numpy()
                if not np.isfinite(scores).all():
                    raise ValueError("Non-finite cluster replay scores")
                ranking = np.argsort(-scores, kind="stable").tolist()
                recall = {}
                for budget in budgets:
                    pos = _positions(ranking, logical, budget, c["keep_first"], c["keep_recent"])
                    kept = {logical[i] for i in pos}
                    hit = len(kept.intersection(reference))
                    recall[str(budget)] = {"hits": hit, "reference_count": len(reference),
                                           "recall": hit / len(reference), "all_retained": hit == len(reference)}
                coarse = set(c["coarse_positions"])
                baseline_hits = len({logical[i] for i in coarse}.intersection(reference))
                rows.append({"req_id": c["req_id"], "layer_id": layer, "step": c["step"],
                             "candidate_count": len(logical), "build_layer_seconds": build_s,
                             "original_algorithm": "segment_minmax_qk_max", "original_segments": meta["segments"],
                             "original_budget": c["coarse_budget"], "original_hits": baseline_hits,
                             "ssd_selected_chunks": reference, "recall_by_budget": recall,
                             "candidate_scores": [{"logical_chunk": logical[i], "score": float(scores[i]),
                                                   "rank": rank + 1} for rank, i in enumerate(ranking)]})
            scopes.append({"req_id": meta["req_id"], "layer_id": layer,
                           "layer_summary_bytes": summary_bytes(len(keys), 1, g["kv_heads"], g["head_dim"], centroids, outliers),
                           "estimated_all_model_layers_bytes": summary_bytes(len(keys), g["num_layers"], g["kv_heads"], g["head_dim"], centroids, outliers)})
            del vectors, populations, keys
    if not rows or coverage_errors:
        raise ValueError(f"Replay has no cases or incomplete coverage: {coverage_errors}")
    aggregate = {}
    for b in budgets:
        values = [r["recall_by_budget"][str(b)] for r in rows]
        hit, total = sum(x["hits"] for x in values), sum(x["reference_count"] for x in values)
        aggregate[str(b)] = {"cases": len(values), "hits": hit, "reference_chunks": total,
                             "mean_recall": hit / total, "min_record_recall": min(x["recall"] for x in values),
                             "all_selected_retained_fraction": sum(x["all_retained"] for x in values) / len(values)}
    checked = aggregate[str(checked_budget)]
    passes = checked["mean_recall"] >= min_mean_recall and checked["min_record_recall"] >= min_record_recall
    return {"schema": "aissd-gpu-cluster-replay-v1", "status": "ANALYZED", "algorithm": ALGORITHM,
            "run_dir": str(root), "device": device, "centroids": centroids, "outliers": outliers,
            "iterations": iterations, "cases": len(rows), "requests": len({r["req_id"] for r in rows}),
            "checked_budget": checked_budget, "recall_thresholds_pass": passes,
            "thresholds": {"min_mean_recall": min_mean_recall, "min_record_recall": min_record_recall},
            "recall_by_budget": aggregate, "summary_storage": scopes, "case_results": rows,
            "limitations": ["Saved cases cover only captured layers/steps/prompts, not all 40 layers or task accuracy.",
                            "This reads captured raw K/Q and stored SSD IDs only; it does not rerun the SSD/NPU reference.",
                            "CPU and CUDA clustering can differ near floating-point assignment ties; CUDA replay uses the runtime implementation.",
                            "Build timings include replay transfers and are not serving latency or peak-memory measurements."]}


def self_test(device="numpy"):
    """Small known-answer tests, plus actual Torch parity when requested."""
    import numpy as np
    x = np.zeros((3, 8, 2, 4), np.float32)
    x[0, :4] = 2
    x[0, 4:] = -2
    x[1, :, 0] = np.arange(8)[:, None]
    x[1, :, 1] = -np.arange(8)[:, None]
    x[2, 7] = 24
    v, p = summarize_numpy(x, 2, 2, 4)
    assert (p.sum(axis=-1) == 8).all() and (p >= 0).all()
    assert (p[..., -2:] == 1).all()
    q = np.array([[1, -2, 3, -1], [-1, 2, -1, 3], [2, 1, -1, 1], [-1, -2, 1, 2]], np.float32)
    scores = score_numpy(v, p, q, .5, 3)
    # Independent explicit multiset, preserving per-head GQA mapping.
    expected = []
    for c in range(len(x)):
        expanded = []
        for h in range(len(q)):
            for r in range(4):
                expanded.extend([float(np.dot(q[h], v[c, h // 2, r]))] * int(p[c, h // 2, r]))
        expected.append(sum(sorted(expanded, reverse=True)[:3]) / 3 * .5)
    np.testing.assert_allclose(scores, expected, rtol=2e-6, atol=2e-6)
    if device != "numpy":
        import torch
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA self-test requested but CUDA is unavailable")
        with torch.no_grad():
            vt, pt = summarize_torch(torch.tensor(x, dtype=torch.bfloat16, device=device), 2, 2, 4)
            np.testing.assert_array_equal(pt.cpu().numpy(), p)
            np.testing.assert_allclose(vt.float().cpu().numpy(), v, rtol=1e-6, atol=1e-6)
            st = score_torch(vt, pt, torch.tensor(q, device=device), .5, 8, 3)
            np.testing.assert_allclose(st.float().cpu().numpy(), expected, rtol=2e-5, atol=2e-5)
    return {"status": "PASS", "device": device, "population_conservation": True,
            "weighted_topm_reference": True, "torch_runtime_parity": device != "numpy"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Existing fixed capture parent, or one explicit run directory")
    parser.add_argument("--output", help="Replay report path")
    parser.add_argument("--device", default="numpy", help="numpy (no Torch), cpu (Torch), or cuda[:index]")
    parser.add_argument("--centroids", type=int, default=2)
    parser.add_argument("--outliers", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--budgets", default="32,64,96,128,256")
    parser.add_argument("--budget", type=int, default=128)
    parser.add_argument("--min-mean-recall", type=float, default=.99)
    parser.add_argument("--min-record-recall", type=float, default=.875)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.device not in ("numpy", "cpu") and not re.fullmatch(r"cuda(?::\d+)?", args.device):
        parser.error("--device must be numpy, cpu or cuda[:index]")
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in (args.min_mean_recall, args.min_record_recall)):
        parser.error("Recall thresholds must be finite in [0,1]")
    if not args.self_test and (not args.input or not args.output):
        parser.error("--input and --output are required for replay")
    try:
        if args.self_test:
            report = self_test(args.device)
        else:
            budgets = tuple(sorted({args.budget, *(int(x) for x in args.budgets.split(","))}))
            report = replay(args.input, centroids=args.centroids, outliers=args.outliers,
                            iterations=args.iterations, budgets=budgets, device=args.device,
                            min_mean_recall=args.min_mean_recall, min_record_recall=args.min_record_recall,
                            checked_budget=args.budget)
    except (ValueError, OSError, RuntimeError, KeyError, TypeError, ImportError, AssertionError) as exc:
        report = {"status": "ERROR", "error": str(exc)}
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        temporary.replace(path)
    print(json.dumps({k: v for k, v in report.items() if k not in ("case_results", "summary_storage")},
                     indent=2, ensure_ascii=False, allow_nan=False))
    return int(report["status"] == "ERROR" or report.get("recall_thresholds_pass") is False)


if __name__ == "__main__":
    sys.exit(main())
