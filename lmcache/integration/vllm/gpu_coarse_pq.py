# SPDX-License-Identifier: Apache-2.0
"""Optional per-token PQ codes and inner-product lookup for GPU coarse search.

No Torch/NumPy import at module import time. Runtime operates on live post-RoPE
K before archive readiness/page release. CLI replay reads v1 K/Q captures by
itself, without importing LMCache, vLLM or gpu_coarse_diagnostic.py.
PQ is a lossy proxy for the full-candidate SSD selector, not a recall guarantee.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys
import time

ALGORITHM = "pq_codebook_lookup_topm"
CAPTURE_SCHEMA = "aissd-gpu-coarse-diagnostic-v1"
GLOBAL_ALGORITHMS = ("global_token_head_topm", "flat_token_head_topm")
TOKEN_BATCH = 1024
CHUNK_BATCH = 16
MAX_TRAIN_TOKENS = 16384


def validate_parameters(dim, subspaces=4, codewords=256, iterations=8, train_tokens=4096):
    if any(isinstance(v, bool) or not isinstance(v, int)
           for v in (dim, subspaces, codewords, iterations, train_tokens)):
        raise ValueError("PQ dimensions/training parameters must be integers")
    if (dim < 1 or not 1 <= subspaces <= 128 or dim % subspaces or
            not 2 <= codewords <= 256 or not 1 <= iterations <= 32 or
            not codewords <= train_tokens <= MAX_TRAIN_TOKENS):
        raise ValueError("PQ requires subspaces dividing head_dim, 2..256 codewords, "
                         "1..32 iterations, and codewords <= train_tokens <= 16384")


def summary_bytes(tokens, layers, heads, dim, subspaces=4, codewords=256, train_tokens=4096):
    # UINT8 codes, BF16 books; effective book size shrinks only for short input.
    effective = min(tokens, codewords, train_tokens)
    return int(layers * heads * (tokens * subspaces + effective * dim * 2))


def _bf16_numpy(values):
    import numpy as np
    bits = np.asarray(values, np.float32).copy().view(np.uint32)
    bits = (bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)
    return bits.view(np.float32)


def _training_indices(tokens, maximum):
    count = min(tokens, maximum)
    if count <= 1:
        return [0]
    # Include both endpoints; no Q, SSD selection, or global RNG state is used.
    return [i * (tokens - 1) // (count - 1) for i in range(count)]


def _seed_indices(tokens, words, head, part):
    return random.Random(20261030 + head * 131 + part).sample(range(tokens), words)


def train_numpy(training, subspaces=4, codewords=256, iterations=8):
    """Independent NumPy Lloyd reference, [training-token, KV-head, dim]."""
    import numpy as np
    x = np.asarray(training, np.float32)
    if x.ndim != 3 or any(n <= 0 for n in x.shape) or not np.isfinite(x).all():
        raise ValueError("PQ training K must be finite [token, KV-head, dim]")
    tokens, heads, dim = x.shape
    validate_parameters(dim, subspaces, codewords, iterations, max(codewords, tokens))
    words, width = min(codewords, tokens), dim // subspaces
    books = np.empty((heads, subspaces, words, width), np.float32)
    eye = np.eye(words, dtype=np.float32)
    for h in range(heads):
        for part in range(subspaces):
            data = np.ascontiguousarray(x[:, h, part * width:(part + 1) * width])
            centers = data[_seed_indices(tokens, words, h, part)].copy()
            for _ in range(iterations):
                count = np.zeros(words, np.float32)
                sums = np.zeros_like(centers)
                norms = (centers * centers).sum(axis=-1)
                for start in range(0, tokens, TOKEN_BATCH):
                    batch = data[start:start + TOKEN_BATCH]
                    labels = (norms[None] - 2 * (batch @ centers.T)).argmin(axis=1)
                    membership = eye[labels]
                    count += membership.sum(axis=0)
                    sums += membership.T @ batch
                centers = np.where(count[:, None] > 0,
                                   sums / np.maximum(count[:, None], 1), centers)
            books[h, part] = _bf16_numpy(centers)
    return books


def train_torch(training, subspaces=4, codewords=256, iterations=8):
    """Same bounded Lloyd implementation on the selected Torch device."""
    import torch
    tokens, heads, dim = training.shape
    validate_parameters(dim, subspaces, codewords, iterations, max(codewords, tokens))
    words, width = min(codewords, tokens), dim // subspaces
    books = torch.empty((heads, subspaces, words, width), dtype=torch.bfloat16, device=training.device)
    eye = torch.eye(words, dtype=torch.float32, device=training.device)
    for h in range(heads):
        for part in range(subspaces):
            data = training[:, h, part * width:(part + 1) * width].float().contiguous()
            seeds = torch.tensor(_seed_indices(tokens, words, h, part), dtype=torch.long, device=data.device)
            centers = data.index_select(0, seeds)
            for _ in range(iterations):
                count = torch.zeros((words,), dtype=torch.float32, device=data.device)
                sums = torch.zeros((words, width), dtype=torch.float32, device=data.device)
                norms = centers.square().sum(dim=-1)
                for start in range(0, tokens, TOKEN_BATCH):
                    batch = data[start:start + TOKEN_BATCH]
                    labels = (norms[None] - 2 * torch.matmul(batch, centers.transpose(0, 1))).argmin(dim=1)
                    membership = eye[labels]
                    count.add_(membership.sum(dim=0))
                    sums.add_(torch.matmul(membership.transpose(0, 1), batch))
                centers = torch.where(count[:, None] > 0,
                                      sums / count[:, None].clamp_min(1), centers)
            books[h, part].copy_(centers.to(dtype=torch.bfloat16))
    return books


def encode_numpy(books, keys):
    import numpy as np
    chunks, tokens, heads, dim = keys.shape
    if books.ndim != 4 or books.shape[0] != heads or books.shape[1] * books.shape[3] != dim:
        raise ValueError("PQ codebook/K geometry mismatch")
    subspaces, width = books.shape[1], books.shape[3]
    flat = np.asarray(keys, np.float32).reshape(-1, heads, dim)
    codes = np.empty((len(flat), heads, subspaces), np.uint8)
    for h in range(heads):
        for part in range(subspaces):
            centers = books[h, part].astype(np.float32)
            norms = (centers * centers).sum(axis=-1)
            for start in range(0, len(flat), TOKEN_BATCH):
                batch = flat[start:start + TOKEN_BATCH, h, part * width:(part + 1) * width]
                labels = (norms[None] - 2 * (batch @ centers.T)).argmin(axis=1)
                codes[start:start + len(batch), h, part] = labels.astype(np.uint8)
    return codes.reshape(chunks, tokens, heads, subspaces)


def encode_torch(books, keys):
    import torch
    chunks, tokens, heads, dim = keys.shape
    if books.shape[0] != heads or books.shape[1] * books.shape[3] != dim:
        raise ValueError("PQ codebook/K geometry mismatch")
    subspaces, width = books.shape[1], books.shape[3]
    flat = keys.reshape(-1, heads, dim)
    codes = torch.empty((flat.shape[0], heads, subspaces), dtype=torch.uint8, device=keys.device)
    for h in range(heads):
        for part in range(subspaces):
            centers = books[h, part].float()
            norms = centers.square().sum(dim=-1)
            for start in range(0, flat.shape[0], TOKEN_BATCH):
                batch = flat[start:start + TOKEN_BATCH, h, part * width:(part + 1) * width].float()
                labels = (norms[None] - 2 * torch.matmul(batch, centers.transpose(0, 1))).argmin(dim=1)
                codes[start:start + batch.shape[0], h, part].copy_(labels.to(dtype=torch.uint8))
    return codes.reshape(chunks, tokens, heads, subspaces)


def compress_numpy(keys, subspaces=4, codewords=256, iterations=8, train_tokens=4096):
    import numpy as np
    keys = np.asarray(keys, np.float32)
    if keys.ndim != 4 or any(n <= 0 for n in keys.shape) or not np.isfinite(keys).all():
        raise ValueError("PQ K must be finite [chunk, token, KV-head, dim]")
    validate_parameters(keys.shape[-1], subspaces, codewords, iterations, train_tokens)
    flat = keys.reshape(-1, keys.shape[2], keys.shape[3])
    books = train_numpy(flat[_training_indices(len(flat), train_tokens)], subspaces, codewords, iterations)
    return books, encode_numpy(books, keys)


def _effective_m(top_m, size):
    return top_m if 0 < top_m <= size else size


def _reduce_numpy(values, top_m, score_mode):
    import numpy as np
    flat = values.reshape(len(values), -1)
    if score_mode == 0:
        return flat.max(axis=1)
    take = _effective_m(int(top_m), flat.shape[1])
    if take == flat.shape[1]:
        return flat.mean(axis=1, dtype=np.float32)
    return np.partition(flat, flat.shape[1] - take, axis=1)[:, -take:].mean(axis=1, dtype=np.float32)


def score_numpy(books, codes, query, scale, top_m=8, score_mode=1, logical_chunks=None):
    """LUT inner products, then token x Q-head Top-M separately per chunk."""
    import numpy as np
    q = np.asarray(query, np.float32)
    heads, parts, words, width = books.shape
    if (codes.ndim != 4 or codes.dtype != np.uint8 or tuple(codes.shape[2:]) != (heads, parts) or
            q.ndim != 2 or q.shape[0] <= 0 or q.shape[0] % heads or q.shape[1] != parts * width or
            not math.isfinite(scale) or scale <= 0 or not np.isfinite(q).all() or not np.isfinite(books).all()):
        raise ValueError("Invalid PQ books/codes/Q/scale")
    if codes.size == 0 or int(codes.max()) >= words:
        raise ValueError("PQ code is outside its codebook")
    logical = list(range(len(codes))) if logical_chunks is None else list(logical_chunks)
    if not logical or any(i < 0 or i >= len(codes) for i in logical):
        raise ValueError("PQ logical chunk out of range")
    groups = q.shape[0] // heads
    lut = np.einsum("hgmd,hmkd->hgmk", q.reshape(heads, groups, parts, width), books.astype(np.float32))
    results = []
    for start in range(0, len(logical), CHUNK_BATCH):
        batch = codes[logical[start:start + CHUNK_BATCH]]
        count, tokens = batch.shape[:2]
        values = np.zeros((heads, groups, count * tokens), np.float32)
        for part in range(parts):
            ids = batch[:, :, :, part].transpose(2, 0, 1).reshape(heads, -1).astype(np.int64)
            values += np.take_along_axis(lut[:, :, part], ids[:, None], axis=2)
        values = values.transpose(2, 0, 1).reshape(count, tokens, heads * groups)
        results.append(_reduce_numpy(values, top_m, score_mode) * float(scale))
    return np.concatenate(results).astype(np.float32)


def score_torch(books, codes, query, scale, top_m=8, score_mode=1, logical_chunks=None):
    import torch
    heads, parts, words, width = books.shape
    q = query.detach().reshape(heads, -1, parts, width).float()
    lut = torch.einsum("hgmd,hmkd->hgmk", q, books.float())
    logical = list(range(codes.shape[0])) if logical_chunks is None else list(logical_chunks)
    results = []
    for start in range(0, len(logical), CHUNK_BATCH):
        indices = torch.tensor(logical[start:start + CHUNK_BATCH], dtype=torch.long, device=codes.device)
        batch = codes.index_select(0, indices)
        count, tokens = batch.shape[:2]
        values = torch.zeros((heads, q.shape[1], count * tokens), dtype=torch.float32, device=codes.device)
        for part in range(parts):
            ids = batch[:, :, :, part].permute(2, 0, 1).reshape(heads, -1).to(dtype=torch.long)
            values.add_(lut[:, :, part].gather(2, ids[:, None].expand(-1, q.shape[1], -1)))
        flat = values.permute(2, 0, 1).reshape(count, tokens * heads * q.shape[1])
        if score_mode == 0:
            reduced = flat.amax(dim=1)
        else:
            take = _effective_m(int(top_m), flat.shape[1])
            reduced = flat.mean(dim=1) if take == flat.shape[1] else torch.topk(flat, take, dim=1).values.mean(dim=1)
        results.append(reduced * float(scale))
    return torch.cat(results)


def exact_scores_numpy(keys, query, scale, top_m=8, score_mode=1, logical_chunks=None):
    """Replay-only FP32 diagnostic; never called by the runtime PQ selector."""
    import numpy as np
    heads, dim = keys.shape[2:]
    q = np.asarray(query, np.float32).reshape(heads, -1, dim)
    logical = list(range(len(keys))) if logical_chunks is None else list(logical_chunks)
    result = []
    for start in range(0, len(logical), CHUNK_BATCH):
        batch = keys[logical[start:start + CHUNK_BATCH]]
        values = np.einsum("cthd,hgd->cthg", batch, q)
        result.append(_reduce_numpy(values, top_m, score_mode) * float(scale))
    return np.concatenate(result).astype(np.float32)


def build_layers(kv_caches, block_ids, prefix_tokens, chunk_size, block_size,
                 num_layers, kv_heads, head_dim, options):
    import torch
    validate_parameters(head_dim, options.pq_subspaces, options.pq_codewords,
                        options.pq_iterations, options.pq_train_tokens)
    if prefix_tokens <= 0 or prefix_tokens % chunk_size or chunk_size % block_size:
        raise RuntimeError("PQ archive must contain complete page-aligned chunks")
    ids = [int(x) for x in block_ids[:prefix_tokens // block_size]]
    if len(ids) != prefix_tokens // block_size or any(x <= 0 for x in ids) or len(set(ids)) != len(ids):
        raise RuntimeError("PQ source has missing/null/duplicate GPU pages")
    layers, chunks = {}, prefix_tokens // chunk_size
    samples = _training_indices(prefix_tokens, options.pq_train_tokens)
    with torch.no_grad():
        for name, cache in kv_caches.items():
            match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", str(name))
            if match is None:
                raise RuntimeError(f"Cannot identify PQ KV layer: {name}")
            layer = int(match.group(1))
            if (not 0 <= layer < num_layers or layer in layers or cache.dtype != torch.bfloat16 or
                    cache.ndim != 5 or list(cache.shape[2:]) != [block_size, kv_heads, head_dim] or
                    (cache.shape[0] != 2 and cache.shape[1] != 2)):
                raise RuntimeError(f"Unsupported/duplicate PQ KV layer/layout: {name}")
            paged = cache if cache.shape[1] == 2 else cache.transpose(0, 1)
            if max(ids) >= paged.shape[0]:
                raise RuntimeError("PQ source GPU page is out of range")
            physical = torch.tensor(ids, dtype=torch.long, device=cache.device)
            sample_pages = torch.tensor([i // block_size for i in samples], dtype=torch.long, device=cache.device)
            offsets = torch.tensor([i % block_size for i in samples], dtype=torch.long, device=cache.device)
            training = paged[physical.index_select(0, sample_pages), 0, offsets].detach()
            if not bool(torch.isfinite(training).all().item()):
                raise RuntimeError("Non-finite PQ training K")
            books = train_torch(training, options.pq_subspaces, options.pq_codewords, options.pq_iterations)
            del training
            codes = torch.empty((chunks, chunk_size, kv_heads, options.pq_subspaces), dtype=torch.uint8, device=cache.device)
            per_chunk = chunk_size // block_size
            for first in range(0, chunks, CHUNK_BATCH):
                last = min(first + CHUNK_BATCH, chunks)
                selected = physical[first * per_chunk:last * per_chunk]
                keys = paged[:, 0].index_select(0, selected).reshape(last - first, chunk_size, kv_heads, head_dim)
                if not bool(torch.isfinite(keys).all().item()):
                    raise RuntimeError("Non-finite PQ source K")
                codes[first:last].copy_(encode_torch(books, keys))
            if not bool(torch.isfinite(books).all().item()):
                raise RuntimeError("Non-finite PQ codebook")
            layers[layer] = (books, codes)
    if set(layers) != set(range(num_layers)):
        raise RuntimeError("PQ summary is missing model layers")
    torch.cuda.synchronize(next(iter(layers.values()))[0].device)
    nbytes = sum(t.numel() * t.element_size() for pair in layers.values() for t in pair)
    return layers, nbytes


def _resolve_capture_run(input_dir):
    root = Path(input_dir).resolve()
    if (root / "run.json").is_file():
        return root
    pointer = root / "latest.json"
    if pointer.is_file():
        value = json.loads(pointer.read_text())
        name = value.get("run_dir") if isinstance(value, dict) else None
        if (not isinstance(value, dict) or value.get("schema") != CAPTURE_SCHEMA or
                not isinstance(name, str) or re.fullmatch(r"run-[0-9a-f]{32}", name) is None):
            raise ValueError("Invalid capture latest.json; refusing to select an older run")
        selected = (root / name).resolve()
        if selected.parent != root or not (selected / "run.json").is_file():
            raise ValueError("Latest capture is missing/invalid; refusing to select an older run")
        return selected
    runs = sorted(p.parent for p in root.glob("run-*/run.json"))
    if len(runs) != 1:
        raise ValueError(f"No latest.json: specify one legacy run; found {len(runs)} under {root}")
    return runs[0]


def _capture_array(root, descriptor):
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
    if isinstance(nbytes, bool) or not isinstance(nbytes, int) or nbytes != expected or path.stat().st_size != expected:
        raise ValueError(f"Truncated capture tensor: {path}")
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
        raise ValueError("Protected chunks exceed PQ replay budget")
    for i in ranking:
        if len(selected) >= min(budget, len(logical)):
            break
        selected.add(int(i))
    return selected


def replay(input_dir, *, subspaces=4, codewords=256, iterations=8, train_tokens=4096,
           budgets=(32, 64, 96, 128, 256), checked_budget=128, device="numpy",
           min_mean_recall=.99, min_record_recall=.875):
    import numpy as np
    root = _resolve_capture_run(input_dir)
    if json.loads((root / "run.json").read_text()).get("schema") != CAPTURE_SCHEMA:
        raise ValueError("Unexpected PQ replay capture schema")
    if checked_budget not in budgets or any(b < 1 or b > 512 for b in budgets):
        raise ValueError("PQ budgets must be 1..512 and include the checked budget")
    torch = None
    if device != "numpy":
        import torch
        if device != "cpu" and re.fullmatch(r"cuda(?::\d+)?", device) is None:
            raise ValueError("PQ device must be numpy, cpu or cuda[:index]")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA PQ replay requested but CUDA is unavailable")
    rows, storage = [], []
    for request_path in sorted(root.glob("request-*/request.json")):
        meta = json.loads(request_path.read_text())
        request_root = request_path.parent
        if meta.get("schema") != CAPTURE_SCHEMA or meta.get("status") != "ready":
            raise ValueError("PQ replay requires a complete pre-release K capture")
        g = meta["geometry"]
        validate_parameters(g["head_dim"], subspaces, codewords, iterations, train_tokens)
        cases = [json.loads(p.read_text()) for p in sorted(request_root.glob("cases/*.json"))
                 if not p.name.endswith(".pending.json")]
        expected = {(s, l) for s in meta["expected_steps"] for l in meta["expected_layers"]}
        actual = [(c["step"], c["layer_id"]) for c in cases]
        if (not expected or len(actual) != len(set(actual)) or set(actual) != expected or
                list(request_root.glob("cases/*.pending.json"))):
            raise ValueError("Incomplete/duplicate PQ replay case coverage")
        for layer in meta["expected_layers"]:
            keys = _capture_array(request_root, meta["layers"][str(layer)]["live_k"])
            if (keys.shape != (g["num_chunks"], g["chunk_size"], g["kv_heads"], g["head_dim"]) or
                    not np.isfinite(keys).all()):
                raise ValueError("PQ replay K geometry/finiteness mismatch")
            started = time.perf_counter()
            sample_ids = _training_indices(len(keys) * g["chunk_size"], train_tokens)
            training = keys.reshape(-1, g["kv_heads"], g["head_dim"])[sample_ids]
            if torch is None:
                books = train_numpy(training, subspaces, codewords, iterations)
                codes = encode_numpy(books, keys)
            else:
                with torch.no_grad():
                    data = torch.tensor(training, dtype=torch.bfloat16, device=device)
                    books = train_torch(data, subspaces, codewords, iterations)
                    del data
                    codes = torch.empty((len(keys), g["chunk_size"], g["kv_heads"], subspaces), dtype=torch.uint8, device=device)
                    for first in range(0, len(keys), CHUNK_BATCH):
                        last = min(first + CHUNK_BATCH, len(keys))
                        batch = torch.tensor(keys[first:last], dtype=torch.bfloat16, device=device)
                        codes[first:last].copy_(encode_torch(books, batch))
                if device.startswith("cuda"):
                    torch.cuda.synchronize(device)
            build_seconds = time.perf_counter() - started
            del training
            for c in (c for c in cases if c["layer_id"] == layer):
                if (c.get("schema") != CAPTURE_SCHEMA or c.get("status") != "complete" or
                        c["req_id"] != meta["req_id"] or c["selector_algorithm"] not in GLOBAL_ALGORITHMS):
                    raise ValueError("Incomplete/unsupported PQ replay case")
                logical, reference = c["logical_chunks"], c["ssd_selected_logical_chunks"]
                if (len(logical) != c["candidate_count"] or len(set(logical)) != len(logical) or
                        any(i < 0 or i >= len(keys) for i in logical) or not reference or
                        len(reference) != c["top_n"] or len(set(reference)) != len(reference) or
                        not set(reference).issubset(logical) or
                        c["token_starts"] != [i * g["chunk_size"] for i in logical] or
                        c["token_ends"] != [(i + 1) * g["chunk_size"] for i in logical]):
                    raise ValueError("Invalid PQ candidate/token/SSD-result mapping")
                native = c["native"]["aissd_candidate_chunk_ids"]
                mapping = dict(zip(native, logical))
                if (len(native) != len(logical) or len(mapping) != len(logical) or
                        [mapping[i] for i in c["ssd_selected_native_ids"]] != reference):
                    raise ValueError("PQ native/logical candidate IDs disagree")
                q = _capture_array(request_root, c["q"]).reshape(c["num_heads"], g["head_dim"])
                if (c["num_heads"] % g["kv_heads"] or not np.isfinite(q).all() or
                        not math.isfinite(c["attention_scale"]) or c["attention_scale"] <= 0):
                    raise ValueError("Invalid PQ replay Q/GQA/scale")
                if torch is None:
                    scores = score_numpy(books, codes, q, c["attention_scale"], c["top_m"], c["score_mode_code"], logical)
                else:
                    with torch.no_grad():
                        qi = torch.tensor(q, dtype=torch.float32, device=device)
                        scores = score_torch(books, codes, qi, c["attention_scale"], c["top_m"], c["score_mode_code"], logical).float().cpu().numpy()
                if not np.isfinite(scores).all():
                    raise ValueError("Non-finite PQ replay scores")
                exact = exact_scores_numpy(keys, q, c["attention_scale"], c["top_m"], c["score_mode_code"], logical)
                if not np.isfinite(exact).all():
                    raise ValueError("Non-finite replay FP32 reference scores")
                ranking = np.argsort(-scores, kind="stable").tolist()
                exact_order = np.argsort(-exact, kind="stable")
                exact_rank = np.empty(len(exact), np.int64)
                exact_rank[exact_order] = np.arange(1, len(exact) + 1)
                curves = {}
                for budget in budgets:
                    chosen = _positions(ranking, logical, budget, c["keep_first"], c["keep_recent"])
                    hits = len({logical[i] for i in chosen}.intersection(reference))
                    curves[str(budget)] = {"hits": hits, "reference_count": len(reference),
                                           "recall": hits / len(reference), "all_retained": hits == len(reference)}
                rows.append({"req_id": c["req_id"], "layer_id": layer, "step": c["step"],
                             "candidate_count": len(logical), "build_layer_seconds": build_seconds,
                             "original_algorithm": "segment_minmax_qk_max", "original_segments": meta["segments"],
                             "original_budget": c["coarse_budget"],
                             "original_hits": len({logical[i] for i in c["coarse_positions"]}.intersection(reference)),
                             "ssd_selected_chunks": reference, "recall_by_budget": curves,
                             "top_m": c["top_m"], "score_mode_code": c["score_mode_code"],
                             "mean_absolute_fp32_score_error": float(np.abs(scores - exact).mean()),
                             "max_absolute_fp32_score_error": float(np.abs(scores - exact).max()),
                             "candidate_scores": [{"logical_chunk": logical[i], "score": float(scores[i]),
                                                   "rank": rank + 1, "fp32_live_score": float(exact[i]),
                                                   "fp32_live_rank": int(exact_rank[i])}
                                                  for rank, i in enumerate(ranking)]})
            tokens = len(keys) * g["chunk_size"]
            storage.append({"req_id": meta["req_id"], "layer_id": layer,
                            "training_tokens": len(sample_ids), "effective_codewords": books.shape[2],
                            "code_dtype": "uint8", "codebook_dtype": "bfloat16",
                            "layer_summary_bytes": summary_bytes(tokens, 1, g["kv_heads"], g["head_dim"], subspaces, codewords, train_tokens),
                            "estimated_all_model_layers_bytes": summary_bytes(tokens, g["num_layers"], g["kv_heads"], g["head_dim"], subspaces, codewords, train_tokens)})
            del keys, books, codes
    if not rows:
        raise ValueError("PQ replay has no captured cases")
    aggregate = {}
    for budget in budgets:
        values = [r["recall_by_budget"][str(budget)] for r in rows]
        hits, total = sum(v["hits"] for v in values), sum(v["reference_count"] for v in values)
        aggregate[str(budget)] = {"cases": len(values), "hits": hits, "reference_chunks": total,
                                  "mean_recall": hits / total, "min_record_recall": min(v["recall"] for v in values),
                                  "all_selected_retained_fraction": sum(v["all_retained"] for v in values) / len(values)}
    checked = aggregate[str(checked_budget)]
    passed = checked["mean_recall"] >= min_mean_recall and checked["min_record_recall"] >= min_record_recall
    return {"schema": "aissd-gpu-pq-replay-v1", "status": "ANALYZED", "algorithm": ALGORITHM,
            "run_dir": str(root), "device": device, "subspaces": subspaces, "codewords": codewords,
            "iterations": iterations, "train_tokens": train_tokens, "training_sampling": "uniform_endpoints",
            "cases": len(rows), "requests": len({r["req_id"] for r in rows}),
            "checked_budget": checked_budget, "recall_thresholds_pass": passed,
            "thresholds": {"min_mean_recall": min_mean_recall, "min_record_recall": min_record_recall},
            "recall_by_budget": aggregate, "summary_storage": storage, "case_results": rows,
            "limitations": ["Only captured layers/steps/prompts are evaluated, not full-model task accuracy.",
                            "SSD recall uses stored full-candidate SSD IDs; the SSD/NPU is not rerun.",
                            "FP32 live-K scores are diagnostic only; they do not replace SSD reference IDs.",
                            "CPU/CUDA Lloyd assignments can differ near floating-point boundaries.",
                            "Replay timings and payload bytes are not serving latency or peak GPU memory."]}


def self_test(device="numpy"):
    import numpy as np
    keys = np.empty((3, 8, 2, 8), np.float32)
    template = np.array([[-4, 2, 0, 1, 4, -2, 1, 0], [2, 0, -4, 1, 0, 4, -2, 1],
                         [0, -4, 2, 1, -2, 1, 0, 4], [4, 0, 1, -2, 2, 0, -4, 1]], np.float32)
    for c in range(3):
        for t in range(8):
            keys[c, t, 0] = template[(c + t) % 4]
            keys[c, t, 1] = template[(2 * c + t) % 4] * 2
    books, codes = compress_numpy(keys, 2, 24, 2, 24)
    q = (np.arange(48, dtype=np.float32).reshape(6, 8) - 23) / 8
    expected = exact_scores_numpy(keys, q, .25, 5)
    actual = score_numpy(books, codes, q, .25, 5)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
    if device != "numpy":
        import torch
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA PQ self-test requested but CUDA is unavailable")
        with torch.no_grad():
            kt = torch.tensor(keys, dtype=torch.bfloat16, device=device)
            bt = train_torch(kt.reshape(-1, 2, 8), 2, 24, 2)
            ct = encode_torch(bt, kt)
            np.testing.assert_allclose(bt.float().cpu().numpy(), books, rtol=1e-6, atol=1e-6)
            np.testing.assert_array_equal(ct.cpu().numpy(), codes)
            st = score_torch(bt, ct, torch.tensor(q, dtype=torch.float32, device=device), .25, 5)
            np.testing.assert_allclose(st.float().cpu().numpy(), expected, rtol=2e-5, atol=2e-5)
    return {"status": "PASS", "device": device, "per_token_codes": True,
            "lut_vs_full_qk": True, "torch_runtime_parity": device != "numpy"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Existing capture parent or explicit run directory")
    parser.add_argument("--output", help="PQ replay report path")
    parser.add_argument("--device", default="numpy", help="numpy (no Torch), cpu, or cuda[:index]")
    parser.add_argument("--subspaces", type=int, default=4)
    parser.add_argument("--codewords", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--train-tokens", type=int, default=4096)
    parser.add_argument("--budgets", default="32,64,96,128,256")
    parser.add_argument("--budget", type=int, default=128)
    parser.add_argument("--min-mean-recall", type=float, default=.99)
    parser.add_argument("--min-record-recall", type=float, default=.875)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.device not in ("numpy", "cpu") and re.fullmatch(r"cuda(?::\d+)?", args.device) is None:
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
            report = replay(args.input, subspaces=args.subspaces, codewords=args.codewords,
                            iterations=args.iterations, train_tokens=args.train_tokens, device=args.device,
                            budgets=budgets, checked_budget=args.budget,
                            min_mean_recall=args.min_mean_recall, min_record_recall=args.min_record_recall)
    except (ValueError, OSError, RuntimeError, KeyError, TypeError, ImportError, AssertionError) as exc:
        report = {"status": "ERROR", "error": str(exc)}
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)
    print(json.dumps({k: v for k, v in report.items() if k not in ("case_results", "summary_storage")},
                     indent=2, ensure_ascii=False, allow_nan=False))
    return int(report["status"] == "ERROR" or report.get("recall_thresholds_pass") is False)


if __name__ == "__main__":
    sys.exit(main())
