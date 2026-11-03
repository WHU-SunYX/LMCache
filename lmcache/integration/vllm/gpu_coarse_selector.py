# SPDX-License-Identifier: Apache-2.0
"""Optional GPU candidate pruning before the existing SSD selector.

The default path does not import this module. Summaries are request-owned,
post-RoPE K min/max envelopes, cluster/outlier representatives, or per-token PQ codes.
All tensor computation below is on the GPU;
only a small candidate ranking is copied to the host for the existing RPC.
No original KV, sidecar, NPU ABI, or resident-page ownership is modified.
"""

from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import re

logger = logging.getLogger(__name__)
SCHEMA = "aissd-gpu-coarse-v1"
ALGORITHM = "segment_minmax_qk_max"
CLUSTER_ALGORITHM = "cluster_centroid_outlier_topm"
PQ_ALGORITHM = "pq_codebook_lookup_topm"
BUCKETS = (8, 16, 32, 64, 96, 128, 256, 384, 512)

# These are the candidate-axis tensors consumed by the unchanged C++ op.
CANDIDATE_KEYS = (
    "aissd_candidate_chunk_ids", "aissd_candidate_block_ids",
    "aissd_candidate_block_lens", "aissd_candidate_token_start",
    "aissd_candidate_token_end", "aissd_candidate_dtype",
    "aissd_candidate_fmt", "aissd_candidate_ndim", "aissd_candidate_shape",
    "aissd_candidate_extent_count", "aissd_candidate_extent_lba",
    "aissd_candidate_extent_bytes",
)


def compiled_bucket(count):
    for bucket in BUCKETS:
        if 0 < count <= bucket:
            return bucket
    raise ValueError(f"Invalid GPU coarse candidate count: {count}")


def _option(extra, name, default):
    key = "lmcache.aissd_gpu_coarse_" + name
    return extra.get(key, os.environ.get("AISSD_GPU_COARSE_" + name.upper(), default))


def _integer(value, name, minimum=0):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if str(value).strip() != str(result) or result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return result


@dataclass(frozen=True)
class CoarseOptions:
    mode: str
    candidates: int
    segments: int
    keep_first: int
    keep_recent: int
    trace_path: str
    shadow_budgets: tuple[int, ...]
    algorithm: str = ALGORITHM
    cluster_centroids: int = 2
    cluster_outliers: int = 2
    cluster_iterations: int = 4
    pq_subspaces: int = 4
    pq_codewords: int = 256
    pq_iterations: int = 8
    pq_train_tokens: int = 4096

    @classmethod
    def from_extra(cls, extra, chunk_size, top_n):
        mode = str(_option(extra, "mode", "off")).strip().lower()
        if mode not in ("off", "shadow", "on"):
            raise ValueError("GPU coarse mode must be off, shadow, or on")
        # Off deliberately ignores all other knobs and does no file/GPU work.
        if mode == "off":
            return None
        algorithm = str(_option(extra, "algorithm", ALGORITHM)).strip().lower()
        if algorithm not in (ALGORITHM, CLUSTER_ALGORITHM, PQ_ALGORITHM):
            raise ValueError("GPU coarse algorithm must be segment_minmax_qk_max, cluster_centroid_outlier_topm, or pq_codebook_lookup_topm")
        candidates = _integer(_option(extra, "candidates", 128), "coarse candidates", 1)
        segments = (_integer(_option(extra, "segments", 4), "coarse segments", 1)
                    if algorithm == ALGORITHM else 0)
        centroids, outliers, iterations = 2, 2, 4
        if algorithm == CLUSTER_ALGORITHM:
            centroids = _integer(_option(extra, "cluster_centroids", 2), "cluster centroids", 1)
            outliers = _integer(_option(extra, "cluster_outliers", 2), "cluster outliers")
            iterations = _integer(_option(extra, "cluster_iterations", 4), "cluster iterations", 1)
            if (centroids > 8 or outliers > 8 or centroids + outliers > min(8, chunk_size) or
                    iterations > 16 or chunk_size > 32767):
                raise ValueError("Cluster needs at most 8 representatives <= chunk_size, and 1..16 iterations")
        pq_subspaces, pq_codewords, pq_iterations, pq_train_tokens = 4, 256, 8, 4096
        if algorithm == PQ_ALGORITHM:
            pq_subspaces = _integer(_option(extra, "pq_subspaces", 4), "PQ subspaces", 1)
            pq_codewords = _integer(_option(extra, "pq_codewords", 256), "PQ codewords", 2)
            pq_iterations = _integer(_option(extra, "pq_iterations", 8), "PQ iterations", 1)
            pq_train_tokens = _integer(_option(extra, "pq_train_tokens", 4096), "PQ train_tokens", 2)
            if (pq_subspaces > 128 or pq_codewords > 256 or pq_iterations > 32 or
                    not pq_codewords <= pq_train_tokens <= 16384):
                raise ValueError("PQ needs <=128 subspaces dividing head_dim, 2..256 codewords, "
                                 "1..32 iterations, and codewords <= train_tokens <= 16384")
        first = _integer(_option(extra, "keep_first", 1), "coarse keep_first")
        recent = _integer(_option(extra, "keep_recent", 4), "coarse keep_recent")
        raw = _option(extra, "shadow_budgets", "32,64,96,128,256")
        raw = raw.split(",") if isinstance(raw, str) else raw
        if not isinstance(raw, (tuple, list)):
            raise ValueError("coarse shadow_budgets must be a list or comma-separated integers")
        budgets = tuple(sorted({candidates, *(
            _integer(v, "coarse shadow budget", 1) for v in raw
        )}))
        if candidates > 512 or any(v > 512 or v < top_n for v in budgets):
            raise ValueError("GPU coarse budgets must be between top_n_chunks and 512")
        if candidates < top_n or (algorithm == ALGORITHM and chunk_size % segments):
            raise ValueError("GPU coarse needs candidates >= top_n and segments dividing chunk_size")
        if first + recent > min(budgets):
            raise ValueError("GPU coarse protected chunks exceed the smallest candidate budget")
        trace = str(_option(extra, "trace_path", "")).strip()
        if mode == "shadow" and not trace:
            raise ValueError("GPU coarse shadow mode requires an explicit trace_path")
        return cls(mode, candidates, segments, first, recent, trace, budgets,
                   algorithm, centroids, outliers, iterations,
                   pq_subspaces, pq_codewords, pq_iterations, pq_train_tokens)


def select_positions(ranking, starts, budget, keep_first, keep_recent):
    """Choose by score, protect boundaries, and restore original input order."""
    n = len(starts)
    if sorted(ranking) != list(range(n)) or len(set(starts)) != n:
        raise RuntimeError("GPU coarse ranking/ranges are not a one-to-one candidate map")
    count = min(int(budget), n)
    if count <= 0:
        raise RuntimeError("GPU coarse received no candidates")
    chronological = sorted(range(n), key=lambda i: starts[i])
    protected = chronological[:keep_first]
    if keep_recent:
        protected += chronological[max(0, n - keep_recent):]
    chosen = set(protected)
    if len(chosen) > count:
        raise RuntimeError("GPU coarse boundary protection exceeds the candidate budget")
    for index in ranking:
        if len(chosen) >= count:
            break
        chosen.add(index)
    return sorted(chosen)


def recall_metrics(reference, kept):
    reference = set(reference)
    if not reference:
        raise RuntimeError("SSD selector returned an empty recall reference")
    hits = len(reference.intersection(kept))
    return {"hits": hits, "reference_count": len(reference),
            "recall": hits / len(reference), "all_retained": hits == len(reference)}


@dataclass
class CoarseSummary:
    layers: dict
    chunk_size: int
    num_chunks: int
    kv_heads: int
    head_dim: int
    nbytes: int
    algorithm: str = ALGORITHM


def build_summary(kv_caches, block_ids, prefix_tokens, chunk_size, block_size,
                  num_layers, kv_heads, head_dim, segments, diagnostic=None, algorithm_options=None):
    """Read live logical pages before release, in bounded 16-chunk batches."""
    import torch

    if algorithm_options is not None and algorithm_options.algorithm == PQ_ALGORITHM:
        if diagnostic is not None:
            raise ValueError("PQ uses independent capture replay; disable min/max diagnostic capture")
        from lmcache.integration.vllm.gpu_coarse_pq import build_layers
        layers, nbytes = build_layers(kv_caches, block_ids, prefix_tokens, chunk_size, block_size,
                                      num_layers, kv_heads, head_dim, algorithm_options)
        return CoarseSummary(layers, chunk_size, prefix_tokens // chunk_size, kv_heads,
                             head_dim, nbytes, PQ_ALGORITHM)
    if algorithm_options is not None and algorithm_options.algorithm == CLUSTER_ALGORITHM:
        if diagnostic is not None:
            raise ValueError("Cluster uses offline raw-K/Q replay; disable min/max diagnostic capture")
        from lmcache.integration.vllm.gpu_coarse_cluster import build_layers
        layers, nbytes = build_layers(kv_caches, block_ids, prefix_tokens, chunk_size, block_size,
                                      num_layers, kv_heads, head_dim, algorithm_options)
        return CoarseSummary(layers, chunk_size, prefix_tokens // chunk_size, kv_heads,
                             head_dim, nbytes, CLUSTER_ALGORITHM)
    if prefix_tokens <= 0 or prefix_tokens % chunk_size or chunk_size % block_size:
        raise RuntimeError("GPU coarse archive must contain complete page-aligned chunks")
    num_chunks = prefix_tokens // chunk_size
    pages = prefix_tokens // block_size
    block_ids = [int(x) for x in block_ids[:pages]]
    if len(block_ids) != pages or any(x <= 0 for x in block_ids):
        raise RuntimeError("GPU coarse summary has missing/null original GPU pages")
    layers = {}
    with torch.no_grad():
        for name, cache in kv_caches.items():
            match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", str(name))
            if match is None:
                raise RuntimeError(f"Cannot identify GPU coarse KV layer: {name}")
            layer = int(match.group(1))
            expected = [block_size, kv_heads, head_dim]
            if (not 0 <= layer < num_layers or layer in layers or
                    cache.dtype != torch.bfloat16 or cache.ndim != 5 or
                    list(cache.shape[2:]) != expected or
                    (cache.shape[0] != 2 and cache.shape[1] != 2)):
                raise RuntimeError(f"Unsupported/duplicate GPU coarse KV layer/layout: {name}")
            paged = cache if cache.shape[1] == 2 else cache.transpose(0, 1)
            if max(block_ids) >= paged.shape[0]:
                raise RuntimeError("GPU coarse source page is out of range")
            page_indices = torch.tensor(block_ids, dtype=torch.long, device=cache.device)
            lo = torch.empty((num_chunks, segments, kv_heads, head_dim),
                             dtype=cache.dtype, device=cache.device)
            hi = torch.empty_like(lo)
            per_chunk = chunk_size // block_size
            for first in range(0, num_chunks, 16):
                last = min(first + 16, num_chunks)
                indices = page_indices[first * per_chunk:last * per_chunk]
                k = paged[:, 0].index_select(0, indices).reshape(
                    last - first, segments, chunk_size // segments, kv_heads, head_dim)
                if diagnostic is not None:
                    diagnostic.capture_k(layer, first, last, k)
                lo[first:last].copy_(k.amin(dim=2))
                hi[first:last].copy_(k.amax(dim=2))
            if not bool(torch.isfinite(lo).all().item()) or not bool(torch.isfinite(hi).all().item()):
                raise RuntimeError("Non-finite GPU coarse summary")
            layers[layer] = (lo, hi)
            if diagnostic is not None:
                diagnostic.capture_summary(layer, lo, hi)
    if set(layers) != set(range(num_layers)):
        raise RuntimeError("GPU coarse summary is missing model layers")
    torch.cuda.synchronize(next(iter(layers.values()))[0].device)
    nbytes = sum(t.numel() * t.element_size() for pair in layers.values() for t in pair)
    return CoarseSummary(layers, chunk_size, num_chunks, kv_heads, head_dim, nbytes)


def score_candidates(summary, layer, query, logical_chunks, num_heads, attention_scale,
                     diagnostic=None, top_m=8, score_mode=1):
    """Conservative QK box score, reduced over segments and GQA query heads.

    The per-head box bounds Q.K for live BF16 K, but the final ranking is an
    approximation to the SSD's INT16 global_token_head_topm chunk objective.
    It is not a correctness guarantee for the final Top-N.
    """
    import torch

    if (layer not in summary.layers or num_heads % summary.kv_heads or
            query.numel() != num_heads * summary.head_dim or
            any(c < 0 or c >= summary.num_chunks for c in logical_chunks)):
        raise RuntimeError("Invalid single-token Q / GPU coarse summary dispatch")
    with torch.no_grad():
        if getattr(summary, "algorithm", ALGORITHM) == PQ_ALGORITHM:
            if diagnostic is not None:
                raise ValueError("PQ uses independent capture replay; disable min/max diagnostic capture")
            from lmcache.integration.vllm.gpu_coarse_pq import score_torch
            books, codes = summary.layers[layer]
            if query.device != books.device or codes.device != books.device:
                raise RuntimeError("GPU PQ Q, codebooks and codes are on different devices")
            scores = score_torch(books, codes, query, attention_scale, top_m, score_mode, logical_chunks)
            if not bool(torch.isfinite(scores).all().item()):
                raise RuntimeError("Non-finite GPU PQ candidate scores")
            return torch.argsort(scores, descending=True, stable=True).to("cpu").tolist()
        if getattr(summary, "algorithm", ALGORITHM) == CLUSTER_ALGORITHM:
            if diagnostic is not None:
                raise ValueError("Cluster uses offline raw-K/Q replay; disable min/max diagnostic capture")
            from lmcache.integration.vllm.gpu_coarse_cluster import score_torch
            vectors, populations = summary.layers[layer]
            if query.device != vectors.device:
                raise RuntimeError("GPU cluster Q and summary are on different devices")
            indices = torch.tensor(logical_chunks, dtype=torch.long, device=vectors.device)
            scores = score_torch(vectors.index_select(0, indices), populations.index_select(0, indices),
                                 query, attention_scale, summary.chunk_size, top_m, score_mode)
            if not bool(torch.isfinite(scores).all().item()):
                raise RuntimeError("Non-finite GPU cluster candidate scores")
            return torch.argsort(scores, descending=True, stable=True).to("cpu").tolist()
        lo, hi = summary.layers[layer]
        if query.device != lo.device:
            raise RuntimeError("GPU coarse Q and summary are on different devices")
        indices = torch.tensor(logical_chunks, dtype=torch.long, device=lo.device)
        q = query.detach().reshape(summary.kv_heads, num_heads // summary.kv_heads,
                                   summary.head_dim).float()
        upper = torch.einsum("cshd,hgd->cshg", hi.index_select(0, indices).float(), q.clamp_min(0))
        upper.add_(torch.einsum("cshd,hgd->cshg", lo.index_select(0, indices).float(), q.clamp_max(0)))
        scores = upper.amax(dim=(1, 2, 3)) * float(attention_scale)
        if not bool(torch.isfinite(scores).all().item()):
            raise RuntimeError("Non-finite GPU coarse candidate scores")
        # Stable ties preserve the original candidate order, as does the
        # subsequent metadata compaction. Copy only N small integer positions.
        ranking = torch.argsort(scores, descending=True, stable=True).to("cpu").tolist()
        if diagnostic is not None:
            diagnostic.capture_scores(query, upper, scores, ranking)
        return ranking


def compact_candidates(context, positions):
    """Replace active aliases only; leave persistent per-layer inputs intact."""
    import torch

    count = int(context["aissd_candidate_count"][0])
    if (len(set(positions)) != len(positions) or not positions or
            any(i < 0 or i >= count for i in positions)):
        raise RuntimeError("Invalid GPU coarse candidate compaction")
    indices = torch.tensor(positions, dtype=torch.long, device="cpu")
    replacement = {}
    for key in CANDIDATE_KEYS:
        tensor = context[key]
        if tensor.device.type != "cpu" or tensor.shape[0] != 1:
            raise RuntimeError("GPU coarse v1 requires one row of CPU native-extent metadata")
        replacement[key] = tensor.index_select(1, indices).contiguous()
    # The RPC uses compact candidate IDs; token ranges remain authoritative for
    # mapping its response back to the request's original logical chunk IDs.
    replacement["aissd_candidate_chunk_ids"] = torch.arange(
        len(positions), dtype=torch.int32, device="cpu").reshape(1, -1)
    replacement["aissd_candidate_count"] = torch.tensor(
        [len(positions)], dtype=torch.int32, device="cpu")
    context.update(replacement)


class GpuCoarseSelector:
    def __init__(self, options, extra=None):
        self.options = options
        self.diagnostic = None
        extra = extra or {}
        if options.algorithm in (CLUSTER_ALGORITHM, PQ_ALGORITHM):
            algorithm = os.environ.get("AISSD_SPARSE_KV_SELECTOR_ALGO", "global_token_head_topm")
            if algorithm not in ("global_token_head_topm", "flat_token_head_topm"):
                raise ValueError("GPU cluster/PQ supports SSD global_token_head_topm only")
        enabled = str(_option(extra, "diag_enabled", "false")).strip().lower()
        if enabled not in ("0", "false", "off", "", "1", "true", "on"):
            raise ValueError("GPU coarse diag_enabled must be a boolean")
        if enabled in ("1", "true", "on"):
            if options.algorithm in (CLUSTER_ALGORITHM, PQ_ALGORITHM):
                raise ValueError("GPU cluster/PQ requires diag_enabled=0; use the corresponding standalone replay module")
            if options.mode != "shadow":
                raise ValueError("GPU coarse diagnostic requires coarse_mode=shadow")
            from lmcache.integration.vllm.gpu_coarse_diagnostic import CoarseDiagnostic
            self.diagnostic = CoarseDiagnostic.from_extra(extra, options)

    def prepare(self, context, query, layer, num_heads, attention_scale,
                summary, req_id, step):
        if (int(context.get("host_active_reqs", 0)) != 1 or
                not context.get("aissd_resident_active") or
                not context.get("host_sparse_decode_ready")):
            raise RuntimeError("GPU coarse v1 requires active single-request resident decode")
        count = int(context["aissd_candidate_count"][0])
        starts = [int(x) for x in context["aissd_candidate_token_start"][0, :count].tolist()]
        ends = [int(x) for x in context["aissd_candidate_token_end"][0, :count].tolist()]
        if count <= 0 or len(starts) != count or any(
                s < 0 or s % summary.chunk_size or e - s != summary.chunk_size
                for s, e in zip(starts, ends)):
            raise RuntimeError("GPU coarse candidates must be complete logical chunks")
        logical = [s // summary.chunk_size for s in starts]
        if getattr(summary, "algorithm", ALGORITHM) != self.options.algorithm:
            raise RuntimeError("GPU coarse algorithm and archived summary disagree")
        diagnostic = None
        if self.diagnostic is not None:
            diagnostic = self.diagnostic.begin_case(
                req_id, step, layer, context, logical, starts, ends,
                num_heads, attention_scale)
        if self.options.algorithm in (CLUSTER_ALGORITHM, PQ_ALGORITHM):
            ranking = score_candidates(summary, layer, query, logical, num_heads, attention_scale,
                                       top_m=int(context.get("aissd_top_m", 8)),
                                       score_mode=int(context.get("aissd_score_mode_code", 1)))
        elif diagnostic is None:
            ranking = score_candidates(summary, layer, query, logical, num_heads, attention_scale)
        else:
            ranking = score_candidates(summary, layer, query, logical, num_heads,
                                       attention_scale, diagnostic=diagnostic)
        opts = self.options
        positions = select_positions(ranking, starts, opts.candidates, opts.keep_first, opts.keep_recent)
        plan = {"req_id": str(req_id), "step": int(step), "layer_id": int(layer),
                "candidate_count": count, "logical": logical, "starts": starts,
                "ranking": ranking, "positions": positions,
                "summary_bytes": summary.nbytes}
        if opts.algorithm in (CLUSTER_ALGORITHM, PQ_ALGORITHM):
            plan["top_m"] = int(context.get("aissd_top_m", 8))
            plan["score_mode"] = int(context.get("aissd_score_mode_code", 1))
        if diagnostic is not None:
            diagnostic.set_positions(positions)
            plan["diagnostic"] = diagnostic
        if opts.mode == "on":
            compact_candidates(context, positions)
        context["aissd_gpu_coarse_input_count"] = count
        context["aissd_gpu_coarse_submitted_count"] = len(positions) if opts.mode == "on" else count
        return plan

    def finish(self, context, plan):
        opts = self.options
        n = int(context["aissd_rpc_selected_chunk_lens"][0])
        ids = [int(x) for x in context["aissd_rpc_selected_chunk_ids"][0, :n].tolist()]
        count = int(context["aissd_candidate_count"][0])
        candidate_ids = context["aissd_candidate_chunk_ids"][0, :count].tolist()
        starts = context["aissd_candidate_token_start"][0, :count].tolist()
        chunk = int(context["chunk_size"])
        mapping = {int(i): int(s) // chunk for i, s in zip(candidate_ids, starts)}
        if (n <= 0 or n > int(context["top_n_chunks"]) or len(ids) != n or
                len(set(ids)) != n or any(i not in mapping for i in ids)):
            raise RuntimeError("Invalid SSD result during GPU coarse completion")
        selected = [mapping[i] for i in ids]
        kept = [plan["logical"][i] for i in plan["positions"]]
        if opts.mode == "on" and not set(selected).issubset(kept):
            raise RuntimeError("SSD returned a chunk outside the GPU coarse candidate set")
        if plan.get("diagnostic") is not None:
            plan["diagnostic"].finish(selected, ids)
        if not opts.trace_path:
            return
        curves = {}
        if opts.mode == "shadow":
            for budget in opts.shadow_budgets:
                positions = select_positions(plan["ranking"], plan["starts"], budget,
                                             opts.keep_first, opts.keep_recent)
                proposed = [plan["logical"][i] for i in positions]
                curves[str(budget)] = {"kept": len(proposed), **recall_metrics(selected, proposed)}
        record = {"schema": SCHEMA, "algorithm": opts.algorithm, "mode": opts.mode,
                  "req_id": plan["req_id"], "step": plan["step"], "layer_id": plan["layer_id"],
                  "candidate_count": plan["candidate_count"], "submitted_count": count,
                  "submitted_bucket": compiled_bucket(count),
                  "configured_candidates": opts.candidates, "segments": opts.segments,
                  "keep_first": opts.keep_first, "keep_recent": opts.keep_recent,
                  "summary_bytes": plan["summary_bytes"], "coarse_logical_chunks": kept,
                  "ssd_selected_logical_chunks": selected,
                  "reference": "full_candidate_ssd_selector" if opts.mode == "shadow" else None,
                  "recall_by_budget": curves}
        if opts.algorithm == CLUSTER_ALGORITHM:
            record.update(cluster_centroids=opts.cluster_centroids, cluster_outliers=opts.cluster_outliers,
                          cluster_iterations=opts.cluster_iterations, cluster_top_m=plan["top_m"],
                          cluster_score_mode=plan["score_mode"])
        if opts.algorithm == PQ_ALGORITHM:
            record.update(pq_subspaces=opts.pq_subspaces, pq_codewords=opts.pq_codewords,
                          pq_iterations=opts.pq_iterations, pq_train_tokens=opts.pq_train_tokens,
                          pq_top_m=plan["top_m"], pq_score_mode=plan["score_mode"])
        path = Path(opts.trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Append across service restarts so prior diagnostics are not silently
        # destroyed. The checker rejects duplicate request/step/layer records.
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
