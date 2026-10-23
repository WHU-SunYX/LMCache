# SPDX-License-Identifier: Apache-2.0
"""First GPU-residency implementation: eager, synchronous and single GPU."""

from dataclasses import dataclass, field
import ctypes
import os
import re
import time

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorWorkerMetadata
from vllm.logger import init_logger

from lmcache.integration.vllm.resident_cache import LayerLRU, layer_read_layout, wait_for_archive

logger = init_logger(__name__)


@dataclass
class ResidentWorkerMetadata(KVConnectorWorkerMetadata):
    aissd_resident_ready_prefixes: dict[str, int] = field(default_factory=dict)

    def aggregate(self, other):
        if not isinstance(other, ResidentWorkerMetadata):
            raise TypeError("Incompatible resident worker metadata")
        # Readiness requires every worker, even though v1 admits one worker only.
        self.aissd_resident_ready_prefixes = {
            req: prefix for req, prefix in self.aissd_resident_ready_prefixes.items()
            if other.aissd_resident_ready_prefixes.get(req) == prefix
        }
        return self


class ResidentConnectorMixin:
    def _init_aissd_residency(self):
        from vllm.v1.core.aissd_resident_cache import resident_options, validate_resident_config
        self._resident_budget, chunk, _ = resident_options(self._vllm_config)
        self._resident_archives = {}
        self._resident_worker = {}
        self._resident_current = {}
        self._resident_scratch = None
        if not self._resident_budget:
            return
        validate_resident_config(self._vllm_config)
        if chunk != self._lmcache_chunk_size:
            raise ValueError("Resident chunk size must equal LMCache chunk_size")
        if (not self.sparse_kv_spec.enabled or not self.sparse_kv_spec.enable_sparse_attention
            or self.sparse_kv_spec.sparse_kv_backend not in ("ssd-npu", "ssd-cpu")):
            raise ValueError("GPU residency requires the SSD sparse selector")
        if self.async_loading or self.config.use_layerwise or self.config.enable_blending:
            raise ValueError("GPU residency requires synchronous, non-layerwise, non-blending LMCache")
        if self.config.save_decode_cache or not self._discard_partial_chunks:
            raise ValueError("GPU residency requires save_decode_cache=false and complete chunks only")
        if os.environ.get("VLLM_SPARSE_ATTENTION_IMPL", "fa_varlen") != "fa_varlen":
            raise ValueError("GPU residency requires VLLM_SPARSE_ATTENTION_IMPL=fa_varlen")
        if os.environ.get("AISSD_SPARSE_KV_LAYER_REUSE", "1").lower() not in ("0", "false", "off"):
            raise ValueError("GPU residency v1 requires AISSD_SPARSE_KV_LAYER_REUSE=0")
        if os.environ.get("AISSD_TOKEN_REUSE_STRATEGY", "none") != "none":
            raise ValueError("GPU residency v1 requires AISSD_TOKEN_REUSE_STRATEGY=none")
        if os.environ.get("AISSD_QK_CALIB_DENSE_BASELINE", "0") != "0":
            raise ValueError("Dense calibration cannot read a released GPU prefix")
        for flag in ("AISSD_QK_CALIB_DUMP", "AISSD_SELECTOR_QUALITY_TRACE", "AISSD_TOKEN_REUSE_Q_DRIFT_SHADOW"):
            if os.environ.get(flag, "0").lower() not in ("0", "false", "off", ""):
                raise ValueError(f"GPU residency v1 requires {flag}=0")
        if os.environ.get("AISSD_SPARSE_KV_PREFILL_POLICY", "full_load") != "full_load":
            raise ValueError("GPU residency v1 requires dense full_load prefill")
        extra = self._aissd_sparse_extra_config
        self._resident_archive_timeout = float(extra.get("lmcache.aissd_gpu_archive_timeout_s", 300))
        if self._resident_archive_timeout <= 0:
            raise ValueError("Resident archive timeout must be positive")
        logger.info("[aissd-resident] enabled budget_tokens=%d chunk=%d eager_single_request=1",
                    self._resident_budget, chunk)

    def _resident_update_tracker(self, tracker, scheduler_output):
        if not self._resident_budget:
            return
        state = scheduler_output.aissd_resident_states.get(tracker.req_id)
        if state:
            tracker.allocated_block_ids = list(state["logical_block_ids"])

    def _resident_finish_metadata(self, meta, scheduler_output):
        if not self._resident_budget:
            return
        meta.aissd_resident_states = scheduler_output.aissd_resident_states
        meta.aissd_resident_reset_ids = set(scheduler_output.finished_req_ids) | set(scheduler_output.preempted_req_ids or ())
        cached = scheduler_output.scheduled_cached_reqs
        computed = dict(zip(cached.req_ids, cached.num_computed_tokens))
        for req in scheduler_output.scheduled_new_reqs:
            computed[req.req_id] = req.num_computed_tokens
        for req in meta.requests:
            state = meta.aissd_resident_states.get(req.req_id)
            tracker = self._request_trackers[req.req_id]
            if state:
                req.sparse_lmcache_cached_tokens = state["prefix_tokens"]
                req.sparse_block_ids = list(state["logical_block_ids"])
                req.sparse_context_token_count = computed[req.req_id]
                req.sparse_is_decode_phase = True
                req.save_spec.can_save = False
            elif len(tracker.token_ids) >= tracker.prompt_len:
                # Preserve the phase-1 warm-prefix boundary. On a cold request,
                # archive the complete prompt chunks, leaving the partial tail local.
                prefix = tracker.num_lmcache_cached_tokens or (tracker.prompt_len // self._lmcache_chunk_size * self._lmcache_chunk_size)
                if prefix:
                    if prefix % self._lmcache_chunk_size:
                        raise ValueError("Resident prefix hit must be chunk-aligned")
                    recomputed_tail = None
                    if prefix == tracker.prompt_len == tracker.num_lmcache_cached_tokens:
                        # A full LMCache hit schedules the last prompt token
                        # again to produce logits. Its live K/V can differ
                        # from the immutable SSD copy. Preserve that update
                        # before the scheduler releases the original pages.
                        token = prefix - 1
                        block_index = token // self._block_size
                        blocks = tracker.allocated_block_ids
                        if block_index >= len(blocks) or int(blocks[block_index]) <= 0:
                            raise RuntimeError("Recomputed resident tail has no live GPU page")
                        recomputed_tail = {
                            "token_index": token, "block_id": int(blocks[block_index]),
                        }
                    meta.aissd_resident_archive_requests[req.req_id] = {
                        "prefix_tokens": prefix, "tokens": list(tracker.token_ids[:prefix]),
                        "request_configs": tracker.request_configs,
                        "recomputed_tail": recomputed_tail,
                    }

    def _resident_capture_recomputed_tail(self, source):
        tail = source.get("recomputed_tail")
        if tail is None:
            return {}
        import torch
        token = int(tail["token_index"])
        if token != int(source["prefix_tokens"]) - 1:
            raise RuntimeError("Invalid recomputed resident tail boundary")
        block = int(tail["block_id"])
        offset = token % self._block_size
        snapshots = {}
        expected_shape = [self._block_size, self._sparse_num_kv_heads, self._sparse_head_size]
        for name, cache in self.kv_caches.items():
            match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", str(name))
            if match is None:
                raise RuntimeError(f"Cannot identify resident KV layer: {name}")
            layer = int(match.group(1))
            if not 0 <= layer < self.num_layers or layer in snapshots:
                raise RuntimeError("Duplicate or out-of-range resident KV layer")
            if (cache.dtype != torch.bfloat16 or cache.ndim != 5
                or list(cache.shape[2:]) != expected_shape
                or (cache.shape[0] != 2 and cache.shape[1] != 2)):
                raise ValueError("Unsupported recomputed resident tail KV layout")
            paged = cache if cache.shape[1] == 2 else cache.transpose(0, 1)
            if not 0 < block < paged.shape[0]:
                raise RuntimeError("Recomputed resident tail GPU page is out of range")
            # Copy directly to CPU, without retaining any GPU page/view or
            # allocating a persistent CUDA tensor outside the resident budget.
            snapshots[layer] = paged[block, :, offset].detach().to(
                device="cpu", copy=True).contiguous()
        if set(snapshots) != set(range(self.num_layers)):
            raise RuntimeError("Recomputed resident tail is missing KV layers")
        return snapshots

    def _resident_restore_recomputed_tail(self, archive, chunk, layer, kv_cache, block_ids):
        tail = archive.get("recomputed_tail")
        if tail is None:
            return
        token = int(tail["token_index"])
        if chunk != token // self._lmcache_chunk_size:
            return
        import torch
        snapshots = archive["tail_kv"]
        if layer not in snapshots or len(block_ids) != self._lmcache_chunk_size // self._block_size:
            raise RuntimeError("Recomputed resident tail refill is incomplete")
        page, offset = divmod(token % self._lmcache_chunk_size, self._block_size)
        paged = kv_cache if kv_cache.shape[1] == 2 else kv_cache.transpose(0, 1)
        # Apply on every SSD miss, including after eviction or pool shrink.
        # Publish LRU validity only after both the raw read and this update.
        paged[block_ids[page], :, offset].copy_(snapshots[layer], non_blocking=False)
        torch.cuda.current_stream(kv_cache.device).synchronize()

    def _resident_gds_backend(self):
        backends = self.lmcache_engine.storage_manager.storage_backends.values()
        found = [b for b in backends if callable(getattr(b, "_load_gds", None))]
        if len(found) != 1:
            raise ValueError("GPU residency requires exactly one GDS backend")
        return found[0]

    def build_connector_worker_meta(self):
        if not self._resident_budget:
            return None
        import torch
        metadata = self._parent._get_connector_metadata()
        ready = {}
        if metadata.aissd_resident_archive_requests:
            # Ensure compute and the producer's GPU reads have finished before
            # the scheduler is allowed to transfer/free the original ownership.
            torch.cuda.synchronize(self.device)
        for req_id, source in metadata.aissd_resident_archive_requests.items():
            if req_id not in self._resident_archives:
                engine = self.lmcache_engine
                backend = self._resident_gds_backend()
                mask = torch.ones(len(source["tokens"]), dtype=torch.bool)
                infos = [(key, start, end) for start, end, key in engine.token_database.process_tokens(
                    tokens=source["tokens"], mask=mask, request_configs=source["request_configs"])]
                expected = source["prefix_tokens"] // self._lmcache_chunk_size
                if len(infos) != expected:
                    raise RuntimeError("Incomplete resident prefix key mapping")
                pipeline = getattr(engine, "_gds_bounded_pipeline", None)
                check = pipeline.raise_if_async_error if pipeline is not None else lambda: None
                wait_for_archive(infos, backend.contains, check, self._resident_archive_timeout)
                chunks = backend.get_sparse_kv_candidate_manifest(
                    infos, req_id=req_id, chunk_size=self._lmcache_chunk_size)
                if len(chunks) != expected:
                    raise RuntimeError("Committed resident archive has missing manifests")
                raw = {int(ch["token_start"]) // self._lmcache_chunk_size: ch for ch in chunks}
                if set(raw) != set(range(expected)):
                    raise RuntimeError("Resident archive has gaps or duplicate chunks")
                hidden = self._sparse_num_kv_heads * self._sparse_head_size
                for ch in chunks:
                    layer_read_layout(ch, 0, self.num_layers, self._lmcache_chunk_size, hidden)
                # This hook runs after the prefill forward, while the scheduler
                # still owns the original pages. Readiness also requires the
                # recomputed tail snapshot; a failed capture forbids release.
                tail_kv = self._resident_capture_recomputed_tail(source)
                self._resident_archives[req_id] = {
                    **source, "raw": raw, "backend": backend, "tail_kv": tail_kv,
                }
                logger.info("[aissd-resident-archive] req=%s prefix_tokens=%d chunks=%d committed=1",
                            req_id, source["prefix_tokens"], expected)
                if tail_kv:
                    logger.info(
                        "[aissd-resident-tail] req=%s token=%d layers=%d host_bytes=%d preserved=1",
                        req_id, source["recomputed_tail"]["token_index"], len(tail_kv),
                        sum(t.numel() * t.element_size() for t in tail_kv.values()),
                    )
            ready[req_id] = source["prefix_tokens"]
        # Every active forward must finish before its blocks can be resized or
        # recycled on the scheduler. This also covers request termination.
        if self._resident_current:
            torch.cuda.synchronize(self.device)
        return ResidentWorkerMetadata(ready)

    def _resident_start_step(self, metadata):
        if not self._resident_budget:
            return
        import torch
        for req_id in metadata.aissd_resident_reset_ids:
            self._resident_archives.pop(req_id, None)
            self._resident_worker.pop(req_id, None)
        self._resident_current = metadata.aissd_resident_states
        context = getattr(self, "_sparse_current_step_context", None)
        if context is not None:
            context.pop("aissd_resident_loader", None)
            context["aissd_resident_active"] = False
        for req_id, state in self._resident_current.items():
            if req_id not in self._resident_archives:
                raise RuntimeError("Resident release occurred before worker archive readiness")
            per_chunk = self._lmcache_chunk_size // self._block_size
            capacity = len(state["hot_block_ids"]) // per_chunk
            worker = self._resident_worker.get(req_id)
            if worker is None or worker["epoch"] != state["epoch"]:
                self._resident_worker[req_id] = {"epoch": state["epoch"], "layers": {
                    layer: LayerLRU(capacity) for layer in range(self.num_layers)
                }, "context_built": False, "step": 0, "totals": {
                    "hits": 0, "misses": 0, "load_bytes": 0, "evictions": 0,
                }}
                worker = self._resident_worker[req_id]
            for cache in worker["layers"].values():
                cache.resize(capacity)
            worker["step"] += 1
            worker["step_stats"] = {"hits": 0, "misses": 0, "load_bytes": 0, "evictions": 0}
            if not worker["context_built"]:
                archive = self._resident_archives[req_id]
                n = state["prefix_tokens"]
                # Candidate tables encode null placeholders. The selector's
                # raw chunk IDs are resolved to real hot pages before Attention.
                self._sparse_runtime_requests[req_id] = {
                    "tokens": archive["tokens"], "token_mask": torch.ones(n, dtype=torch.bool),
                    "slot_mapping": torch.zeros(n, dtype=torch.long, device=self.device),
                    "lmcache_cached_tokens": n, "vllm_cached_tokens": 0,
                    "request_configs": archive["request_configs"],
                    "sparse_spec": self.sparse_kv_spec,
                    "indexcache_policy": next(r.indexcache_policy for r in metadata.requests if r.req_id == req_id),
                    "dense_fallback_safe": False,
                }

    def _resident_publish_context(self, context):
        if self._resident_budget and self._resident_current:
            context["aissd_resident_active"] = True
            context["aissd_resident_loader"] = self._resident_resolve_layer
            context["dense_fallback_safe"] = False
            for req_id in self._resident_current:
                self._resident_worker[req_id]["context_built"] = True

    def _resident_read_layer(self, backend, manifest, layer, kv_cache, block_ids):
        import torch
        chunk = self._lmcache_chunk_size
        hidden = self._sparse_num_kv_heads * self._sparse_head_size
        tail = [self._block_size, self._sparse_num_kv_heads, self._sparse_head_size]
        if (kv_cache.dtype != torch.bfloat16 or kv_cache.ndim != 5
            or list(kv_cache.shape[2:]) != tail
            or (kv_cache.shape[0] != 2 and kv_cache.shape[1] != 2)):
            raise ValueError(
                "Unsupported resident GPU KV cache geometry: "
                f"shape={tuple(kv_cache.shape)} dtype={kv_cache.dtype} "
                f"stride={tuple(kv_cache.stride())}; expected BF16 "
                f"[blocks, 2, {tail[0]}, {tail[1]}, {tail[2]}] or "
                f"[2, blocks, {tail[0]}, {tail[1]}, {tail[2]}]"
            )
        # The current vLLM backend supplies [blocks, 2, BS, H, D]. Older
        # K/V-first tensors are accepted through a view, without copying the
        # preallocated KV pool. Prefer the native blocks-first convention.
        paged_cache = kv_cache if kv_cache.shape[1] == 2 else kv_cache.transpose(0, 1)
        k_offset, v_offset, size = layer_read_layout(manifest, layer, self.num_layers, chunk, hidden)
        if self._resident_scratch is None:
            self._resident_scratch = torch.empty((2, chunk, hidden), dtype=torch.bfloat16, device=kv_cache.device)
        scratch = self._resident_scratch
        stream = torch.cuda.current_stream(kv_cache.device)
        stream.synchronize()  # preceding attention and any previous scratch use
        for which, offset in enumerate((k_offset, v_offset)):
            actual = backend._load_gds(manifest["path"], offset,
                                       ctypes.c_void_p(scratch[which].data_ptr()), size, 0)
            if actual != size:
                raise IOError(f"Short resident layer read: {actual}/{size} bytes from {manifest['path']}")
        # _load_gds is synchronous. Conservatively fence a GDS implementation
        # before launching the scatter on PyTorch's current stream.
        torch.cuda.synchronize(kv_cache.device)
        shaped = scratch.reshape(2, chunk // self._block_size, self._block_size,
                                 self._sparse_num_kv_heads, self._sparse_head_size)
        for index, block_id in enumerate(block_ids):
            paged_cache[block_id].copy_(shaped[:, index])
        stream.synchronize()
        return 2 * size

    def _resident_resolve_layer(self, context, layer, kv_cache):
        import torch
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Resident KV refill cannot execute inside CUDA graph capture")
        if len(self._resident_current) != 1 or not 0 <= layer < self.num_layers:
            raise RuntimeError("Invalid resident request/layer dispatch")
        req_id, state = next(iter(self._resident_current.items()))
        worker = self._resident_worker[req_id]
        archive = self._resident_archives[req_id]
        selected_count = int(context["aissd_rpc_selected_chunk_lens"][0])
        selected_ids = context["aissd_rpc_selected_chunk_ids"][0, :selected_count].tolist()
        if not 0 < selected_count <= self.sparse_kv_spec.top_n_chunks:
            raise RuntimeError("Resident selector returned an invalid selected count")
        # IDs returned by the native selector are positions AFTER candidate
        # capping/reordering, not raw logical chunk numbers. Decode their ranges
        # from this layer's candidate table; do not assume index == chunk ID.
        candidate_ids = context["aissd_candidate_chunk_ids"][0].tolist()
        starts = context["aissd_candidate_token_start"][0].tolist()
        ends = context["aissd_candidate_token_end"][0].tolist()
        candidate_count = int(context["aissd_candidate_count"][0])
        ranges = {int(candidate_ids[i]): (int(starts[i]), int(ends[i])) for i in range(candidate_count)}
        logical = []
        for selected in selected_ids:
            if selected not in ranges:
                raise RuntimeError("Resident selector returned an unknown candidate ID")
            start, end = ranges[selected]
            if start % self._lmcache_chunk_size or end - start != self._lmcache_chunk_size:
                raise RuntimeError("Resident candidate is not one complete source chunk")
            logical.append(start // self._lmcache_chunk_size)
        if any(chunk not in archive["raw"] for chunk in logical):
            raise RuntimeError("Selected resident chunk is absent from the committed SSD archive")
        per_chunk = self._lmcache_chunk_size // self._block_size
        hot = state["hot_block_ids"]
        context["selected_ready_flags"][0] = 0
        def load(chunk, slot):
            blocks = hot[slot * per_chunk:(slot + 1) * per_chunk]
            loaded = self._resident_read_layer(
                archive["backend"], archive["raw"][chunk], layer, kv_cache, blocks)
            self._resident_restore_recomputed_tail(archive, chunk, layer, kv_cache, blocks)
            return loaded
        t0 = time.perf_counter()
        resolved, stats = worker["layers"][layer].resolve(logical, load)
        remote = [b for _, slot in resolved for b in hot[slot * per_chunk:(slot + 1) * per_chunk]]
        local_count = int(context["local_block_lens"][0].item())
        local_tokens = int(context["local_token_lens"][0].item())
        local = context["local_block_table"][0, :local_count].tolist()
        if any(b == 0 for b in remote + local) or set(remote) & set(local):
            raise RuntimeError("Resident selected/local GPU pages alias or contain null blocks")
        composed = remote + local
        if len(composed) > context["fa_block_table"].shape[1]:
            raise RuntimeError("Resident attention metadata capacity exceeded")
        ids = torch.tensor(composed, dtype=torch.int32, device=kv_cache.device)
        context["selected_block_table"][0].fill_(-1)
        context["selected_block_table"][0, :len(remote)].copy_(ids[:len(remote)])
        context["selected_block_lens"][0] = len(remote)
        context["fa_block_table"][0].zero_()
        context["fa_block_table"][0, :len(composed)].copy_(ids)
        context["fa_block_lens"][0] = len(composed)
        context["fa_seq_lens"][0] = len(remote) * self._block_size + local_tokens
        context["selected_ready_flags"][0] = 1
        context["sparse_selected_load_bytes"] = stats["load_bytes"]
        context["sparse_selected_loaded_chunks"] = stats["misses"]
        context["sparse_selected_load_ms"] = (time.perf_counter() - t0) * 1000
        for name, value in stats.items():
            worker["step_stats"][name] += value
            worker["totals"][name] += value
        if layer == self.num_layers - 1:
            block_bytes = 2 * self.num_layers * self._block_size * self._sparse_num_kv_heads * self._sparse_head_size * 2
            logger.info(
                "[aissd-resident-step] req=%s step=%d epoch=%d budget_tokens=%d "
                "owned_blocks=%d owned_kv_bytes=%d released_blocks=%d pool_free_blocks=%d "
                "layer_chunk_hits=%d layer_chunk_misses=%d gds_read_bytes=%d evictions=%d "
                "scratch_bytes=%d cumulative_gds_read_bytes=%d",
                req_id, worker["step"], state["epoch"], state["budget_tokens"],
                state["owned_blocks"], state["owned_blocks"] * block_bytes,
                state["released_blocks"], state["pool_free_blocks"],
                worker["step_stats"]["hits"], worker["step_stats"]["misses"],
                worker["step_stats"]["load_bytes"], worker["step_stats"]["evictions"],
                0 if self._resident_scratch is None else self._resident_scratch.numel() * 2,
                worker["totals"]["load_bytes"],
            )
