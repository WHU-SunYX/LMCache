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
        from vllm.v1.core.aissd_resident_cache import (
            resident_options, staged_prefill_enabled, validate_resident_config,
        )
        self._resident_budget, chunk, _ = resident_options(self._vllm_config)
        self._resident_staged_prefill = staged_prefill_enabled(self._vllm_config)
        self._resident_prefill_current = None
        self._resident_prefill_stager = None
        self._resident_archives = {}
        self._resident_worker = {}
        self._resident_current = {}
        self._resident_scratch = None
        self._gpu_coarse = None
        extra = self._aissd_sparse_extra_config
        self._resident_page_access = str(extra.get(
            "lmcache.aissd_gpu_page_access", "legacy")).strip().lower()
        if self._resident_page_access not in ("legacy", "batched"):
            raise ValueError("lmcache.aissd_gpu_page_access must be legacy or batched")
        if self._resident_page_access == "batched" and not self._resident_budget:
            raise ValueError("Batched page access requires aissd_gpu_resident_tokens > 0")
        coarse_mode = str(extra.get("lmcache.aissd_gpu_coarse_mode",
                                   os.environ.get("AISSD_GPU_COARSE_MODE", "off"))).strip().lower()
        if coarse_mode != "off":
            from lmcache.integration.vllm.gpu_coarse_selector import CoarseOptions, GpuCoarseSelector
            options = CoarseOptions.from_extra(extra, chunk, self.sparse_kv_spec.top_n_chunks)
            if not self._resident_budget:
                raise ValueError("GPU coarse v1 requires aissd_gpu_resident_tokens > 0")
            self._gpu_coarse = GpuCoarseSelector(options, extra=extra)
        validate_resident_config(self._vllm_config)
        if not self._resident_budget:
            return
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
        if self._resident_staged_prefill:
            if self._gpu_coarse is not None and self._gpu_coarse.options.algorithm != "pq_codebook_lookup_topm":
                raise ValueError("AISSD staged prefill supports coarse off or PQ")
            logger.info("[aissd-prefill] mode=staged dense_causal=1")
        logger.info("[aissd-resident] enabled budget_tokens=%d chunk=%d eager_single_request=1",
                    self._resident_budget, chunk)
        if self._resident_page_access == "batched":
            logger.info("[aissd-resident-page-access] mode=batched")
        if self._gpu_coarse is not None:
            options = self._gpu_coarse.options
            enabled, _, fallback, _ = self._aissd_qkpack_sidecar_policy()
            if not enabled or fallback not in ("error", "fail"):
                raise ValueError("GPU coarse v1 requires qkpack sidecars and fallback=error")
            if os.environ.get("AISSD_SPARSE_KV_FRAGMENT_POLICY", "fail") != "fail":
                raise ValueError("GPU coarse v1 requires AISSD_SPARSE_KV_FRAGMENT_POLICY=fail")
            if options.algorithm == "pq_codebook_lookup_topm":
                logger.info("[aissd-gpu-coarse] mode=%s algorithm=%s candidates=%d segments=%d trace=%s",
                            options.mode, options.algorithm, options.candidates, options.segments,
                            options.trace_path or "disabled")
                logger.info("[aissd-gpu-coarse-pq] subspaces=%d codewords=%d iterations=%d "
                            "train_tokens=%d code_dtype=uint8 codebook_dtype=bfloat16",
                            options.pq_subspaces, options.pq_codewords,
                            options.pq_iterations, options.pq_train_tokens)
            else:
                logger.info("[aissd-gpu-coarse] mode=%s algorithm=%s candidates=%d segments=%d "
                            "cluster_centroids=%d cluster_outliers=%d cluster_iterations=%d trace=%s",
                            options.mode, options.algorithm, options.candidates, options.segments,
                            options.cluster_centroids, options.cluster_outliers, options.cluster_iterations,
                            options.trace_path or "disabled")

    def _resident_update_tracker(self, tracker, scheduler_output):
        if not self._resident_budget:
            return
        state = scheduler_output.aissd_resident_states.get(tracker.req_id)
        if state:
            tracker.allocated_block_ids = list(state["logical_block_ids"])
            if state.get("prefill"):
                # LMCache's tracker otherwise treats every one-token update
                # as decode, which can skip saving a just-completed chunk.
                tracker.is_decode_phase = False

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
            if state and state.get("prefill"):
                self._resident_plan_prefill(meta, req, tracker, state)
            elif state:
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
                    if getattr(self, "_gpu_coarse", None) is not None:
                        # Scheduler metadata contains IDs only, never tensor
                        # views. The worker reads live post-prefill K before
                        # publishing readiness and releasing these pages.
                        meta.aissd_resident_archive_requests[req.req_id]["coarse_block_ids"] = list(
                            tracker.allocated_block_ids[:prefix // self._block_size])

    def _resident_plan_prefill(self, meta, req, tracker, state):
        chunk = self._lmcache_chunk_size
        if tracker.disagg_spec is not None:
            raise RuntimeError("AISSD staged prefill does not support disaggregation")
        if req.save_spec is not None:
            req.save_spec.skip_leading_tokens = max(req.save_spec.skip_leading_tokens,
                                                    state["prefix_tokens"])
        done = min(state["total_tokens"], tracker.prompt_len)
        prefix = done // chunk * chunk
        if not prefix:
            return
        if (req.save_spec is None or (prefix > req.save_spec.skip_leading_tokens
                                      and not req.save_spec.can_save)):
            raise RuntimeError("AISSD staged prefill requires saving newly completed KV chunks")
        if len(tracker.token_ids) < done:
            raise RuntimeError("AISSD staged prefill is missing scheduled prompt tokens")
        final = state["total_tokens"] >= tracker.prompt_len
        tail = None
        if state["external_tail_tokens"]:
            token = tracker.prompt_len - 1
            block = state["logical_block_ids"][token // self._block_size]
            if not final or not 0 < block:
                raise RuntimeError("AISSD recomputed tail has no live page")
            tail = {"token_index": token, "block_id": block}
        meta.aissd_resident_archive_requests[req.req_id] = {
            "prefix_tokens": prefix, "tokens": list(tracker.token_ids[:prefix]),
            "request_configs": tracker.request_configs, "staged": True,
            "final": final, "recomputed_tail": tail,
        }

    def _resident_prefill_archive_infos(self, tokens, prefix, request_configs):
        import torch

        chunk = self._lmcache_chunk_size
        if len(tokens) != prefix or prefix % chunk:
            raise RuntimeError("AISSD staged prefill token/chunk boundary mismatch")
        mask = torch.ones(prefix, dtype=torch.bool)
        infos = [(key, start, end) for start, end, key in self.lmcache_engine.token_database.process_tokens(
            tokens=tokens, mask=mask, request_configs=request_configs)]
        if (len(infos) != prefix // chunk or any(
                start != i * chunk or end != (i + 1) * chunk
                for i, (_, start, end) in enumerate(infos))):
            raise RuntimeError("AISSD staged prefill archive has missing token ranges")
        return infos

    def _resident_begin_prefill(self, metadata, states):
        import torch
        from lmcache.integration.vllm.aissd_prefill_stage import PrefillLayerStager

        if not self._resident_staged_prefill or len(states) != 1:
            raise RuntimeError("Unexpected AISSD staged prefill batch")
        req_id, state = next(iter(states.items()))
        matches = [r for r in metadata.requests if r.req_id == req_id]
        if len(matches) != 1:
            raise RuntimeError("Missing AISSD staged prefill connector request")
        req = matches[0]
        prefix = state["prefix_tokens"]
        restore = state["external_tail_tokens"]
        archived = prefix + (self._lmcache_chunk_size if restore else 0)
        stager = self._resident_prefill_stager
        if stager is None:
            stager = PrefillLayerStager(
                max_tokens=self._vllm_config.model_config.max_model_len,
                num_layers=self.num_layers, chunk_size=self._lmcache_chunk_size,
                block_size=self._block_size, heads=self._sparse_num_kv_heads,
                head_dim=self._sparse_head_size)
            self._resident_prefill_stager = stager
        backend = self._resident_gds_backend()
        plan = ()
        if archived:
            infos = self._resident_prefill_archive_infos(req.token_ids[:archived], archived,
                                                        req.request_configs)
            manifests = backend.get_sparse_kv_candidate_manifest(
                infos, req_id=req_id, chunk_size=self._lmcache_chunk_size)
            plan = stager.plan_archive(manifests, prefix_tokens=archived)
        total = state["total_tokens"]
        prefix_pages = prefix // self._block_size
        logical_pages = state["logical_block_ids"]
        local_pages = logical_pages[prefix_pages:]
        if (len(local_pages) != state["owned_blocks"] or any(i <= 0 for i in local_pages)
                or len(set(local_pages)) != len(local_pages)
                or len(logical_pages) != (total + self._block_size - 1) // self._block_size
                or any(logical_pages[:prefix_pages])):
            raise RuntimeError("AISSD staged prefill logical table has invalid live pages")
        table = None
        if plan:
            pages = (total + self._block_size - 1) // self._block_size
            table = torch.arange(pages, dtype=torch.int32, device=self.device).reshape(1, -1)
        self._resident_prefill_current = {
            "req_id": req_id, "state": state, "plan": plan, "backend": backend,
            "local_pages": local_pages, "table": table, "layers": set(),
            "read_bytes": 2 * archived * stager.token_bytes * self.num_layers,
            "started": time.perf_counter(),
        }

    def _resident_prefill_forward(self, attention, layer_id, layer, query, key, value,
                                  cache, attn_metadata, output):
        from copy import copy

        current = self._resident_prefill_current
        state = current["state"]
        total = state["total_tokens"]
        num_query = total - state["computed_before"]
        if (attn_metadata is None or attn_metadata.num_actual_tokens != num_query
                or attn_metadata.max_query_len != num_query or attn_metadata.max_seq_len != total
                or attn_metadata.seq_lens.numel() != 1 or not attn_metadata.causal
                or not 0 <= layer_id < self.num_layers or layer_id in current["layers"]):
            raise RuntimeError("Unexpected AISSD staged prefill attention metadata")
        if current["plan"]:
            if cache.ndim != 5 or cache.shape[1] != 2:
                raise RuntimeError("AISSD staged FlashAttention requires a blocks-first KV cache")
            stager = self._resident_prefill_stager
            stager.stage(current["backend"], current["plan"], layer_id, cache,
                         current["local_pages"], local_tokens=total - state["prefix_tokens"],
                         restore_tokens=state["external_tail_tokens"])
            cache = stager.paged_view(total, pages_first=True)
            attn_metadata = copy(attn_metadata)
            attn_metadata.block_table = current["table"]
            attn_metadata.max_seq_len = total
            attn_metadata.use_cascade = False
            attn_metadata.scheduler_metadata = None
            attn_metadata.prefix_scheduler_metadata = None
            attn_metadata.max_num_splits = 0
        result = attention(layer, query, key, value, cache, attn_metadata, output=output)
        current["layers"].add(layer_id)
        return result

    def _resident_archive_pq(self, archive):
        import torch
        from lmcache.integration.vllm.gpu_coarse_pq import (
            CHUNK_BATCH, _training_indices, encode_torch, train_torch,
        )
        from lmcache.integration.vllm.gpu_coarse_selector import CoarseSummary, PQ_ALGORITHM

        stager = self._resident_prefill_stager
        opts = self._gpu_coarse.options
        chunk, heads, dim = stager.chunk_size, stager.heads, stager.head_dim
        tokens = archive["prefix_tokens"]
        plan = stager.plan_archive([archive["raw"][i] for i in range(tokens // chunk)],
                                  prefix_tokens=tokens)
        samples = torch.tensor(_training_indices(tokens, opts.pq_train_tokens),
                               dtype=torch.long, device=self.device)
        layers = {}
        for layer in range(self.num_layers):
            keys = stager.archive_key(archive["backend"], plan, layer, self.device)
            if archive["recomputed_tail"] is not None:
                keys[-1].copy_(archive["tail_kv"][layer][0], non_blocking=False)
            training = keys.index_select(0, samples)
            if not bool(torch.isfinite(training).all().item()):
                raise RuntimeError("Non-finite AISSD staged PQ training K")
            books = train_torch(training, opts.pq_subspaces, opts.pq_codewords, opts.pq_iterations)
            del training
            codes = torch.empty((tokens // chunk, chunk, heads, opts.pq_subspaces),
                                dtype=torch.uint8, device=self.device)
            for begin in range(0, len(codes), CHUNK_BATCH):
                end = min(begin + CHUNK_BATCH, len(codes))
                batch = keys[begin * chunk:end * chunk].view(end - begin, chunk, heads, dim)
                if not bool(torch.isfinite(batch).all().item()):
                    raise RuntimeError("Non-finite AISSD staged PQ K")
                codes[begin:end].copy_(encode_torch(books, batch))
            if not bool(torch.isfinite(books).all().item()):
                raise RuntimeError("Non-finite AISSD staged PQ codebook")
            layers[layer] = (books, codes)
        nbytes = sum(t.numel() * t.element_size() for pair in layers.values() for t in pair)
        logger.info("[aissd-prefill-pq] prefix_tokens=%d archive_key_read_bytes=%d",
                    tokens, tokens * stager.token_bytes * self.num_layers)
        return CoarseSummary(layers, chunk, tokens // chunk, heads, dim, nbytes, PQ_ALGORITHM)

    def _resident_check_prefill_forward(self):
        prefill = getattr(self, "_resident_prefill_current", None)
        if prefill and prefill["layers"] != set(range(self.num_layers)):
            raise RuntimeError("AISSD staged prefill did not execute all model layers")

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
        prefill = getattr(self, "_resident_prefill_current", None)
        if metadata.aissd_resident_archive_requests or prefill:
            # Ensure compute and the producer's GPU reads have finished before
            # the scheduler is allowed to transfer/free the original ownership.
            torch.cuda.synchronize(self.device)
        self._resident_check_prefill_forward()
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
                if source.get("staged") and not source["final"]:
                    ready[req_id] = source["prefix_tokens"]
                    continue
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
                if getattr(self, "_gpu_coarse", None) is not None:
                    from lmcache.integration.vllm.gpu_coarse_selector import build_summary
                    try:
                        build_started = time.perf_counter()
                        if source.get("staged"):
                            summary = self._resident_archive_pq(self._resident_archives[req_id])
                        else:
                            capture = self._resident_begin_coarse_diagnostic(req_id, source, raw)
                            diagnostic_kwargs = {} if capture is None else {"diagnostic": capture}
                            if self._gpu_coarse.options.algorithm != "segment_minmax_qk_max":
                                diagnostic_kwargs["algorithm_options"] = self._gpu_coarse.options
                            summary = build_summary(
                                self.kv_caches, source["coarse_block_ids"], source["prefix_tokens"],
                                self._lmcache_chunk_size, self._block_size, self.num_layers,
                                self._sparse_num_kv_heads, self._sparse_head_size,
                                self._gpu_coarse.options.segments, **diagnostic_kwargs)
                            if capture is not None:
                                capture.complete_summary(summary)
                    except Exception:
                        # A failed build must never leave a ready-looking
                        # archive which a later callback could acknowledge.
                        self._resident_archives.pop(req_id, None)
                        diagnostic = getattr(self._gpu_coarse, "diagnostic", None)
                        if diagnostic is not None:
                            diagnostic.abort_request(req_id)
                        raise
                    self._resident_archives[req_id]["coarse_summary"] = summary
                    logger.info("[aissd-gpu-coarse-summary] req=%s chunks=%d layers=%d "
                                "algorithm=%s segments=%d summary_bytes=%d build_ms=%.3f kv_page_budget_unchanged=1",
                                req_id, summary.num_chunks, len(summary.layers),
                                self._gpu_coarse.options.algorithm, self._gpu_coarse.options.segments,
                                summary.nbytes, (time.perf_counter() - build_started) * 1000)
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
        if self._resident_current or prefill:
            torch.cuda.synchronize(self.device)
        if prefill:
            state = prefill["state"]
            stager = self._resident_prefill_stager
            logger.info("[aissd-prefill-step] req=%s computed=%d archived_before=%d "
                        "ready=%d owned_blocks=%d released_blocks=%d staged_read_bytes=%d "
                        "workspace_bytes=%d peak_allocated_bytes=%d elapsed_ms=%.3f",
                        prefill["req_id"], state["total_tokens"], state["prefix_tokens"],
                        ready.get(prefill["req_id"], 0), state["owned_blocks"],
                        state["released_blocks"], prefill["read_bytes"],
                        stager.allocated_bytes, torch.cuda.max_memory_allocated(self.device),
                        (time.perf_counter() - prefill["started"]) * 1000)
        return ResidentWorkerMetadata(ready)

    def _resident_start_step(self, metadata):
        if not self._resident_budget:
            return
        import torch
        for req_id in metadata.aissd_resident_reset_ids:
            diagnostic = getattr(getattr(self, "_gpu_coarse", None), "diagnostic", None)
            if diagnostic is not None:
                diagnostic.close_request(req_id)
            self._resident_archives.pop(req_id, None)
            self._resident_worker.pop(req_id, None)
        states = metadata.aissd_resident_states
        self._resident_prefill_current = None
        prefill = {req: state for req, state in states.items() if state.get("prefill")}
        self._resident_current = {req: state for req, state in states.items() if not state.get("prefill")}
        if prefill:
            self._resident_begin_prefill(metadata, prefill)
        else:
            # The previous eager forward was fenced before the scheduler could
            # advance. Its one-layer workspace is not needed during decode.
            self._resident_prefill_stager = None
        context = getattr(self, "_sparse_current_step_context", None)
        if context is not None:
            context.pop("aissd_resident_loader", None)
            context.pop("aissd_gpu_coarse_prepare", None)
            context.pop("aissd_gpu_coarse_finish", None)
            context.pop("aissd_gpu_coarse_required", None)
            context["aissd_resident_active"] = False
        for req_id, state in self._resident_current.items():
            if req_id not in self._resident_archives:
                raise RuntimeError("Resident release occurred before worker archive readiness")
            per_chunk = self._lmcache_chunk_size // self._block_size
            capacity = len(state["hot_block_ids"]) // per_chunk
            worker = self._resident_worker.get(req_id)
            if worker is None or worker["epoch"] != state["epoch"]:
                self._resident_worker[req_id] = {"epoch": state["epoch"], "layers": {
                    layer: LayerLRU(capacity, bootstrap=state.get("bootstrap", True))
                    for layer in range(self.num_layers)
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
            if getattr(self, "_gpu_coarse", None) is not None:
                context["aissd_gpu_coarse_required"] = True
                context["aissd_gpu_coarse_prepare"] = self._resident_coarse_prepare
                context["aissd_gpu_coarse_finish"] = self._gpu_coarse.finish
            for req_id in self._resident_current:
                self._resident_worker[req_id]["context_built"] = True

    def _resident_begin_coarse_diagnostic(self, req_id, source, raw):
        diagnostic = getattr(self._gpu_coarse, "diagnostic", None)
        if diagnostic is None or not diagnostic.wants_request(req_id):
            return None
        chunks = [raw[i] for i in sorted(raw)]
        sidecars = {}
        for layer in diagnostic.options.layers:
            rewritten, mode, bucket = self._aissd_maybe_rewrite_chunks_to_qkpack_sidecars(
                req_id, chunks, layer_id=layer)
            if mode != f"qkpack_c{bucket}":
                raise RuntimeError("GPU coarse diagnostic requires validated sidecar inputs")
            sidecars[layer] = rewritten
        return diagnostic.start_request(
            req_id, source, raw, sidecars, layer_read_layout,
            num_layers=self.num_layers, chunk_size=self._lmcache_chunk_size,
            block_size=self._block_size, kv_heads=self._sparse_num_kv_heads,
            head_dim=self._sparse_head_size)

    def _resident_coarse_prepare(self, context, query, layer, num_heads, attention_scale):
        if len(self._resident_current) != 1:
            raise RuntimeError("GPU coarse v1 requires exactly one resident request")
        req_id = next(iter(self._resident_current))
        archive = self._resident_archives[req_id]
        if "coarse_summary" not in archive:
            raise RuntimeError("GPU coarse summary was not built before page release")
        return self._gpu_coarse.prepare(
            context, query, layer, num_heads, attention_scale,
            archive["coarse_summary"], req_id, self._resident_worker[req_id]["step"])

    def _resident_read_layer(self, backend, manifest, layer, kv_cache, block_ids,
                             *, page_indices=None):
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
        if page_indices is None:
            shaped = scratch.reshape(2, chunk // self._block_size, self._block_size,
                                     self._sparse_num_kv_heads, self._sparse_head_size)
            for index, block_id in enumerate(block_ids):
                paged_cache[block_id].copy_(shaped[:, index])
        else:
            from lmcache.integration.vllm.aissd_page_access import scatter_kv_pages
            scatter_kv_pages(paged_cache, page_indices, scratch.view(
                2, chunk, self._sparse_num_kv_heads, self._sparse_head_size))
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
        batched = getattr(self, "_resident_page_access", "legacy") == "batched"
        page_indices = None

        def load(chunk, slot):
            nonlocal page_indices
            start, end = slot * per_chunk, (slot + 1) * per_chunk
            blocks = hot[start:end]
            kwargs = {}
            if batched:
                if page_indices is None:
                    from lmcache.integration.vllm.aissd_page_access import (
                        make_page_index, page_major,
                    )
                    paged = page_major(kv_cache, self._block_size,
                                       self._sparse_num_kv_heads, self._sparse_head_size)
                    identity = (tuple(hot), paged.shape[0], kv_cache.device)
                    entry = worker.get("batched_page_index")
                    if entry is None or entry[0] != identity:
                        entry = (identity, make_page_index(
                            hot, paged.shape[0], kv_cache.device,
                            expected=worker["layers"][layer].capacity * per_chunk))
                        worker["batched_page_index"] = entry
                    page_indices = entry[1]
                kwargs["page_indices"] = page_indices[start:end]
            loaded = self._resident_read_layer(
                archive["backend"], archive["raw"][chunk], layer, kv_cache, blocks,
                **kwargs)
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
