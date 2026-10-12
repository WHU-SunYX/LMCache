# SPDX-License-Identifier: Apache-2.0
"""Opt-in, bounded GDS staging for the single-GPU vLLM path.

The banks are leases from the EXISTING CuFileMemoryAllocator, not additional
GPU allocations. Operations are serialized, so reads and writes share one
budget. CUDA copies overlap the other bank's blocking GDS I/O. Leases are
never recycled before every consumer has finished. Store completion can be
either drained (the compatibility default) or bounded asynchronous.
"""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
import ctypes
import math
import threading
import time

import torch

from lmcache.logging import init_logger
from lmcache.utils import CacheStoreEvent, convert_tokens_to_list
from lmcache.v1.memory_management import CuFileMemoryAllocator, MemoryFormat
from lmcache.v1.storage_backend.gds_backend import GdsBackend, _METADATA_MAX_SIZE

logger = init_logger(__name__)


def wait_all(futures):
    """Drain every submission before propagating the first error.

    Cancelling a Python future does not cancel an in-flight cuFile operation.
    These futures remain private and are never cancelled by this pipeline.
    """
    first_error = None
    results = []
    for future in futures:
        try:
            results.append(future.result())
        except BaseException as exc:
            results.append(None)
            if first_error is None:
                first_error = exc
    if first_error is not None:
        raise first_error
    return results


@dataclass(frozen=True)
class Chunk:
    key: object
    start: int
    end: int
    shape: object
    dtype: object
    fmt: object
    nbytes: int
    path: str = ""
    file_offset: int = _METADATA_MAX_SIZE

    @property
    def allocation_bytes(self):
        return (self.nbytes + 4095) // 4096 * 4096


@dataclass
class PendingStoreBank:
    chunks: list
    objects: list
    io_futures: list
    req_id: str
    started: float
    deferred_publish: bool
    drain_future: object = None


def plan_batches(chunks, capacity, reserve_ratio=0.1, max_chunks=0):
    """Plan from actual, aligned chunk bytes, including a partial last chunk."""
    if not 0 <= reserve_ratio < 1 or capacity <= 0 or max_chunks < 0:
        raise ValueError("Invalid GDS pipeline capacity/reserve/max_batch_chunks")
    usable = int(capacity * (1 - reserve_ratio)) // 4096 * 4096
    if not chunks:
        return [], 2, usable // 2
    largest = max(c.allocation_bytes for c in chunks)
    if largest <= 0 or largest > usable:
        raise ValueError(
            f"GDS pipeline cannot fit one chunk: chunk={largest} bytes, "
            f"usable={usable} bytes. Increase gds_buffer_size or reduce reserve."
        )
    banks = 2 if largest <= usable // 2 else 1
    budget = usable // banks // 4096 * 4096
    batches, batch, size = [], [], 0
    for chunk in chunks:
        if batch and (
            size + chunk.allocation_bytes > budget
            or (max_chunks and len(batch) >= max_chunks)
        ):
            batches.append(batch)
            batch, size = [], 0
        batch.append(chunk)
        size += chunk.allocation_bytes
    if batch:
        batches.append(batch)
    return batches, banks, budget


class BoundedGdsPipeline:
    def __init__(self, engine):
        self.engine = engine
        self.manager = engine.storage_manager
        self.gpu = engine.gpu_connector
        self._lock = threading.RLock()
        self._closed = False
        self._quarantined = []
        self._poisoned = False
        self._pending_stores = []
        self._async_error = None
        self._async_error_lock = threading.Lock()
        cfg = engine.config
        reasons = []
        for name, enabled in (
            ("use_layerwise", engine.use_layerwise),
            ("enable_async_loading", engine.async_loading),
            ("save_only_first_rank", engine.save_only_first_rank),
            ("enable_pd", cfg.enable_pd),
            ("enable_blending", cfg.enable_blending),
            ("local_cpu", cfg.local_cpu),
            ("MLA", engine.metadata.use_mla),
            ("world_size != 1", engine.metadata.world_size != 1),
            ("remove_after_retrieve", engine.remove_after_retrieve),
        ):
            if enabled:
                reasons.append(name)
        if type(self.gpu).__name__ not in (
            "VLLMPagedMemGPUConnectorV2", "VLLMPagedMemGPUConnectorV3"
        ):
            reasons.append("GPU connector is not vLLM paged V2/V3")
        if reasons:
            raise ValueError(
                "gds_enable_bounded_pipeline does not support: " + ", ".join(reasons)
            )
        if self.manager is None:
            raise ValueError("GDS pipeline requires a worker StorageManager")
        candidates = [
            (name, backend) for name, backend in self.manager.storage_backends.items()
            if isinstance(backend, GdsBackend)
        ]
        if len(candidates) != 1:
            raise ValueError("GDS pipeline requires exactly one GdsBackend")
        self.backend_name, self.backend = candidates[0]
        if not self.backend.use_gds or self.backend.gds_backend != "cufile":
            raise ValueError("GDS pipeline requires use_gds=true, gds_backend=cufile")
        self.allocator = self.backend.memory_allocator
        if not isinstance(self.allocator, CuFileMemoryAllocator):
            raise ValueError("GDS pipeline requires the registered CuFile GPU pool")
        self.device = self.allocator.tensor.device
        self.capacity = min(
            int(cfg.gds_buffer_size * 1024**2), self.allocator.tensor.numel()
        )
        extra = cfg.extra_config or {}
        self.reserve_ratio = float(extra.get("gds_pipeline_reserve_ratio", 0.10))
        self.max_chunks = int(extra.get("gds_pipeline_max_batch_chunks", 0))
        self.store_completion = str(
            extra.get("gds_pipeline_store_completion", "drained")
        ).strip().lower()
        if self.store_completion not in ("drained", "async"):
            raise ValueError(
                "gds_pipeline_store_completion must be 'drained' or 'async'"
            )
        self.raw_reader_visibility = str(
            extra.get("gds_raw_reader_visibility", "none")
        ).strip().lower()
        if self.raw_reader_visibility not in ("none", "syncfs"):
            raise ValueError(
                "gds_raw_reader_visibility must be 'none' or 'syncfs'"
            )
        if self.store_completion == "async" and engine.kv_events_enabled:
            raise ValueError(
                "gds_pipeline_store_completion=async does not support KV events; "
                "use drained completion or disable enable_kv_events"
            )
        plan_batches([], self.capacity, self.reserve_ratio, self.max_chunks)
        self._validate_backend()
        # These workers only wait for already-submitted backend futures and
        # release their bank leases. They never issue CUDA copies.
        self._store_reaper = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="lmcache-gds-bank"
        )
        logger.info(
            "[GDS_PIPE] enabled pool=%.1fMiB reserve=%.1f%% max_batch_chunks=%d "
            "shared_read_write_pool=true store_completion=%s "
            "raw_reader_visibility=%s cache_mode=%s",
            self.capacity / 1024**2, 100 * self.reserve_ratio, self.max_chunks,
            self.store_completion, self.raw_reader_visibility,
            engine._gds_cache_mode,
        )

    def _validate_backend(self):
        # The disabled CPU backend may exist only as the native allocator.
        # Bypass it here: caching a reference there would pin GDS leases.
        for name, backend in self.manager.storage_backends.items():
            if backend is self.backend:
                continue
            if name == "LocalCPUBackend" and not self.engine.config.local_cpu:
                continue
            raise ValueError(
                f"GDS pipeline requires GDS-only storage; additional backend: {name}"
            )
        if self.manager.storage_backends.get(self.backend_name) is not self.backend:
            raise RuntimeError("GDS backend changed; restart before using the pipeline")
        if self.engine.store_location not in (None, self.backend_name):
            raise ValueError("GDS pipeline store_location must select GdsBackend")
        locations = self.engine.retrieve_locations
        if locations and set(locations) != {self.backend_name}:
            raise ValueError("GDS pipeline retrieve_locations must select GdsBackend")

    @contextmanager
    def _operation(self):
        # Also exclude dynamic backend recreation while leases are alive.
        with self._lock, self.manager.manager_lock:
            if self._closed or self._poisoned:
                raise RuntimeError("GDS pipeline is closed/poisoned; restart the worker")
            self._reap_store_banks(wait=False)
            self._validate_backend()
            if not any(
                b is self.backend
                for _, b in self.manager.get_active_storage_backends(self.backend_name)
            ):
                raise RuntimeError("GDS pipeline backend is frozen or bypassed")
            with torch.cuda.device(self.device):
                yield

    def close(self):
        with self._lock, self.manager.manager_lock:
            if self._closed:
                return
            try:
                self._reap_store_banks(wait=True)
            except BaseException as exc:
                logger.error(
                    "[GDS_PIPE] pending asynchronous store failed during close: "
                    "%s: %s",
                    type(exc).__name__, exc,
                )
            finally:
                self._closed = True
                self._store_reaper.shutdown(wait=True)

    def flush(self):
        """Wait for all accepted stores and surface any asynchronous failure."""
        with self._lock, self.manager.manager_lock:
            if self._closed:
                raise RuntimeError("GDS pipeline is closed")
            self._reap_store_banks(wait=True)

    def raise_if_async_error(self):
        """Fail a lookup before it can consume a partially failed async store."""
        with self._lock:
            if self._closed or self._poisoned:
                raise RuntimeError("GDS pipeline is closed/poisoned; restart the worker")
            self._reap_store_banks(wait=False)

    def _chunk(self, key, start, end, shape, dtype, fmt, path="", offset=None):
        shape = torch.Size(shape)
        if not shape or any(d <= 0 for d in shape) or start < 0 or end <= start:
            raise ValueError("Invalid GDS chunk geometry or token interval")
        if fmt != MemoryFormat.KV_2LTD:
            raise ValueError("GDS pipeline currently requires homogeneous KV_2LTD")
        if len(shape) != 4 or shape[0] != 2 or shape[2] != end - start:
            raise ValueError("GDS chunk shape does not match its token interval")
        expected_shapes = self.engine.metadata.get_shapes(end - start)
        expected_dtypes = self.engine.metadata.get_dtypes()
        if (
            len(expected_shapes) != 1 or len(expected_dtypes) != 1
            or shape != expected_shapes[0] or dtype != expected_dtypes[0]
        ):
            raise ValueError("GDS chunk does not match this model's KV shape/dtype")
        nbytes = math.prod(shape) * dtype.itemsize
        return Chunk(key, start, end, shape, dtype, fmt, nbytes, path,
                     _METADATA_MAX_SIZE if offset is None else offset)

    def _plan(self, chunks, operation):
        batches, banks, budget = plan_batches(
            chunks, self.capacity, self.reserve_ratio, self.max_chunks
        )
        logger.info(
            "[GDS_PIPE] op=%s chunks=%d batches=%d banks=%d bank_budget=%.1fMiB",
            operation, len(chunks), len(batches), banks, budget / 1024**2,
        )
        return batches, banks

    def _allocate(self, chunks):
        objects = []
        try:
            for chunk in chunks:
                obj = self.allocator.allocate(chunk.shape, chunk.dtype, fmt=chunk.fmt)
                if obj is None:
                    raise RuntimeError(
                        "GDS pipeline allocation failed within its byte budget; "
                        "check other pool users/fragmentation. No chunks were skipped."
                    )
                objects.append(obj)
            return objects
        except BaseException:
            self._release(objects)
            raise

    @staticmethod
    def _release(objects):
        for obj in objects:
            obj.ref_count_down()
        objects.clear()

    def _capture_store_bank_failure(self, future):
        try:
            future.result()
        except BaseException as exc:
            # Drained mode reports the same exception directly on the caller.
            # Async mode needs a latch because its original store() has already
            # returned and the failure must surface on the next operation.
            if self.store_completion == "async":
                with self._async_error_lock:
                    if self._async_error is None:
                        self._async_error = exc
                logger.error(
                    "[GDS_PIPE] asynchronous store bank failed: %s: %s",
                    type(exc).__name__, exc,
                )

    def _finish_deferred_publish(self, bank, publish):
        first_error = None
        for chunk, obj, io_future in zip(
            bank.chunks, bank.objects, bank.io_futures, strict=True
        ):
            try:
                self.backend.finish_bounded_put(
                    chunk.key,
                    obj,
                    publish=publish,
                    owner=getattr(io_future, "_lmcache_bounded_owner", None),
                )
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    def _drain_store_bank(self, bank):
        barrier_ms = 0.0
        published = False
        try:
            wait_all(bank.io_futures)
            if bank.deferred_publish:
                barrier_started = time.perf_counter()
                self.backend.bounded_raw_reader_visibility_barrier(
                    self.raw_reader_visibility
                )
                barrier_ms = (time.perf_counter() - barrier_started) * 1000
                self._finish_deferred_publish(bank, publish=True)
                published = True
            elapsed = time.perf_counter() - bank.started
            total = sum(chunk.nbytes for chunk in bank.chunks)
            logger.info(
                "[GDS_PIPE] op=store_commit req_id=%s completed_chunks=%d "
                "bytes=%d commit_ms=%.3f barrier_ms=%.3f "
                "raw_reader_visibility=%s",
                bank.req_id, len(bank.chunks), total, elapsed * 1000,
                barrier_ms, self.raw_reader_visibility,
            )
        except BaseException:
            if bank.deferred_publish and not published:
                try:
                    self._finish_deferred_publish(bank, publish=False)
                except BaseException as cleanup_error:
                    logger.error(
                        "[GDS_PIPE] failed to abort deferred publication: %s: %s",
                        type(cleanup_error).__name__, cleanup_error,
                    )
            raise
        finally:
            self._release(bank.objects)

    def _submit_store_bank(self, chunks, objects, io_futures, req_id):
        bank = PendingStoreBank(
            chunks=list(chunks), objects=objects, io_futures=io_futures,
            req_id=req_id, started=time.perf_counter(),
            deferred_publish=self.raw_reader_visibility != "none",
        )
        try:
            bank.drain_future = self._store_reaper.submit(
                self._drain_store_bank, bank
            )
        except BaseException:
            # Backend submissions may already reference these leases.
            self._drain_store_bank(bank)
            raise
        bank.drain_future.add_done_callback(self._capture_store_bank_failure)
        self._pending_stores.append(bank)

    def _reap_store_banks(self, wait):
        first_error = None
        remaining = []
        for bank in self._pending_stores:
            if wait or bank.drain_future.done():
                try:
                    bank.drain_future.result()
                except BaseException as exc:
                    if first_error is None:
                        first_error = exc
            else:
                remaining.append(bank)
        self._pending_stores = remaining
        if first_error is not None:
            with self._async_error_lock:
                self._async_error = None
            self._poisoned = True
            raise first_error
        with self._async_error_lock:
            async_error = self._async_error
        if async_error is not None:
            self._poisoned = True
            raise RuntimeError(
                "A bounded asynchronous GDS store failed; restart the worker"
            ) from async_error

    def _wait_for_store_slot(self, banks):
        self._reap_store_banks(wait=False)
        while len(self._pending_stores) >= banks:
            bank = self._pending_stores.pop(0)
            try:
                bank.drain_future.result()
            except BaseException as exc:
                self._poisoned = True
                # A failed bank must not cause later banks to be abandoned;
                # cuFile work cannot be cancelled and still owns its leases.
                try:
                    self._reap_store_banks(wait=True)
                except BaseException:
                    pass
                raise exc

    def _allocate_store_bank(self, chunks):
        while True:
            try:
                return self._allocate(chunks)
            except RuntimeError:
                # A prior single-bank batch can be larger than the current
                # two-bank budget. Drain it and retry before calling this
                # allocator pressure or fragmentation.
                if not self._pending_stores:
                    raise
                bank = self._pending_stores.pop(0)
                try:
                    bank.drain_future.result()
                except BaseException as exc:
                    self._poisoned = True
                    try:
                        self._reap_store_banks(wait=True)
                    except BaseException:
                        pass
                    raise exc

    def _copy(self, chunks, objects, direction, kwargs):
        stream = self.gpu.load_stream if direction == "read" else self.gpu.store_stream
        mapping = kwargs.get("slot_mapping")
        if mapping is None or mapping.numel() < max(c.end for c in chunks):
            raise ValueError("GDS pipeline requires a full-length slot_mapping")
        # slot_mapping and source KV may have just been produced on the caller's
        # stream. Pointer initialization and the copy execute on the same stream.
        stream.wait_stream(torch.cuda.current_stream(self.device))
        fn = self.gpu.to_gpu if direction == "read" else self.gpu.from_gpu
        with torch.cuda.stream(stream):
            for chunk, obj in zip(chunks, objects, strict=True):
                fn(obj, chunk.start, chunk.end, **kwargs)
            done = torch.cuda.Event()
            done.record(stream)
        return done

    def _read_one(self, chunk, obj):
        with torch.cuda.device(self.device), torch.inference_mode():
            result = self.backend._load_gds(
                chunk.path, chunk.file_offset,
                ctypes.c_void_p(self.backend.gds_base_pointer),
                chunk.nbytes, obj.metadata.address,
            )
        if result != chunk.nbytes:
            raise OSError(
                f"GDS pipeline short/failed read: {chunk.path}: "
                f"got {result}, expected {chunk.nbytes}"
            )

    def _read_batch(self, chunks, objects):
        futures = []
        try:
            for chunk, obj in zip(chunks, objects, strict=True):
                futures.append(self.backend._thread_pool.submit(self._read_one, chunk, obj))
        finally:
            wait_all(futures)

    def _cleanup(self, direction, owned):
        # An exception may happen after some kernels launched, before an event
        # could be recorded. Do not recycle any lease until that stream drains.
        stream = self.gpu.load_stream if direction == "read" else self.gpu.store_stream
        try:
            stream.synchronize()
        except BaseException:
            self._poisoned = True
            for objects in owned:
                self._quarantined.extend(objects)
                objects.clear()
            raise
        finally:
            if not self._poisoned:
                for objects in owned:
                    self._release(objects)

    def _read(self, chunks, ret_mask, kwargs, operation):
        batches, banks = self._plan(chunks, operation)
        if not batches:
            return 0
        owned, pending = [], None
        total = 0

        def finish():
            nonlocal pending, total
            if pending is None:
                return
            batch, objects, event = pending
            event.synchronize()
            for chunk in batch:
                ret_mask[chunk.start:min(chunk.end, len(ret_mask))] = True
                total += chunk.nbytes
            self._release(objects)
            pending = None

        try:
            for batch in batches:
                if banks == 1:
                    finish()
                objects = self._allocate(batch)
                owned.append(objects)
                # Previous scatter is running while this batch reads from SSD.
                self._read_batch(batch, objects)
                finish()
                event = self._copy(batch, objects, "read", kwargs)
                pending = (batch, objects, event)
            finish()
        finally:
            self._cleanup("read", owned)
        return total

    def retrieve(self, tokens, mask, ret_mask, **kwargs):
        started = time.perf_counter()
        with self._operation():
            # Store jobs share the registered pool. Drain their tail before
            # allocating read banks; model execution between calls has already
            # provided the intended overlap window.
            self._reap_store_banks(wait=True)
            chunks = []
            for start, end, key in self.engine.token_database.process_tokens(
                tokens=tokens, mask=mask, request_configs=kwargs.get("request_configs")
            ):
                # Preserve contiguous-prefix semantics; stop at the first miss.
                if not self.backend.contains(key):
                    break
                with self.backend.hot_lock:
                    entry = self.backend.hot_cache.get(key)
                if entry is None:
                    break
                chunks.append(self._chunk(
                    key, start, end, entry.shape, entry.dtype, entry.fmt, entry.path
                ))
            total = self._read(chunks, ret_mask, kwargs, "retrieve")
        self._log("retrieve", total, len(chunks), started, kwargs)
        return total

    def retrieve_selected(self, manifest, **kwargs):
        started = time.perf_counter()
        with self._operation():
            self._reap_store_banks(wait=True)
            chunks = []
            for selected in manifest.get("selected_chunks", []):
                if selected.get("backend") != self.backend_name:
                    raise ValueError("Sparse selected chunk is not from this GdsBackend")
                chunk = self._chunk(
                    None, int(selected["token_start"]), int(selected["token_end"]),
                    selected["shape"],
                    self.backend._parse_sparse_manifest_dtype(selected["dtype"]),
                    self.backend._parse_sparse_manifest_fmt(selected["fmt"]),
                    selected["path"], int(selected.get("file_offset", _METADATA_MAX_SIZE)),
                )
                if int(selected["nbytes"]) != chunk.nbytes or chunk.file_offset < 0:
                    raise ValueError("Sparse selected chunk byte count/offset mismatch")
                chunks.append(chunk)
            tokens_len = int(kwargs.get("tokens_len", 0))
            if tokens_len <= 0:
                tokens_len = max((c.end for c in chunks), default=0)
            ret_mask = torch.zeros(tokens_len, dtype=torch.bool, device="cpu")
            total = self._read(chunks, ret_mask, kwargs, "selected")
        duration_ms = (time.perf_counter() - started) * 1000
        manifest["__aissd_selected_load_stats"] = {
            "selected_chunks": len(chunks), "loaded_chunks": len(chunks),
            "bytes": total, "duration_ms": duration_ms,
            "bw_GBps": total / (duration_ms * 1e6) if duration_ms else 0.0,
        }
        self._log("selected", total, len(chunks), started, kwargs)
        return ret_mask

    def store(self, tokens, hashes, offsets, mask, **kwargs):
        started = time.perf_counter()
        if kwargs.get("transfer_spec") is not None:
            raise ValueError("GDS bounded pipeline does not support PD transfer_spec")
        with self._operation():
            chunks = []
            for start, end, key in self.engine.token_database.process_tokens(
                tokens, hashes, offsets, mask, request_configs=kwargs.get("request_configs")
            ):
                shapes = self.engine.metadata.get_shapes(end - start)
                dtypes = self.engine.metadata.get_dtypes()
                if len(shapes) != 1 or len(dtypes) != 1:
                    raise ValueError("GDS bounded store requires homogeneous single-group KV")
                chunks.append(self._chunk(
                    key, start, end, shapes[0], dtypes[0],
                    self.engine.fmt or MemoryFormat.KV_2LTD,
                ))
            batches, banks = self._plan(chunks, "store")
            if not batches:
                return 0
            req_id = str(kwargs.get("req_id", "unspecified"))
            deferred_publish = self.raw_reader_visibility != "none"
            for batch in batches:
                self._wait_for_store_slot(banks)
                objects = self._allocate_store_bank(batch)
                try:
                    # Earlier SSD writes run while this bank gathers GPU KV.
                    self._copy(batch, objects, "write", kwargs).synchronize()
                except BaseException:
                    # CUDA could have accepted only a prefix of the gather
                    # kernels. Without a successful synchronization, reusing
                    # this bank is unsafe; quarantine it until worker restart.
                    self._poisoned = True
                    self._quarantined.extend(objects)
                    objects.clear()
                    try:
                        self._reap_store_banks(wait=True)
                    except BaseException:
                        pass
                    raise
                io_futures = []
                try:
                    for chunk, obj in zip(batch, objects, strict=True):
                        io_futures.append(self.backend.submit_bounded_put(
                            chunk.key, obj, defer_publish=deferred_publish
                        ))
                except BaseException:
                    # Never recycle a lease still referenced by an already
                    # submitted backend task. Deferred keys must leave the
                    # in-flight registry without becoming visible.
                    try:
                        wait_all(io_futures)
                    except BaseException:
                        pass
                    if deferred_publish:
                        partial = PendingStoreBank(
                            list(batch[:len(io_futures)]),
                            list(objects[:len(io_futures)]), io_futures, req_id,
                            time.perf_counter(), True,
                        )
                        try:
                            self._finish_deferred_publish(partial, publish=False)
                        except BaseException:
                            pass
                    self._release(objects)
                    try:
                        self._reap_store_banks(wait=True)
                    except BaseException:
                        pass
                    raise
                try:
                    self._submit_store_bank(batch, objects, io_futures, req_id)
                except BaseException:
                    try:
                        self._reap_store_banks(wait=True)
                    except BaseException:
                        pass
                    raise

            if self.store_completion == "drained":
                self._reap_store_banks(wait=True)
            self._store_events(chunks, tokens, hashes)
        total = sum(c.nbytes for c in chunks)
        self._log(
            "store", total, len(chunks), started, kwargs,
            completion=(
                "completed" if self.store_completion == "drained" else "submitted"
            ),
        )
        return sum(c.end - c.start for c in chunks)

    def _store_events(self, chunks, tokens, hashes):
        if not self.engine.kv_events_enabled:
            return
        previous_hash = 0
        for chunk in chunks:
            event = CacheStoreEvent(
                block_hashes=[chunk.key.chunk_hash],
                parent_block_hash=None if chunk.start == 0 else previous_hash,
                token_ids=[], block_size=chunk.end - chunk.start,
                lora_id=None, medium="cpu", lora_name=None,
            )
            if tokens is not None:
                event.token_ids = convert_tokens_to_list(tokens, chunk.start, chunk.end)
                if isinstance(tokens, torch.Tensor):
                    event.medium = tokens.device
            elif hashes is not None:
                event.token_ids = hashes[chunk.start:chunk.end + 1]
            self.engine.kv_events.append(event)
            previous_hash = chunk.key.chunk_hash

    @staticmethod
    def _log(operation, nbytes, chunks, started, kwargs, completion="completed"):
        elapsed = time.perf_counter() - started
        logger.info(
            "[GDS_PIPE] op=%s req_id=%s %s_chunks=%d bytes=%d "
            "wall_ms=%.3f effective_GiBps=%.4f",
            operation, kwargs.get("req_id", "unspecified"), completion,
            chunks, nbytes, elapsed * 1000,
            nbytes / 1024**3 / elapsed if elapsed else 0.0,
        )
