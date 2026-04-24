# SPDX-License-Identifier: Apache-2.0
# Standard
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Sequence
from dataclasses import dataclass
import asyncio
import os
import threading
import time

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor
from lmcache.utils import CacheEngineKey, DiskCacheMetadata, _lmcache_nvtx_annotate
from lmcache.v1.cache_controller.message import OpType
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface
from lmcache.v1.storage_backend.batched_message_sender import BatchedMessageSender
from lmcache.v1.storage_backend.cache_policy import get_cache_policy
from lmcache.v1.storage_backend.job_executor.pq_executor import (
    AsyncPQThreadPoolExecutor,
)
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.path_sharder import PathSharder

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.cache_controller.worker import LMCacheWorker

logger = init_logger(__name__)


class LocalAiSSDWorker:
    """
    Worker for LocalAiSSDBackend.

    This intentionally mirrors LocalDiskWorker so the backend can be swapped in
    with minimal coupling. Today it still uses local file I/O; later you can
    replace submit_task()/write/read internals with SSD-CPU service or DMA/RPC
    without touching the rest of LMCache.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop, max_workers: int = 4) -> None:
        self.put_lock = threading.Lock()
        self.put_tasks: List[CacheEngineKey] = []

        self.prefetch_lock = threading.Lock()
        self.prefetch_tasks: dict[CacheEngineKey, Future] = {}

        self.executor = AsyncPQThreadPoolExecutor(loop, max_workers=max_workers)
        self.loop = loop
        self._closed = False

    async def submit_task(
        self,
        task_type: str,
        task: Callable,
        *args,
        **kwargs,
    ) -> Any:
        if task_type == "prefetch":
            priority = 0
        elif task_type == "delete":
            priority = 1
        elif task_type == "put":
            priority = 2
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        return await self.executor.submit_job(
            task,
            *args,
            priority=priority,
            **kwargs,
        )

    def remove_put_task(self, key: CacheEngineKey) -> None:
        with self.put_lock:
            if key in self.put_tasks:
                self.put_tasks.remove(key)
            else:
                logger.warning("Key %s not found in put tasks.", key)

    def insert_put_task(self, key: CacheEngineKey) -> None:
        with self.put_lock:
            self.put_tasks.append(key)

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        with self.put_lock:
            return key in self.put_tasks

    def get_put_queue_depth(self) -> int:
        with self.put_lock:
            return len(self.put_tasks)

    def get_prefetch_queue_depth(self) -> int:
        with self.prefetch_lock:
            return len(self.prefetch_tasks)

    def close(self) -> bool:
        if self._closed:
            return False
        self._closed = True
        self.executor.shutdown(wait=True)
        return True


@dataclass
class PendingAppendBatch:
    keys: list[CacheEngineKey]
    memory_objs: list[MemoryObj]
    on_complete_callback: Optional[Callable[[CacheEngineKey], None]]
    enqueue_ts: float




class LocalAiSSDBackend(StorageBackendInterface):
    """
    Drop-in backend for experimenting with AI SSD data-plane optimizations.

    Design goals:
    - Keep the public backend contract aligned with LocalDiskBackend.
    - Minimize coupling with upstream LMCache by living in a separate file.
    - Reserve clear hook points for future SSD-CPU service / RPC / DMA paths.

    Current implementation still uses local file I/O, but all tracing/log names,
    queue management, and configuration keys are isolated under LocalAiSSDBackend.
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
        dst_device: str = "cuda",
        lmcache_worker: Optional["LMCacheWorker"] = None,
        metadata: Optional[LMCacheMetadata] = None,
    ):
        if torch.cuda.is_available():
            super().__init__(dst_device)
        else:
            super().__init__("cpu")

        self.cache_policy = get_cache_policy(config.cache_policy)
        self.dict = self.cache_policy.init_mutable_mapping()

        self.dst_device = dst_device
        self.local_cpu_backend = local_cpu_backend
        self.disk_lock = threading.Lock()

        self.config = config
        extra = config.extra_config or {}

        raw_path = extra.get("aissd.path", None) or config.local_disk
        if raw_path is None:
            raise ValueError(
                "LocalAiSSDBackend requires config.local_disk or extra_config['aissd.path']"
            )

        sharder = PathSharder(
            raw_csv=raw_path,
            strategy=extra.get("aissd.path_sharding", config.local_disk_path_sharding),
            dst_device=dst_device,
            create_dirs=True,
        )
        self.path: str = sharder.selected

        logger.info(
            "Local AI SSD cache path: %s (device %s, %d path(s) configured)",
            self.path,
            dst_device,
            len(sharder.all_paths),
        )

        self.loop = loop
        self.use_local_cpu = config.local_cpu

        stat = os.statvfs(self.path)
        self.os_disk_bs = stat.f_bsize
        self.use_odirect = bool(extra.get("aissd.use_odirect", extra.get("use_odirect", False)))
        logger.info("Using O_DIRECT for AI SSD I/O: %s", self.use_odirect)

        worker_count = int(extra.get("aissd.max_workers", 4))
        self.disk_worker = LocalAiSSDWorker(loop, max_workers=worker_count)
        self.segment_parallelism = max(1, int(extra.get("aissd.segment_parallelism", worker_count)))
        self.segment_min_items = max(1, int(extra.get("aissd.segment_min_items", 4)))

        self.max_cache_size = int(
            float(extra.get("aissd.max_cache_size_gb", config.max_local_disk_size))
            * 1024**3
        )
        self.current_cache_size = 0.0

        # Segment metadata for batched writes:
        # key -> (segment_path, offset, size)
        self.segment_index: dict[CacheEngineKey, tuple[str, int, int]] = {}
        # segment_path -> refcount
        self.segment_refcount: dict[str, int] = {}
        self.segment_lock = threading.Lock()
        self.segment_counter = 0

        # Single-file append log mode for batched writes
        self.append_log_path = os.path.join(
            self.path,
            str(extra.get("aissd.append_log_name", "kv_cache.log")),
        )
        self.append_lock = threading.Lock()
        self.append_io_lock = threading.Lock()
        self.append_offset = 0
        self.single_io_append = bool(extra.get("aissd.single_io_append", True))

        # Cross-request append coalescing
        self.append_batch_max_bytes = int(
            float(extra.get("aissd.append_batch_max_bytes", 2 * 1024 * 1024 * 1024))
        )
        self.append_batch_max_items = int(extra.get("aissd.append_batch_max_items", 64))
        self.append_batch_timeout_ms = float(extra.get("aissd.append_batch_timeout_ms", 2.0))
        self.append_queue: list[PendingAppendBatch] = []
        self.append_queue_bytes = 0
        self.append_queue_cv = threading.Condition()
        self.append_flush_stop = False
        self.append_flush_thread = threading.Thread(
            target=self._append_flush_loop,
            name="LocalAiSSDAppendFlush",
            daemon=True,
        )
        self.append_flush_thread.start()

        self.keys_in_request: List[CacheEngineKey] = []

        self.lmcache_worker = lmcache_worker
        self.instance_id = config.lmcache_instance_id
        self.stats_monitor = LMCStatsMonitor.GetOrCreate()
        self.usage = 0

        self.batched_msg_sender: Optional[BatchedMessageSender] = None
        if lmcache_worker and metadata is not None:
            self.batched_msg_sender = BatchedMessageSender(
                metadata=metadata,
                config=config,
                location=str(self),
                lmcache_worker=lmcache_worker,
            )
        else:
            logger.warning("Controller message sender is not initialized")

    def __str__(self) -> str:
        return "LocalAiSSDBackend"

    def _put_queue_depth(self) -> int:
        return self.disk_worker.get_put_queue_depth()

    def _prefetch_queue_depth(self) -> int:
        return self.disk_worker.get_prefetch_queue_depth()

    def _trace_queue_state(self, op: str, duration_ms: float, **kwargs: Any) -> None:
        self._trace_backend(
            op,
            duration_ms,
            put_queue_depth=self._put_queue_depth(),
            prefetch_queue_depth=self._prefetch_queue_depth(),
            **kwargs,
        )

    def _key_to_path(self, key: CacheEngineKey) -> str:
        return os.path.join(self.path, key.to_string().replace("/", "-") + ".pt")

    def _next_segment_path(self) -> str:
        with self.segment_lock:
            self.segment_counter += 1
            seg_id = self.segment_counter
        return os.path.join(self.path, f"segment_{seg_id:08d}.bin")

    def _reserve_append_region(self, size: int) -> tuple[str, int]:
        with self.append_lock:
            offset = self.append_offset
            self.append_offset += size
        return self.append_log_path, offset

    def _pwritev_backend(
        self,
        buffers: list[memoryview | bytes | bytearray],
        path: str,
        offset: int,
    ) -> bool:
        total_size = sum(len(buf) for buf in buffers)
        if total_size == 0:
            fd = os.open(path, os.O_CREAT | os.O_WRONLY, 0o644)
            os.close(fd)
            return False

        has_pwritev = hasattr(os, "pwritev")
        aligned = (
            offset % self.os_disk_bs == 0
            and all(len(buf) % self.os_disk_bs == 0 for buf in buffers)
        )

        if has_pwritev and self.use_odirect and aligned:
            try:
                fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_DIRECT, 0o644)
                try:
                    written = os.pwritev(fd, buffers, offset)
                    if written != total_size:
                        raise IOError(
                            f"Short pwritev to {path}: expected {total_size} bytes, got {written}"
                        )
                    return True
                finally:
                    os.close(fd)
            except OSError as e:
                logger.debug(
                    "O_DIRECT pwritev failed for %s offset=%d, fallback to buffered pwritev: %s",
                    path,
                    offset,
                    e,
                )

        if has_pwritev:
            fd = os.open(path, os.O_CREAT | os.O_WRONLY, 0o644)
            try:
                written = os.pwritev(fd, buffers, offset)
                if written != total_size:
                    raise IOError(
                        f"Short buffered pwritev to {path}: expected {total_size} bytes, got {written}"
                    )
                return False
            finally:
                os.close(fd)

        # Conservative fallback: serialize into one write at a fixed offset
        # Keep this path unlikely on modern Python.
        payload = b"".join(bytes(buf) for buf in buffers)
        fd = os.open(path, os.O_CREAT | os.O_WRONLY, 0o644)
        try:
            os.lseek(fd, offset, os.SEEK_SET)
            written = os.write(fd, payload)
            if written != len(payload):
                raise IOError(
                    f"Short fallback write to {path}: expected {len(payload)} bytes, got {written}"
                )
            return False
        finally:
            os.close(fd)

    def _split_into_segment_groups(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
    ) -> list[tuple[list[CacheEngineKey], list[MemoryObj]]]:
        num_items = len(memory_objs)
        if num_items == 0:
            return []

        target_groups = min(self.segment_parallelism, num_items)
        if num_items < self.segment_min_items * 2:
            target_groups = 1
        else:
            target_groups = min(target_groups, max(1, num_items // self.segment_min_items))

        base = num_items // target_groups
        rem = num_items % target_groups

        groups: list[tuple[list[CacheEngineKey], list[MemoryObj]]] = []
        start = 0
        for i in range(target_groups):
            group_size = base + (1 if i < rem else 0)
            end = start + group_size
            groups.append((list(keys[start:end]), list(memory_objs[start:end])))
            start = end
        return groups

    def _enqueue_append_batch(
        self,
        keys: list[CacheEngineKey],
        memory_objs: list[MemoryObj],
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]],
    ) -> None:
        batch_bytes = sum(len(memory_obj.byte_array) for memory_obj in memory_objs)
        pending = PendingAppendBatch(
            keys=keys,
            memory_objs=memory_objs,
            on_complete_callback=on_complete_callback,
            enqueue_ts=time.time(),
        )
        with self.append_queue_cv:
            self.append_queue.append(pending)
            self.append_queue_bytes += batch_bytes
            self.append_queue_cv.notify()

    def _drain_append_batches_locked(self) -> list[PendingAppendBatch]:
        if not self.append_queue:
            return []

        drained: list[PendingAppendBatch] = []
        drained_bytes = 0
        drained_items = 0
        while self.append_queue:
            nxt = self.append_queue[0]
            nxt_bytes = sum(len(memory_obj.byte_array) for memory_obj in nxt.memory_objs)
            nxt_items = len(nxt.memory_objs)

            if drained and (
                drained_bytes + nxt_bytes > self.append_batch_max_bytes
                or drained_items + nxt_items > self.append_batch_max_items
            ):
                break

            drained.append(self.append_queue.pop(0))
            self.append_queue_bytes -= nxt_bytes
            drained_bytes += nxt_bytes
            drained_items += nxt_items

            if drained_bytes >= self.append_batch_max_bytes or drained_items >= self.append_batch_max_items:
                break

        return drained

    def _flush_pending_batches(self, batches: list[PendingAppendBatch]) -> None:
        if not batches:
            return

        start_time = self._trace_now()
        flat_keys: list[CacheEngineKey] = []
        flat_memory_objs: list[MemoryObj] = []
        callbacks: list[Optional[Callable[[CacheEngineKey], None]]] = []

        total_size = 0
        oldest_enqueue_ts = batches[0].enqueue_ts
        for batch in batches:
            flat_keys.extend(batch.keys)
            flat_memory_objs.extend(batch.memory_objs)
            callbacks.extend([batch.on_complete_callback] * len(batch.keys))
            total_size += sum(len(memory_obj.byte_array) for memory_obj in batch.memory_objs)
            if batch.enqueue_ts < oldest_enqueue_ts:
                oldest_enqueue_ts = batch.enqueue_ts

        rel_offset = 0
        key_records: list[tuple[CacheEngineKey, int, int, MemoryObj, Optional[Callable[[CacheEngineKey], None]]]] = []
        chunk_views: list[memoryview | bytes | bytearray] = []

        for key, memory_obj, cb in zip(flat_keys, flat_memory_objs, callbacks, strict=False):
            chunk = memory_obj.byte_array
            chunk_size = len(chunk)
            chunk_views.append(chunk)
            key_records.append((key, rel_offset, chunk_size, memory_obj, cb))
            rel_offset += chunk_size

        self.usage += total_size
        self.stats_monitor.update_local_storage_usage(self.usage)

        queue_wait_start = self._trace_now()
        with self.append_io_lock:
            io_lock_wait_ms = (self._trace_now() - queue_wait_start) * 1000.0
            log_path, base_offset = self._reserve_append_region(total_size)
            self.appendv_file(chunk_views, log_path, base_offset)

        now = time.time()
        oldest_wait_ms = (now - oldest_enqueue_ts) * 1000.0

        for key, chunk_offset, chunk_size, memory_obj, cb in key_records:
            shape = memory_obj.metadata.shape
            dtype = memory_obj.metadata.dtype
            fmt = memory_obj.metadata.fmt
            cached_positions = memory_obj.metadata.cached_positions
            memory_obj.ref_count_down()

            self.insert_key(
                key,
                chunk_size,
                shape,
                dtype,
                fmt,
                cached_positions=cached_positions,
                path=log_path,
                offset=base_offset + chunk_offset,
            )
            self.disk_worker.remove_put_task(key)

            if cb is not None:
                try:
                    cb(key)
                except Exception as e:
                    logger.warning("on_complete_callback failed for key %s: %s", key, e)

        self._trace_queue_state(
            "flush_append_batches",
            (self._trace_now() - start_time) * 1000.0,
            batch_count=len(batches),
            items=len(flat_keys),
            bytes=total_size,
            avg_bytes_per_item=(total_size // len(flat_keys) if flat_keys else 0),
            path=log_path,
            base_offset=base_offset,
            write_mode="append_pwritev_coalesced",
            oldest_wait_ms=oldest_wait_ms,
            io_lock_wait_ms=io_lock_wait_ms,
            single_io_append=self.single_io_append,
        )

    def _append_flush_loop(self) -> None:
        timeout_s = self.append_batch_timeout_ms / 1000.0
        while True:
            with self.append_queue_cv:
                while not self.append_flush_stop and not self.append_queue:
                    self.append_queue_cv.wait()

                if self.append_flush_stop and not self.append_queue:
                    break

                if self.append_queue:
                    first_ts = self.append_queue[0].enqueue_ts
                    age = time.time() - first_ts
                    enough_bytes = self.append_queue_bytes >= self.append_batch_max_bytes
                    enough_items = sum(len(b.keys) for b in self.append_queue) >= self.append_batch_max_items

                    if not enough_bytes and not enough_items and age < timeout_s:
                        self.append_queue_cv.wait(timeout_s - age)

                batches = self._drain_append_batches_locked()

            if batches:
                self._flush_pending_batches(batches)


    # Hook point for future SSD-CPU service / RPC routing.
    def _write_backend(self, buffer: memoryview | bytes | bytearray, path: str) -> None:
        size = len(buffer)
        if size % self.os_disk_bs != 0 or not self.use_odirect:
            with open(path, "wb") as f:
                f.write(buffer)
        else:
            fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_DIRECT, 0o644)
            try:
                written = os.write(fd, buffer)
                if written != size:
                    raise IOError(
                        f"Short write to {path}: expected {size} bytes, got {written}"
                    )
            finally:
                os.close(fd)

    def _writev_backend(
        self,
        buffers: list[memoryview | bytes | bytearray],
        path: str,
    ) -> bool:
        total_size = sum(len(buf) for buf in buffers)
        if total_size == 0:
            with open(path, "wb"):
                pass
            return False

        if self.use_odirect and all(len(buf) % self.os_disk_bs == 0 for buf in buffers):
            try:
                fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_DIRECT, 0o644)
                try:
                    written = os.writev(fd, buffers)
                    if written != total_size:
                        raise IOError(
                            f"Short writev to {path}: expected {total_size} bytes, got {written}"
                        )
                    return True
                finally:
                    os.close(fd)
            except OSError as e:
                logger.debug(
                    "O_DIRECT writev failed for %s, fallback to buffered writev: %s",
                    path,
                    e,
                )

        fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
        try:
            written = os.writev(fd, buffers)
            if written != total_size:
                raise IOError(
                    f"Short buffered writev to {path}: expected {total_size} bytes, got {written}"
                )
            return False
        finally:
            os.close(fd)

    # Hook point for future SSD-CPU service / RPC routing.
    def _read_backend(
        self,
        buffer: memoryview | bytearray,
        path: str,
        offset: int = 0,
    ) -> None:
        size = len(buffer)
        fblock_aligned = size % self.os_disk_bs == 0 and offset % self.os_disk_bs == 0
        if (offset != 0 or not fblock_aligned) and self.use_odirect:
            logger.debug(
                "Falling back to buffered read for path=%s offset=%d size=%d",
                path,
                offset,
                size,
            )

        if offset != 0 or not fblock_aligned or not self.use_odirect:
            with open(path, "rb") as f:
                f.seek(offset)
                read = f.readinto(buffer)
                if read != len(buffer):
                    raise IOError(
                        f"Short read from {path}: expected {len(buffer)} bytes, got {read}"
                    )
        else:
            fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
            try:
                os.lseek(fd, offset, os.SEEK_SET)
                with os.fdopen(fd, "rb", buffering=0) as fdo:
                    read = fdo.readinto(buffer)
                    if read != len(buffer):
                        raise IOError(
                            f"Short read from {path}: expected {len(buffer)} bytes, got {read}"
                        )
            except Exception:
                os.close(fd)
                raise

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        with self.disk_lock:
            if key not in self.dict:
                return False
            if pin:
                self.dict[key].pin()
                self.keys_in_request.append(key)
            return True

    def touch_cache(self) -> None:
        with self.disk_lock:
            for key in reversed(self.keys_in_request):
                self.cache_policy.update_on_hit(key, self.dict)
            self.keys_in_request = []

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        return self.disk_worker.exists_in_put_tasks(key)

    def pin(self, key: CacheEngineKey) -> bool:
        with self.disk_lock:
            if key in self.dict:
                self.dict[key].pin()
                return True
            return False

    def unpin(self, key: CacheEngineKey) -> bool:
        with self.disk_lock:
            if key in self.dict:
                self.dict[key].unpin()
                return True
            return False

    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:
        if force:
            self.disk_lock.acquire()

        if not (meta := self.dict.pop(key, None)):
            if force:
                self.disk_lock.release()
            return False

        path = meta.path
        size = meta.size
        self.usage -= size
        self.stats_monitor.update_local_storage_usage(self.usage)

        delete_segment = False
        with self.segment_lock:
            seg_entry = self.segment_index.pop(key, None)
            if seg_entry is not None:
                seg_path, _, _ = seg_entry
                if seg_path in self.segment_refcount:
                    self.segment_refcount[seg_path] -= 1
                    if self.segment_refcount[seg_path] <= 0:
                        delete_segment = True
                        del self.segment_refcount[seg_path]
                        path = seg_path
            else:
                delete_segment = True

        if delete_segment and path != self.append_log_path and os.path.exists(path):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass

        if force:
            self.cache_policy.update_on_force_evict(key)
            self.disk_lock.release()

        if self.batched_msg_sender is not None:
            self.batched_msg_sender.add_kv_op(op_type=OpType.EVICT, key=key.chunk_hash)
        return True

    def insert_key(
        self,
        key: CacheEngineKey,
        size: int,
        shape: torch.Size,
        dtype: torch.dtype,
        fmt: MemoryFormat,
        cached_positions: Optional[torch.Tensor] = None,
        path: Optional[str] = None,
        offset: int = 0,
    ) -> None:
        real_path = path or self._key_to_path(key)
        has_stored = False
        with self.disk_lock:
            if key in self.dict:
                self.cache_policy.update_on_hit(key, self.dict)
                has_stored = True
            else:
                self.dict[key] = DiskCacheMetadata(
                    real_path, size, shape, dtype, cached_positions, fmt, 0
                )

        with self.segment_lock:
            self.segment_index[key] = (real_path, offset, size)
            if real_path != self.append_log_path:
                self.segment_refcount[real_path] = self.segment_refcount.get(real_path, 0) + 1

        if self.batched_msg_sender is not None and not has_stored:
            self.batched_msg_sender.add_kv_op(op_type=OpType.ADMIT, key=key.chunk_hash)

    def submit_put_task(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> None | Future:
        start_time = self._trace_now()
        assert memory_obj.tensor is not None

        if self.exists_in_put_tasks(key):
            logger.debug("Put task for %s is already in progress.", key)
            return None

        self.disk_worker.insert_put_task(key)

        required_size = memory_obj.get_physical_size()
        all_evict_keys: list[CacheEngineKey] = []
        evict_success = True
        with self.disk_lock:
            while self.current_cache_size + required_size > self.max_cache_size:
                evict_keys = self.cache_policy.get_evict_candidates(
                    self.dict, num_candidates=1
                )
                if not evict_keys:
                    logger.warning("No eviction candidates found. AI SSD space under pressure.")
                    evict_success = False
                    break

                for evict_key in evict_keys:
                    self.current_cache_size -= self.dict[evict_key].size

                self.batched_remove(evict_keys, force=False)
                all_evict_keys.extend(evict_keys)

            if evict_success:
                self.current_cache_size += required_size
                self.cache_policy.update_on_put(key)

        if not evict_success:
            self._trace_queue_state(
                "submit_put_task",
                (self._trace_now() - start_time) * 1000.0,
                key=getattr(key, "chunk_hash", key),
                scheduled=False,
                required_size=required_size,
                eviction_failed=True,
            )
            return None

        memory_obj.ref_count_up()
        fut = asyncio.run_coroutine_threadsafe(
            self.disk_worker.submit_task(
                "put",
                self.async_save_bytes_to_aissd,
                key=key,
                memory_obj=memory_obj,
                on_complete_callback=on_complete_callback,
            ),
            self.loop,
        )
        self._trace_queue_state(
            "submit_put_task",
            (self._trace_now() - start_time) * 1000.0,
            key=getattr(key, "chunk_hash", key),
            scheduled=True,
            required_size=required_size,
            evicted=len(all_evict_keys),
            current_cache_size=self.current_cache_size,
        )
        return fut

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> None:
        start_time = self._trace_now()
        total_bytes = sum(memory_obj.get_physical_size() for memory_obj in memory_objs)
        num_items = len(memory_objs)

        if num_items <= 1:
            for key, memory_obj in zip(keys, memory_objs, strict=False):
                self.submit_put_task(
                    key, memory_obj, on_complete_callback=on_complete_callback
                )
            self._trace_queue_state(
                "batched_submit_put_task",
                (self._trace_now() - start_time) * 1000.0,
                items=num_items,
                bytes=total_bytes,
                avg_bytes_per_item=(total_bytes // num_items if num_items else 0),
                mode="single_fallback",
            )
            return

        groups = (
            [(list(keys), list(memory_objs))]
            if self.single_io_append
            else self._split_into_segment_groups(keys, memory_objs)
        )

        all_evict_keys: list[CacheEngineKey] = []
        evict_success = True
        with self.disk_lock:
            while self.current_cache_size + total_bytes > self.max_cache_size:
                evict_keys = self.cache_policy.get_evict_candidates(
                    self.dict, num_candidates=1
                )
                if not evict_keys:
                    logger.warning("No eviction candidates found. AI SSD space under pressure.")
                    evict_success = False
                    break

                for evict_key in evict_keys:
                    self.current_cache_size -= self.dict[evict_key].size

                self.batched_remove(evict_keys, force=False)
                all_evict_keys.extend(evict_keys)

            if evict_success:
                self.current_cache_size += total_bytes
                for key in keys:
                    self.cache_policy.update_on_put(key)

        if not evict_success:
            self._trace_queue_state(
                "batched_submit_put_task",
                (self._trace_now() - start_time) * 1000.0,
                items=num_items,
                bytes=total_bytes,
                scheduled=False,
                eviction_failed=True,
            )
            return

        for key in keys:
            self.disk_worker.insert_put_task(key)
        for memory_obj in memory_objs:
            memory_obj.ref_count_up()

        for group_keys, group_memory_objs in groups:
            if self.single_io_append:
                self._enqueue_append_batch(
                    group_keys,
                    group_memory_objs,
                    on_complete_callback,
                )
            else:
                asyncio.run_coroutine_threadsafe(
                    self.disk_worker.submit_task(
                        "put",
                        self.async_save_segment_to_aissd,
                        keys=group_keys,
                        memory_objs=group_memory_objs,
                        on_complete_callback=on_complete_callback,
                    ),
                    self.loop,
                )

        self._trace_queue_state(
            "batched_submit_put_task",
            (self._trace_now() - start_time) * 1000.0,
            items=num_items,
            bytes=total_bytes,
            avg_bytes_per_item=(total_bytes // num_items if num_items else 0),
            evicted=len(all_evict_keys),
            current_cache_size=self.current_cache_size,
            mode=("queued_single_io_append" if self.single_io_append else "parallel_segment"),
            segment_groups=len(groups),
            single_io_append=self.single_io_append,
        )

    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        start_time = self._trace_now()
        self.disk_lock.acquire()
        if key not in self.dict:
            self.disk_lock.release()
            self._trace_backend(
                "get_blocking",
                (self._trace_now() - start_time) * 1000.0,
                hit=False,
                key=getattr(key, "chunk_hash", key),
            )
            return None

        self.cache_policy.update_on_hit(key, self.dict)
        disk_meta = self.dict[key]
        path = disk_meta.path
        offset, read_size = 0, disk_meta.size
        if key in self.segment_index:
            seg_path, offset, read_size = self.segment_index[key]
            path = seg_path
        dtype = disk_meta.dtype
        shape = disk_meta.shape
        fmt = disk_meta.fmt
        assert dtype is not None
        assert shape is not None
        self.disk_lock.release()

        memory_obj = self.load_bytes_from_aissd(key, path, dtype=dtype, shape=shape, fmt=fmt, offset=offset, read_size=read_size)
        self._trace_backend(
            "get_blocking",
            (self._trace_now() - start_time) * 1000.0,
            hit=memory_obj is not None,
            key=getattr(key, "chunk_hash", key),
            bytes=(memory_obj.get_physical_size() if memory_obj is not None else 0),
        )
        return memory_obj

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        transfer_spec: Any = None,
    ) -> list[MemoryObj]:
        start_time = self._trace_now()
        mem_objs: list[MemoryObj] = []
        paths: list[str] = []
        offsets: list[int] = []
        read_sizes: list[int] = []

        logger.debug("lookup_id: %s; Prefetching %d keys from AI SSD.", lookup_id, len(keys))
        for key in keys:
            self.disk_lock.acquire()
            assert key in self.dict, f"Key {key} not found in AI SSD cache after pinning"

            path = self.dict[key].path
            offset, read_size = 0, self.dict[key].size
            if key in self.segment_index:
                seg_path, offset, read_size = self.segment_index[key]
                path = seg_path
            dtype = self.dict[key].dtype
            shape = self.dict[key].shape
            fmt = self.dict[key].fmt
            assert dtype is not None
            assert shape is not None

            memory_obj = self.local_cpu_backend.allocate(
                shape,
                dtype,
                fmt,
                busy_loop=False,
            )
            if memory_obj is None:
                logger.error(
                    "Memory allocation failed during async AI SSD load for key %s. "
                    "CPU staging pool may be exhausted.",
                    key,
                )
                self._trace_queue_state(
                    "batched_get_non_blocking",
                    (self._trace_now() - start_time) * 1000.0,
                    lookup_id=lookup_id,
                    requested=len(keys),
                    prepared=len(mem_objs),
                    allocation_failed=True,
                )
                return mem_objs

            self.dict[key].pin()
            self.cache_policy.update_on_hit(key, self.dict)
            self.disk_lock.release()

            memory_obj.pin()
            mem_objs.append(memory_obj)
            paths.append(path)
            offsets.append(offset)
            read_sizes.append(read_size)

        result = await self.disk_worker.submit_task(
            "prefetch",
            self.batched_async_load_bytes_from_aissd,
            paths=paths,
            keys=keys,
            memory_objs=mem_objs,
            offsets=offsets,
            read_sizes=read_sizes,
        )
        total_bytes = sum(mem_obj.get_physical_size() for mem_obj in result)
        self._trace_queue_state(
            "batched_get_non_blocking",
            (self._trace_now() - start_time) * 1000.0,
            lookup_id=lookup_id,
            requested=len(keys),
            loaded=len(result),
            bytes=total_bytes,
            avg_bytes_per_item=(total_bytes // len(result) if result else 0),
        )
        return result

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        start_time = self._trace_now()
        num_hit_counts = 0
        with self.disk_lock:
            for key in keys:
                if key not in self.dict:
                    self._trace_backend(
                        "batched_async_contains",
                        (self._trace_now() - start_time) * 1000.0,
                        lookup_id=lookup_id,
                        requested=len(keys),
                        hits=num_hit_counts,
                        pin=pin,
                    )
                    return num_hit_counts
                if pin:
                    self.dict[key].pin()
                    self.keys_in_request.append(key)
                num_hit_counts += 1
        self._trace_backend(
            "batched_async_contains",
            (self._trace_now() - start_time) * 1000.0,
            lookup_id=lookup_id,
            requested=len(keys),
            hits=num_hit_counts,
            pin=pin,
        )
        return num_hit_counts

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def async_save_segment_to_aissd(
        self,
        keys: list[CacheEngineKey],
        memory_objs: list[MemoryObj],
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> None:
        start_time = self._trace_now()

        total_size = sum(len(memory_obj.byte_array) for memory_obj in memory_objs)

        rel_offset = 0
        key_records: list[tuple[CacheEngineKey, int, int, MemoryObj]] = []
        chunk_views: list[memoryview | bytes | bytearray] = []

        for key, memory_obj in zip(keys, memory_objs, strict=False):
            chunk = memory_obj.byte_array
            chunk_size = len(chunk)
            chunk_views.append(chunk)
            key_records.append((key, rel_offset, chunk_size, memory_obj))
            rel_offset += chunk_size

        self.usage += total_size
        self.stats_monitor.update_local_storage_usage(self.usage)

        queue_wait_start = self._trace_now()
        with self.append_io_lock:
            queue_wait_ms = (self._trace_now() - queue_wait_start) * 1000.0
            log_path, base_offset = self._reserve_append_region(total_size)
            self.appendv_file(chunk_views, log_path, base_offset)

        for key, chunk_offset, chunk_size, memory_obj in key_records:
            shape = memory_obj.metadata.shape
            dtype = memory_obj.metadata.dtype
            fmt = memory_obj.metadata.fmt
            cached_positions = memory_obj.metadata.cached_positions
            memory_obj.ref_count_down()

            self.insert_key(
                key,
                chunk_size,
                shape,
                dtype,
                fmt,
                cached_positions=cached_positions,
                path=log_path,
                offset=base_offset + chunk_offset,
            )
            self.disk_worker.remove_put_task(key)

            if on_complete_callback is not None:
                try:
                    on_complete_callback(key)
                except Exception as e:
                    logger.warning("on_complete_callback failed for key %s: %s", key, e)

        self._trace_queue_state(
            "async_save_segment_to_aissd",
            (self._trace_now() - start_time) * 1000.0,
            items=len(keys),
            bytes=total_size,
            avg_bytes_per_item=(total_size // len(keys) if keys else 0),
            path=log_path,
            base_offset=base_offset,
            segment_items=len(keys),
            segment_bytes=total_size,
            write_mode="append_pwritev",
            queue_wait_ms=queue_wait_ms,
            single_io_append=self.single_io_append,
        )

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def async_save_bytes_to_aissd(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> None:
        start_time = self._trace_now()
        kv_chunk = memory_obj.tensor
        assert kv_chunk is not None
        buffer = memory_obj.byte_array
        path = self._key_to_path(key)

        size = len(buffer)
        self.usage += size
        self.stats_monitor.update_local_storage_usage(self.usage)

        self.write_file(buffer, path)

        size = memory_obj.get_physical_size()
        shape = memory_obj.metadata.shape
        dtype = memory_obj.metadata.dtype
        fmt = memory_obj.metadata.fmt
        cached_positions = memory_obj.metadata.cached_positions
        memory_obj.ref_count_down()

        self.insert_key(key, size, shape, dtype, fmt, cached_positions=cached_positions)
        self.disk_worker.remove_put_task(key)

        self._trace_queue_state(
            "async_save_bytes_to_aissd",
            (self._trace_now() - start_time) * 1000.0,
            key=getattr(key, "chunk_hash", key),
            bytes=size,
            path=path,
        )

        if on_complete_callback is not None:
            try:
                on_complete_callback(key)
            except Exception as e:
                logger.warning("on_complete_callback failed for key %s: %s", key, e)

    def batched_async_load_bytes_from_aissd(
        self,
        paths: list[str],
        keys: list[CacheEngineKey],
        memory_objs: list[MemoryObj],
        offsets: Optional[list[int]] = None,
        read_sizes: Optional[list[int]] = None,
        write_back: bool = False,
    ) -> list[MemoryObj]:
        start_time = self._trace_now()
        logger.debug("Executing batched async load from AI SSD.")
        loaded_objs: list[MemoryObj] = []
        if offsets is None:
            offsets = [0] * len(paths)
        if read_sizes is None:
            read_sizes = [mem_obj.get_physical_size() for mem_obj in memory_objs]

        for path, key, mem_obj, offset, read_size in zip(paths, keys, memory_objs, offsets, read_sizes, strict=False):
            buffer = mem_obj.byte_array
            ok = self.read_file(key, buffer, path, offset=offset, read_size=read_size)
            if not ok:
                logger.warning("Failed to read key %s from %s during batched AI SSD load", key, path)
                continue

            cached_positions = self.dict[key].cached_positions
            mem_obj.metadata.cached_positions = cached_positions

            self.disk_lock.acquire()
            self.dict[key].unpin()
            self.disk_lock.release()
            loaded_objs.append(mem_obj)

        total_bytes = sum(mem_obj.get_physical_size() for mem_obj in loaded_objs)
        self._trace_queue_state(
            "batched_async_load_bytes_from_aissd",
            (self._trace_now() - start_time) * 1000.0,
            items=len(loaded_objs),
            bytes=total_bytes,
            avg_bytes_per_item=(total_bytes // len(loaded_objs) if loaded_objs else 0),
        )
        return loaded_objs

    def load_bytes_from_aissd(
        self,
        key: CacheEngineKey,
        path: str,
        dtype: torch.dtype,
        shape: torch.Size,
        fmt: MemoryFormat,
        offset: int = 0,
        read_size: Optional[int] = None,
    ) -> Optional[MemoryObj]:
        start_time = self._trace_now()
        memory_obj = self.local_cpu_backend.allocate(shape, dtype, fmt)
        assert memory_obj is not None, "Memory allocation failed during AI SSD load."

        buffer = memory_obj.byte_array
        ok = self.read_file(key, buffer, path, offset=offset, read_size=read_size)
        if not ok:
            memory_obj.ref_count_down()
            return None

        cached_positions = self.dict[key].cached_positions
        memory_obj.metadata.cached_positions = cached_positions

        self._trace_backend(
            "load_bytes_from_aissd",
            (self._trace_now() - start_time) * 1000.0,
            key=getattr(key, "chunk_hash", key),
            bytes=memory_obj.get_physical_size(),
            path=path,
            offset=offset,
            read_size=read_size,
        )
        return memory_obj

    def write_file(self, buffer: memoryview | bytes | bytearray, path: str) -> None:
        start_time = time.time()
        size = len(buffer)
        self._write_backend(buffer, path)
        disk_write_time = time.time() - start_time
        self._trace_backend(
            "write_file",
            disk_write_time * 1000.0,
            bytes=size,
            path=path,
            use_odirect=self.use_odirect,
            aligned=(size % self.os_disk_bs == 0),
            bandwidth_mb_s=(size / disk_write_time / 1e6 if disk_write_time > 0 else None),
        )

    def appendv_file(
        self,
        buffers: list[memoryview | bytes | bytearray],
        path: str,
        offset: int,
    ) -> None:
        start_time = time.time()
        total_size = sum(len(buf) for buf in buffers)
        used_odirect = self._pwritev_backend(buffers, path, offset)
        disk_write_time = time.time() - start_time
        self._trace_backend(
            "appendv_file",
            disk_write_time * 1000.0,
            bytes=total_size,
            items=len(buffers),
            path=path,
            offset=offset,
            use_odirect=used_odirect,
            all_sizes_aligned=all(len(buf) % self.os_disk_bs == 0 for buf in buffers),
            bandwidth_mb_s=(total_size / disk_write_time / 1e6 if disk_write_time > 0 else None),
        )

    def read_file(
        self,
        key: CacheEngineKey,
        buffer: bytearray | memoryview,
        path: str,
        offset: int = 0,
        read_size: Optional[int] = None,
    ) -> bool:
        start_time = time.time()
        size = read_size if read_size is not None else len(buffer)
        fblock_aligned = size % self.os_disk_bs == 0 and offset % self.os_disk_bs == 0
        try:
            self._read_backend(memoryview(buffer)[:size], path, offset=offset)
        except FileNotFoundError:
            logger.warning("File not found on AI SSD cache: %s", path)
            if self.dict.get(key, None):
                self.dict.pop(key)
            return False

        disk_read_time = time.time() - start_time
        self._trace_backend(
            "read_file",
            disk_read_time * 1000.0,
            bytes=size,
            path=path,
            use_odirect=self.use_odirect,
            aligned=fblock_aligned,
            offset=offset,
            bandwidth_mb_s=(size / disk_read_time / 1e6 if disk_read_time > 0 else None),
        )
        return True

    def get_allocator_backend(self) -> LocalCPUBackend:
        return self.local_cpu_backend

    def close(self) -> None:
        start_time = self._trace_now()
        if self.batched_msg_sender is not None:
            self.batched_msg_sender.close()

        with self.append_queue_cv:
            self.append_flush_stop = True
            self.append_queue_cv.notify_all()
        if self.append_flush_thread.is_alive():
            self.append_flush_thread.join(timeout=2.0)

        self.disk_worker.close()
        self._trace_queue_state(
            "close",
            (self._trace_now() - start_time) * 1000.0,
            append_queue_len=len(self.append_queue),
            append_queue_bytes=self.append_queue_bytes,
        )
