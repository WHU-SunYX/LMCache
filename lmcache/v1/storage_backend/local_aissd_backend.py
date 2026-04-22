# SPDX-License-Identifier: Apache-2.0
# Standard
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Sequence
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

        self.max_cache_size = int(
            float(extra.get("aissd.max_cache_size_gb", config.max_local_disk_size))
            * 1024**3
        )
        self.current_cache_size = 0.0

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

    # Hook point for future SSD-CPU service / RPC routing.
    def _write_backend(self, buffer: memoryview | bytes | bytearray, path: str) -> None:
        size = len(buffer)
        if size % self.os_disk_bs != 0 or not self.use_odirect:
            with open(path, "wb") as f:
                f.write(buffer)
        else:
            fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_DIRECT, 0o644)
            try:
                os.write(fd, buffer)
            finally:
                os.close(fd)

    # Hook point for future SSD-CPU service / RPC routing.
    def _read_backend(self, buffer: memoryview | bytearray, path: str) -> None:
        size = len(buffer)
        fblock_aligned = size % self.os_disk_bs == 0
        if not fblock_aligned and self.use_odirect:
            logger.warning(
                "Cannot use O_DIRECT for this file, size is not aligned to disk block size."
            )

        if not fblock_aligned or not self.use_odirect:
            with open(path, "rb") as f:
                f.readinto(buffer)
        else:
            fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
            try:
                with os.fdopen(fd, "rb", buffering=0) as fdo:
                    fdo.readinto(buffer)
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
        os.remove(path)

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
    ) -> None:
        path = self._key_to_path(key)
        has_stored = False
        with self.disk_lock:
            if key in self.dict:
                self.cache_policy.update_on_hit(key, self.dict)
                has_stored = True
            else:
                self.dict[key] = DiskCacheMetadata(
                    path, size, shape, dtype, cached_positions, fmt, 0
                )

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
        total_bytes = 0
        num_items = 0
        for key, memory_obj in zip(keys, memory_objs, strict=False):
            total_bytes += memory_obj.get_physical_size()
            num_items += 1
            self.submit_put_task(
                key, memory_obj, on_complete_callback=on_complete_callback
            )
        self._trace_queue_state(
            "batched_submit_put_task",
            (self._trace_now() - start_time) * 1000.0,
            items=num_items,
            bytes=total_bytes,
            avg_bytes_per_item=(total_bytes // num_items if num_items else 0),
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
        dtype = disk_meta.dtype
        shape = disk_meta.shape
        fmt = disk_meta.fmt
        assert dtype is not None
        assert shape is not None
        self.disk_lock.release()

        memory_obj = self.load_bytes_from_aissd(key, path, dtype=dtype, shape=shape, fmt=fmt)
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

        logger.debug("lookup_id: %s; Prefetching %d keys from AI SSD.", lookup_id, len(keys))
        for key in keys:
            self.disk_lock.acquire()
            assert key in self.dict, f"Key {key} not found in AI SSD cache after pinning"

            path = self.dict[key].path
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

        result = await self.disk_worker.submit_task(
            "prefetch",
            self.batched_async_load_bytes_from_aissd,
            paths=paths,
            keys=keys,
            memory_objs=mem_objs,
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
        write_back: bool = False,
    ) -> list[MemoryObj]:
        start_time = self._trace_now()
        logger.debug("Executing batched async load from AI SSD.")
        loaded_objs: list[MemoryObj] = []
        for path, key, mem_obj in zip(paths, keys, memory_objs, strict=False):
            buffer = mem_obj.byte_array
            ok = self.read_file(key, buffer, path)
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
    ) -> Optional[MemoryObj]:
        start_time = self._trace_now()
        memory_obj = self.local_cpu_backend.allocate(shape, dtype, fmt)
        assert memory_obj is not None, "Memory allocation failed during AI SSD load."

        buffer = memory_obj.byte_array
        ok = self.read_file(key, buffer, path)
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

    def read_file(self, key: CacheEngineKey, buffer: bytearray | memoryview, path: str) -> bool:
        start_time = time.time()
        size = len(buffer)
        fblock_aligned = size % self.os_disk_bs == 0
        try:
            self._read_backend(buffer, path)
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
            bandwidth_mb_s=(size / disk_read_time / 1e6 if disk_read_time > 0 else None),
        )
        return True

    def get_allocator_backend(self) -> LocalCPUBackend:
        return self.local_cpu_backend

    def close(self) -> None:
        start_time = self._trace_now()
        if self.batched_msg_sender is not None:
            self.batched_msg_sender.close()
        self.disk_worker.close()
        self._trace_queue_state("close", (self._trace_now() - start_time) * 1000.0)
