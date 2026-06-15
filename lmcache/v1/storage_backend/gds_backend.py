# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
import asyncio
import ctypes
import json
import mmap
import os
import struct
import threading
import time
import urllib.parse
import uuid

# Third Party
import aiofile
import numpy as np
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, DiskCacheMetadata, _lmcache_nvtx_annotate
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    CuFileMemoryAllocator,
    HipFileMemoryAllocator,
    MemoryFormat,
    MemoryObj,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.abstract_backend import AllocatorBackendInterface
from lmcache.v1.storage_backend.path_sharder import PathSharder

logger = init_logger(__name__)

_METADATA_FILE_SUFFIX = ".metadata"
_DATA_FILE_SUFFIX = ".kvcache.safetensors"
_WEKA_DATA_FILE_SUFFIX = ".weka1"
_METADATA_VERSION = 1
_METADATA_MAX_SIZE = 4096  # reserve 4K for metadata.
# TODO: It is possible to read this 4KB block without triggering read-ahead by
# various means.
_DEFAULT_THREAD_COUNT = 4


class UnsupportedMetadataVersion(Exception):
    pass


torch_dtypes = {
    torch.half: "F16",
    torch.bfloat16: "BF16",
    torch.float32: "F32",
    torch.float64: "F64",
    torch.uint8: "U8",
    torch.uint16: "U16",
    torch.uint32: "U32",
    torch.uint64: "U64",
    torch.int8: "I8",
    torch.int16: "I16",
    torch.int32: "I32",
    torch.int64: "I64",
    torch.float8_e4m3fn: "F8E4M3FN",
    torch.float8_e5m2: "F8E5M2",
}

torch_dtypes_inverse = dict([(v, k) for k, v in torch_dtypes.items()])


def get_fstype(path):
    with open("/proc/mounts", "r") as f:
        lines = f.readlines()

    # Find the best matching mount point
    best_match = ""
    best_fstype = ""
    for line in lines:
        parts = line.split()
        if len(parts) >= 3:
            _, mount_point, fstype = parts[0], parts[1], parts[2]
            if path.startswith(mount_point) and len(mount_point) > len(best_match):
                best_match = mount_point
                best_fstype = fstype

    if not best_fstype:
        raise RuntimeError(f"Unable to detect fstype for {path}")

    return best_fstype


def pack_metadata(tensor, fmt: MemoryFormat, **extra_metadata) -> bytes:
    if tensor.dtype not in torch_dtypes:
        raise RuntimeError(f"unhandled dtype {tensor.dtype}")

    # Metadata
    data_size = tensor.numel() * tensor.element_size()
    tensor_meta = {
        "dtype": torch_dtypes[tensor.dtype],
        "shape": list(tensor.size()),
        "data_offsets": [0, data_size],
        "fmt": fmt.value,
        "__metadata__": extra_metadata,
    }
    meta = {"kvcache": tensor_meta}
    str_meta = json.dumps(meta).encode("utf-8")
    meta_len = len(str_meta)
    assert meta_len <= _METADATA_MAX_SIZE - 8

    # Align to _METADATA_MAX_SIZE - 8
    str_meta += b" " * (_METADATA_MAX_SIZE - 8 - meta_len)

    # Pack it all up so it is sized _METADATA_MAX_SIZE exactly.
    return struct.pack("<Q", len(str_meta)) + str_meta


def unpack_metadata(buffer: bytes):
    meta_len = struct.unpack("<Q", buffer[:8])[0]

    str_meta = buffer[8 : 8 + meta_len]
    json_meta = str_meta.rstrip(b" ")

    meta = json.loads(json_meta.decode("utf-8"))
    tensor_meta = meta["kvcache"]

    shape = tensor_meta["shape"]
    dtype_str = tensor_meta["dtype"]
    data_offsets = tensor_meta["data_offsets"]
    fmt = MemoryFormat(tensor_meta["fmt"])

    nbytes = data_offsets[1] - data_offsets[0]
    dtype = torch_dtypes_inverse[dtype_str]

    return torch.Size(shape), dtype, nbytes, fmt, tensor_meta["__metadata__"]


def rand_suffix(n: int):
    # Generates a random UUID hex string (e.g. "a8098c1a")
    return uuid.uuid4().hex[:n]


async def save_metadata(path: str, tmp: str, metadata: bytes):
    tmp_path = path + tmp
    async with aiofile.async_open(tmp_path, "wb") as f:
        await f.write(metadata)
    os.rename(tmp_path, path)


def get_extra_config_bool(key, config: LMCacheEngineConfig) -> bool | None:
    """Extract a boolean value from the config's extra_config dict.

    Args:
        key: The key to look up in extra_config.
        config: The LMCacheEngineConfig instance.

    Returns:
        The boolean value if present, or None if not set.

    Raises:
        RuntimeError: If the value is not a valid boolean representation.
    """
    value = config.extra_config.get(key, None)
    if value is None:
        return None

    if isinstance(value, str):
        bool_value = value.lower() == "true"
    elif value in [False, True]:
        bool_value = value
    else:
        raise RuntimeError(f"Invalid value `{value}` for `{key}` in extra_config")

    logger.info(f"Getting {key} = {bool_value} from extra_config")
    return bool_value



# ---------------------------------------------------------------------------
# AI-SSD sparse KV RPC bridge.
# ---------------------------------------------------------------------------
_AISSD_BACKEND_CODE = {"host": 0, "ssd-cpu": 1, "ssd-npu": 2}
_AISSD_DTYPE_CODE = {
    "torch.float32": 1,
    "torch.float16": 2,
    "torch.bfloat16": 3,
    "torch.int8": 4,
}
_AISSD_SCORE_CODE = {"topm_mean": 1, "max": 2}
_AISSD_MAX_SELECTED_CHUNKS = 256
_AISSD_MAX_SELECTED_BLOCKS = 4096
_AISSD_PROTOCOL_VERSION = 1
_AISSD_FLAG_Q_INLINE_CMB = 1 << 0
_AISSD_FLAG_RESULT_MANIFEST = 1 << 1


class _AissdSparseKvRunReq(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("cmd", ctypes.c_int32),
        ("version", ctypes.c_uint32),
        ("job_id", ctypes.c_uint64),
        ("request_id", ctypes.c_uint64),
        ("layer_id", ctypes.c_uint32),
        ("backend", ctypes.c_uint32),
        ("num_q_heads", ctypes.c_uint32),
        ("num_kv_heads", ctypes.c_uint32),
        ("head_dim", ctypes.c_uint32),
        ("chunk_size", ctypes.c_uint32),
        ("block_size", ctypes.c_uint32),
        ("top_n_chunks", ctypes.c_uint32),
        ("top_m", ctypes.c_uint32),
        ("score_mode", ctypes.c_uint32),
        ("q_dtype", ctypes.c_uint32),
        ("kv_dtype", ctypes.c_uint32),
        ("q_token_count", ctypes.c_uint32),
        ("candidate_chunk_count", ctypes.c_uint32),
        ("q_cmb_offset", ctypes.c_uint64),
        ("q_bytes", ctypes.c_uint32),
        ("reserved0", ctypes.c_uint32),
        ("k_manifest_lba", ctypes.c_uint64),
        ("k_manifest_bytes", ctypes.c_uint32),
        ("manifest_block_size", ctypes.c_uint32),
        ("result_lba", ctypes.c_uint64),
        ("result_manifest_lba", ctypes.c_uint64),
        ("flags", ctypes.c_uint32),
        ("reserved1", ctypes.c_uint32),
    ]


class _AissdSparseKvRunResp(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("status", ctypes.c_int32),
        ("version", ctypes.c_uint32),
        ("job_id", ctypes.c_uint64),
        ("request_id", ctypes.c_uint64),
        ("layer_id", ctypes.c_uint32),
        ("backend", ctypes.c_uint32),
        ("result_lba", ctypes.c_uint64),
        ("result_manifest_lba", ctypes.c_uint64),
        ("result_manifest_bytes", ctypes.c_uint32),
        ("result_bytes", ctypes.c_uint32),
        ("selected_chunk_count", ctypes.c_uint32),
        ("selected_block_count", ctypes.c_uint32),
        ("block_size", ctypes.c_uint32),
        ("chunk_size", ctypes.c_uint32),
        ("error_code", ctypes.c_uint32),
        ("reserved0", ctypes.c_uint32),
        ("selected_chunk_ids", ctypes.c_uint32 * _AISSD_MAX_SELECTED_CHUNKS),
        ("selected_chunk_scores", ctypes.c_float * _AISSD_MAX_SELECTED_CHUNKS),
        ("selected_block_ids", ctypes.c_uint32 * _AISSD_MAX_SELECTED_BLOCKS),
    ]


class _AissdSparseKvManifestWriteResult(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("status", ctypes.c_int32),
        ("block_size", ctypes.c_uint32),
        ("job_id", ctypes.c_uint64),
        ("k_manifest_lba", ctypes.c_uint64),
        ("k_data_lba", ctypes.c_uint64),
        ("manifest_bytes", ctypes.c_uint64),
        ("k_bytes", ctypes.c_uint64),
        ("k_alloc_bytes", ctypes.c_uint64),
    ]


class _AissdSparseKvResultAllocResult(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("status", ctypes.c_int32),
        ("block_size", ctypes.c_uint32),
        ("job_id", ctypes.c_uint64),
        ("result_lba", ctypes.c_uint64),
        ("result_manifest_lba", ctypes.c_uint64),
        ("result_alloc_bytes", ctypes.c_uint64),
        ("manifest_alloc_bytes", ctypes.c_uint64),
    ]


def _stable_u64(value: Any) -> int:
    data = str(value).encode("utf-8")
    # BLAKE2 is deterministic across Python processes; built-in hash() is not.
    import hashlib

    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "little")



def _tensor_to_c_contiguous_bytes(tensor: torch.Tensor) -> tuple[bytes, int, int, int]:
    if not isinstance(tensor, torch.Tensor):
        raise RuntimeError("q_manifest must carry the live query tensor for AI-SSD sparse KV RPC")
    if tensor.ndim < 2:
        raise RuntimeError(f"query tensor rank must be >=2, got shape={tuple(tensor.shape)}")
    q_token_count = int(tensor.numel() // (int(tensor.shape[-2]) * int(tensor.shape[-1])))
    num_q_heads = int(tensor.shape[-2])
    head_dim = int(tensor.shape[-1])
    cpu = tensor.detach().contiguous().to(device="cpu")
    raw = cpu.view(torch.uint8).numpy().tobytes()
    return raw, q_token_count, num_q_heads, head_dim

class GdsBackend(AllocatorBackendInterface):
    """
    Originally based on the open sourced WekaGdsBackend, this is a backend that
    leverages GPU Direct Storage APIs to issue GDS requests directly to the
    GDS-supported remote filesystem.  In order to use it, users need to specify
    `gds_path` and `gds_buffer_size` in their LMCache config.

    The GDS library to use is controlled by the `gds_backend` config field
    (default: ``"cufile"``). Setting ``use_gds=False`` disables the GDS API
    and falls back to POSIX I/O via cudart.

    Cache Directory Structure created by this Backend:
    /{gds_path}/{first_level}/{second_level}/{data & metadata} This structure
    is semi-arbitrary. We create two levels in the directory hierarchy to
    parallelize loading the data during initialization in the Python code.

    NOTE: If GPUDirect is not supported on that other filesystem, then the GDS
    library will fall back to POSIX I/O.
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        loop: asyncio.AbstractEventLoop,
        dst_device: str = "cuda",
    ):
        assert dst_device.startswith("cuda")
        super().__init__(dst_device=dst_device)

        self.config = config
        self.loop = loop
        self.dst_device = dst_device

        assert config.gds_path is not None, "Need to specify gds_path for GdsBackend"

        sharder = PathSharder(
            raw_csv=config.gds_path,
            strategy=config.gds_path_sharding,
            dst_device=dst_device,
        )
        self.gds_paths = sharder.all_paths
        self.gds_path = sharder.selected
        self.fstype = get_fstype(self.gds_path)

        # Log the fstype - this is useful in reports and varying optimizations
        # based on the kind of fstype used.
        logger.info(
            f"GDS backend using fstype '{self.fstype}' on path '{self.gds_path}'"
            f" ({len(self.gds_paths)} path(s) configured)"
        )

        self.use_gds = config.use_gds
        self.gds_backend = config.gds_backend
        # _user_set_keys is populated by LMCacheEngineConfig when a field is
        # explicitly provided via a config file, environment variable, or
        # keyword argument — as opposed to falling back to its default value.
        # We check it here so that the tmpfs/overlayfs auto-disable logic
        # below can distinguish "the user never mentioned use_gds (default
        # True)" from "the user explicitly wrote use_gds: true".  In the
        # first case we auto-disable GDS on unsupported filesystems; in the
        # second we respect the user's explicit intent and leave it enabled.
        user_set_keys: set[str] = getattr(config, "_user_set_keys", set())
        use_gds_explicitly_set = "use_gds" in user_set_keys

        # Now initialize the memory allocator
        self.memory_allocator = self.initialize_allocator(config, metadata)

        self.data_suffix = _DATA_FILE_SUFFIX
        self._thread_pool = None

        if self.fstype in ["tmpfs", "overlayfs"]:
            # TODO: we can replace the auto-detection of unsupported GDS
            # file systems by doing a small GDS API test on them. If a
            # read/write test fails, we can fallback to not using GDS APIs.
            if use_gds_explicitly_set:
                logger.warning("No automatic disabling of GDS usage due to fstype")
            else:
                logger.info("Automatic disabling of GDS usage due to fstype")
                self.use_gds = False
        elif self.fstype == "wekafs":
            logger.info("Weka filesystem detected, GDS usage is enforced")
            assert self.use_gds
            self.data_suffix = _WEKA_DATA_FILE_SUFFIX

        # Always enable the thread pool for parallel I/O
        self.use_thread_pool = self.use_gds

        if self.use_thread_pool:
            thread_count = _DEFAULT_THREAD_COUNT
            if config.extra_config is not None:
                thread_count = config.extra_config.get(
                    "gds_io_threads", _DEFAULT_THREAD_COUNT
                )
            self._thread_pool = ThreadPoolExecutor(
                max_workers=thread_count, thread_name_prefix="gds-io"
            )

        if self.use_gds:
            logger.info("Using GDS backend '%s'", self.gds_backend)
            if self.gds_backend == "cufile":
                # HACK(Jiayi): cufile import is buggy on some hardware
                # (e.g., without GPUDirect), so it's temporarily put here.
                # Third Party
                import cufile

                self.cudart = None
                self.gds_module = cufile
                self._gds_driver = self.gds_module.CuFileDriver()
            elif self.gds_backend == "hipfile":
                # HACK: hipfile import may be buggy on some hardware
                # (e.g., without GPUDirect), so it's temporarily put here.
                # First Party
                from lmcache.v1.storage_backend import hipfile_shim

                self.cudart = None
                self.gds_module = hipfile_shim
                self._gds_driver = self.gds_module.CuFileDriver()
            else:
                raise ValueError(f"Unsupported gds_backend '{self.gds_backend}'")
        else:
            logger.info("GDS disabled, using POSIX fallback")
            self.gds_module = None
            self.cudart = ctypes.CDLL("libcudart.so")

        self.use_direct_io = False

        # Values for retrying allocations and loads in case of failures potentially
        # due to memory pressure
        self.max_alloc_attempts = (config.extra_config or {}).get(
            "max_alloc_attempts", 10
        )
        self.alloc_attempt_delay_secs = (config.extra_config or {}).get(
            "allocation_attempt_delay_secs", 0.1
        )

        if config.extra_config is not None:
            use_direct_io = get_extra_config_bool("use_direct_io", config)
            if use_direct_io is not None:
                self.use_direct_io = use_direct_io

        for p in self.gds_paths:
            os.makedirs(p, exist_ok=True)

        self.stats = None  # TODO: plug into LMCache Statistics

        self.hot_lock = threading.Lock()
        self.hot_cache: OrderedDict[CacheEngineKey, DiskCacheMetadata] = OrderedDict()
        self.metadata_dirs: set[str] = set()

        self.put_lock = threading.Lock()
        self.put_tasks: set[CacheEngineKey] = set()

        if hasattr(self.memory_allocator, "base_pointer"):
            logger.debug(f"Using base pointer {self.memory_allocator.base_pointer}")
            self.gds_base_pointer = self.memory_allocator.base_pointer
        else:
            logger.info("No base pointer found, GDS will use bounce buffers")
            self.gds_base_pointer = None
        self._scan_metadata_future = asyncio.run_coroutine_threadsafe(
            self._scan_metadata(), self.loop
        )
        self.save_metadata_tasks: set[asyncio.Task] = set()

        # flag for extra assertions to catch bugs but harm performance
        self._debug_asserts = False
        # flag to use O_NOATIME during metadata file read for performance improvement
        self._use_noatime = True

    async def _scan_metadata(self):
        # TODO: even though we only run it once on startup, this is still
        # not super scalable - test whether Rust code will be faster here, or
        # whether we can serialize meta-data in groups for faster loading.
        tasks = []
        start = time.perf_counter()
        for p in self.gds_paths:
            with os.scandir(p) as it:
                for entry in it:
                    if not entry.is_dir():
                        continue
                    l1_dir = os.path.basename(entry.name)
                    if len(l1_dir) != 2:
                        continue
                    tasks.append(
                        asyncio.to_thread(
                            self._scan_metadata_subdir,
                            os.path.join(p, l1_dir),
                            l1_dir,
                        )
                    )
        # TODO: If Python 3.11+, can we use TaskGroup instead?
        await asyncio.gather(*tasks)
        end = time.perf_counter()
        logger.info(
            f"Read {len(self.hot_cache)} cache entries from persistent "
            f"storage in {end - start:.2f} seconds"
        )

    def _scan_metadata_subdir(self, path, l1_dir):
        target_suffix = self.data_suffix + _METADATA_FILE_SUFFIX
        with os.scandir(path) as it:
            for entry in it:
                if not entry.is_dir():
                    continue
                l2_dir = os.path.basename(entry.name)
                if len(l2_dir) != 2:
                    continue
                with os.scandir(os.path.join(path, l2_dir)) as it2:
                    for fentry in it2:
                        if not fentry.is_file():
                            continue
                        if not fentry.name.endswith(target_suffix):
                            continue
                        filename = os.path.basename(fentry.name)
                        key_str = urllib.parse.unquote(filename[: -len(target_suffix)])
                        try:
                            key = CacheEngineKey.from_string(key_str)
                        except ValueError as e:
                            logger.error(
                                f"Filename {filename} can't be converted "
                                f"back into cache key: {e}"
                            )
                            continue
                        try:
                            self._read_metadata(key, fentry.path, l1_dir + l2_dir)
                        except UnsupportedMetadataVersion:
                            logger.error(
                                "Unsupported metadata version for %s; "
                                "ignoring during GDS start",
                                fentry.path,
                            )
                        except Exception:
                            logger.error(
                                "Failed to read metadata file %s during GDS start; "
                                "raising the error to fail startup",
                                fentry.path,
                                exc_info=True,
                            )
                            raise

    def _read_metadata_info(self, filename: str):
        # Use O_NOATIME to prevent updating access time and improve performance
        # Instead of using Python's open() and read(), we use the OS's open() and
        # read() because it is faster - the metadata file is small and we don't
        # need any buffering.
        # Additionally, we use O_NOATIME to improve performance
        if self._use_noatime:
            try:
                fd = os.open(filename, os.O_RDONLY | os.O_NOATIME)
            except (
                # PermissionError: User doesn't own the file
                # AttributeError: O_NOATIME not available on this platform
                # OSError: Filesystem doesn't support O_NOATIME (EINVAL)
                PermissionError,
                AttributeError,
                OSError,
            ):  # fallback to normal open if O_NOATIME is not supported
                self._use_noatime = False
                logger.info(
                    "O_NOATIME flag not supported during metadata file read, "
                    "falling back to normal open"
                )
                fd = os.open(filename, os.O_RDONLY)
        else:
            fd = os.open(filename, os.O_RDONLY)
        try:
            buf = os.read(fd, _METADATA_MAX_SIZE)
        finally:
            os.close(fd)
        return unpack_metadata(buf)

    def _read_metadata(
        self,
        key: CacheEngineKey,
        filename: str,
        subdir_key: str,
    ):
        shape, dtype, size, fmt, extra_metadata = self._read_metadata_info(filename)
        if extra_metadata["lmcache_version"] != str(_METADATA_VERSION):
            raise UnsupportedMetadataVersion("unhandled lmcache metadata")
        logger.debug(
            f"Read metadata for {key} from {filename}: "
            f"shape={shape}, dtype={dtype}, size={size}, fmt={fmt}, "
            f"extra_metadata={extra_metadata}"
        )
        # TODO(extra_metadata)
        # TODO(Jiayi): need to support `cached_positions`.
        # Currently we just fill it as None.
        metadata = DiskCacheMetadata(
            filename.removesuffix(_METADATA_FILE_SUFFIX),
            size,
            shape,
            dtype,
            None,
            fmt,
        )
        with self.hot_lock:
            self.metadata_dirs.add(subdir_key)
            self.hot_cache[key] = metadata
        return metadata

    def __str__(self):
        return self.__class__.__name__

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        # TODO: implement pin() semantics
        with self.hot_lock:
            res = key in self.hot_cache
        if res:
            return True
        if self._try_to_read_metadata(key):
            return True
        return False

    def _try_to_read_metadata(self, key: CacheEngineKey) -> Optional[DiskCacheMetadata]:
        for p in self.gds_paths:
            path, subdir_key, _, _ = self._key_to_path(key, base_path=p)
            path += _METADATA_FILE_SUFFIX
            if os.path.exists(path):
                try:
                    return self._read_metadata(key, path, subdir_key)
                except FileNotFoundError:
                    logger.warning(
                        f"[GDS] File not found for key {key.to_string()} "
                        f"at expected path {path}, returning None"
                    )
                except PermissionError:
                    logger.warning(
                        f"[GDS]: Permission Denied for PID {os.getpid()} on {path},"
                        f" returning None"
                    )
                except UnsupportedMetadataVersion:
                    logger.error(f"Unsupported metadata version for {path}, ignoring")
                except (OSError, IOError) as e:
                    logger.error(
                        f"Failed to read metadata file {path}: {type(e).__name__}: "
                        f"{e}. File may be corrupted or inaccessible. "
                        f"Ignoring cache entry for key {key.to_string()}."
                    )
                except Exception as e:
                    logger.error(
                        f"Unexpected error reading metadata file {path}: "
                        f"{type(e).__name__}: {e}. Ignoring cache entry for key "
                        f"{key.to_string()}."
                    )

        return None

    def _key_to_path(
        self,
        key: CacheEngineKey,
        base_path: Optional[str] = None,
    ) -> Tuple[str, str, str, str]:
        hash = str(key.chunk_hash)
        l1_dir = hash[:2]
        l2_dir = hash[2:4]
        key_str = key.to_string()
        if base_path is None:
            base_path = self.gds_path
        return (
            os.path.join(
                base_path,
                l1_dir,
                l2_dir,
                urllib.parse.quote(key_str, safe="") + self.data_suffix,
            ),
            l1_dir + l2_dir,
            l1_dir,
            l2_dir,
        )

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        with self.put_lock:
            return key in self.put_tasks

    def submit_put_task(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> Future:
        """
        Submit a put task to store KV cache to GDS asynchronously.

        :param on_complete_callback: Optional callback invoked after the GDS
            write completes. Callback exceptions are caught and logged.
        """
        assert memory_obj.tensor is not None
        memory_obj.ref_count_up()

        with self.put_lock:
            self.put_tasks.add(key)

        future = asyncio.run_coroutine_threadsafe(
            self._async_save_bytes_to_disk(key, memory_obj, on_complete_callback),
            self.loop,
        )
        return future

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> Union[List[Future], None]:
        """
        Submit batched put tasks to store KV caches to GDS asynchronously.

        :param on_complete_callback: Optional callback invoked once per key
            after that key's write completes (not once per batch).
        """
        futures = []
        for key, memory_obj in zip(keys, memory_objs, strict=False):
            future = self.submit_put_task(
                key, memory_obj, on_complete_callback=on_complete_callback
            )
            futures.append(future)
        return futures

    async def _async_save_bytes_to_disk(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> None:
        """
        Convert KV to bytes and async store bytes to disk.

        :param on_complete_callback: Optional callback invoked after the GDS
            write completes for this key. Callback exceptions are caught.
        """
        try:
            kv_chunk = memory_obj.tensor
            assert kv_chunk is not None
            path, subdir_key, l1_dir, l2_dir = self._key_to_path(key)
            # TODO: maybe remove `metadata_dirs` and insert mkdir calls
            # only for the case where creating the CuFile fails on ENOENT. It
            # also makes the code more resilient to out-of-band deletions
            if subdir_key not in self.metadata_dirs:
                os.makedirs(os.path.join(self.gds_path, l1_dir, l2_dir), exist_ok=True)
                self.metadata_dirs.add(subdir_key)
            tmp = ".tmp" + rand_suffix(8)
            fmt = memory_obj.metadata.fmt
            try:
                metadata = await asyncio.to_thread(
                    self._save_gds,
                    path,
                    tmp,
                    kv_chunk,
                    fmt,
                    self.gds_base_pointer,
                    memory_obj.metadata.address,
                )
            except Exception as e:
                logger.error(
                    f"GDS write operation failed for key {key.to_string()} at "
                    f"path {path}: tensor_shape={kv_chunk.shape}, "
                    f"tensor_dtype={kv_chunk.dtype}, "
                    f"tensor_size_bytes={kv_chunk.nbytes}, error={e}",
                    exc_info=True,
                )
                return

            # Register key in cache
            logger.debug(
                f"Saved {kv_chunk.numel()} elements of {kv_chunk.dtype} "
                f"to {path} with metadata {metadata}"
            )
            self.insert_key(key, memory_obj)
            try:
                task = asyncio.create_task(
                    save_metadata(path + _METADATA_FILE_SUFFIX, tmp, metadata)
                )
                self.save_metadata_tasks.add(task)
                task.add_done_callback(self.save_metadata_tasks.discard)
                # Add callback to check for exceptions during task execution
                task.add_done_callback(
                    lambda t: self._handle_metadata_write_completion(t, key, path)
                )
            except Exception as e:
                logger.error(
                    f"POSIX metadata write operation failed for key {key.to_string()} "
                    f"at path {path + _METADATA_FILE_SUFFIX}: "
                    f"metadata_size_bytes={len(metadata)}, "
                    f"tmp_suffix={tmp}, error={e}",
                    exc_info=True,
                )
                with self.hot_lock:
                    self.hot_cache.pop(key, None)
                return
        finally:
            memory_obj.ref_count_down()
            with self.put_lock:
                self.put_tasks.discard(key)

        # Call the completion callback if provided
        if on_complete_callback is not None:
            try:
                on_complete_callback(key)
            except Exception as e:
                logger.error(
                    f"on_complete_callback failed for key {key.to_string()}: {e}",
                    exc_info=True,
                )

    def _handle_metadata_write_completion(
        self, task: asyncio.Task, key: CacheEngineKey, path: str
    ) -> None:
        """Handle completion of metadata write task, checking for exceptions."""
        try:
            # Retrieve exception if task failed
            exception = task.exception()
            if exception is not None:
                logger.error(
                    f"Metadata write task failed for key {key.to_string()} "
                    f"at path {path + _METADATA_FILE_SUFFIX}: {exception}",
                    exc_info=exception,
                )
                with self.hot_lock:
                    self.hot_cache.pop(key, None)
        except Exception as e:
            # Exception calling task.exception() (e.g., task was cancelled)
            logger.error(
                f"Error checking metadata write task status for key "
                f"{key.to_string()}: {e}",
                exc_info=True,
            )

    def insert_key(self, key: CacheEngineKey, memory_obj: MemoryObj) -> None:
        path, _, _, _ = self._key_to_path(key)
        size = memory_obj.get_physical_size()
        shape = memory_obj.metadata.shape
        dtype = memory_obj.metadata.dtype
        fmt = memory_obj.metadata.fmt
        with self.hot_lock:
            # TODO(Jiayi): need to support `cached_positions`.
            self.hot_cache[key] = DiskCacheMetadata(path, size, shape, dtype, None, fmt)

    def submit_prefetch_task(
        self,
        key: CacheEngineKey,
    ) -> bool:
        # with self.hot_lock:
        #     entry = self.hot_cache.get(key)
        # if entry is None:
        #     return None

        # path = entry.path
        # dtype = entry.dtype
        # shape = entry.shape
        # fmt = entry.fmt
        # assert dtype is not None
        # assert shape is not None
        # assert fmt is not None
        # return asyncio.run_coroutine_threadsafe(
        #     self._async_load_bytes_from_disk(key, path, dtype, shape，fmt), self.loop
        # )

        # TODO(Jiayi): Need to modify this when prefetch interface is determined.

        # TODO(Jiayi): add `test_gds_backend_sanity` back after implementing this
        return False

    async def _async_load_bytes_from_disk(
        self,
        key: CacheEngineKey,
        path: str,
        dtype: torch.dtype,
        shape: torch.Size,
        fmt: MemoryFormat,
    ) -> Optional[MemoryObj]:
        return self._load_bytes_from_disk_with_allocation(
            key, path, dtype, shape, fmt=fmt
        )

    def get_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[MemoryObj]:
        with self.hot_lock:
            entry = self.hot_cache.get(key)
        if entry is None:
            return None

        path = entry.path
        dtype = entry.dtype
        shape = entry.shape
        fmt = entry.fmt
        logger.warning(entry)
        assert dtype is not None
        assert shape is not None
        assert fmt is not None
        return self._load_bytes_from_disk_with_allocation(
            key, path, dtype=dtype, shape=shape, fmt=fmt
        )

    def _load_bytes_from_disk_with_allocation(
        self,
        key: CacheEngineKey,
        path: str,
        dtype: torch.dtype,
        shape: torch.Size,
        fmt: MemoryFormat,
    ) -> Optional[MemoryObj]:
        """
        Load byte array from disk by first allocating memory, then loading.

        Args:
            key: Cache key for error handling
            path: File path to load from
            dtype: Data type for memory allocation
            shape: Shape for memory allocation

        Returns:
            A new memory object with loaded data, or None if allocation or
            loading failed
        """
        memory_obj = self.memory_allocator.allocate(shape, dtype, fmt=fmt)
        if memory_obj is None:
            logger.error("Memory allocation failed during sync disk load.")
            return None
        if self._debug_asserts:
            assert memory_obj.tensor is not None
            assert memory_obj.tensor.is_cuda
            assert torch.device(self.dst_device) == torch.device(
                memory_obj.tensor.device
            )

        return self._load_bytes_from_disk_with_memory(key, path, memory_obj)

    def _load_bytes_from_disk_with_memory(
        self,
        key: CacheEngineKey,
        path: str,
        memory_obj: Optional[MemoryObj],
    ) -> Optional[MemoryObj]:
        """
        Load byte array from disk into a pre-allocated memory object.

        Args:
            key: Cache key for error handling
            path: File path to load from
            memory_obj: Pre-allocated memory object to load data into

        Returns:
            The memory object with loaded data, or None if loading failed
        """
        if memory_obj is None or not memory_obj.is_valid():
            return None

        offset = _METADATA_MAX_SIZE
        if self.gds_base_pointer is None:
            tensor = memory_obj.tensor
            assert tensor is not None
            if self._debug_asserts:
                assert tensor.is_cuda
                assert torch.device(self.dst_device) == torch.device(tensor.device)
            addr = ctypes.c_void_p(tensor.data_ptr())
            dev_offset = 0
        else:
            addr = ctypes.c_void_p(self.gds_base_pointer)
            dev_offset = memory_obj.metadata.address
        ret = self._load_gds(path, offset, addr, memory_obj.get_size(), dev_offset)
        if ret != memory_obj.get_size():
            if ret < 0:
                logger.error(
                    f"Error loading {path}: ret: {ret} removing entry from cache"
                )
                with self.hot_lock:
                    self.hot_cache.pop(key)
            else:
                # TODO: we should probably count errors and
                # remove the entry if it's a persistent problem.
                logger.error(
                    f"Error loading {path}: got only {ret} bytes "
                    f"out of {memory_obj.get_size()}, ignoring"
                )
            memory_obj.ref_count_down()
            return None
        return memory_obj

    def get_sparse_kv_candidate_manifest(
        self,
        blocks: List[Tuple[CacheEngineKey, int, int]],
        req_id: Optional[str] = None,
        layer_name: Optional[str] = None,
        slot_mapping: Any = None,
        chunk_size: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Return GDS file/offset candidates for chunk-level sparse KV.

        This does not read data. It exposes enough host-side metadata for the
        next SSD-CPU/NPU selector RPC. Data starts after the fixed 4KB metadata
        header used by the GDS backend.
        """
        manifest: list[dict[str, Any]] = []
        slot_mapping_cpu = None
        if slot_mapping is not None:
            try:
                slot_mapping_cpu = slot_mapping.to(device="cpu")
            except Exception:
                slot_mapping_cpu = None

        with self.hot_lock:
            for chunk_index, (key, start, end) in enumerate(blocks):
                entry = self.hot_cache.get(key)
                if entry is None:
                    self._try_to_read_metadata(key)
                    entry = self.hot_cache.get(key)
                if entry is None:
                    continue
                slot_start = None
                slot_end = None
                if slot_mapping_cpu is not None and end <= slot_mapping_cpu.numel():
                    try:
                        slot_start = int(slot_mapping_cpu[start].item())
                        slot_end = int(slot_mapping_cpu[end - 1].item()) + 1
                    except Exception:
                        slot_start = None
                        slot_end = None
                manifest.append(
                    {
                        "backend": self.__class__.__name__,
                        "req_id": req_id,
                        "layer_name": layer_name,
                        "chunk_index": chunk_index,
                        "key": key.to_string(),
                        "chunk_hash": str(key.chunk_hash),
                        "token_start": int(start),
                        "token_end": int(end),
                        "num_tokens": int(end - start),
                        "slot_start": slot_start,
                        "slot_end": slot_end,
                        "path": entry.path,
                        "file_offset": _METADATA_MAX_SIZE,
                        "nbytes": int(entry.size),
                        "shape": list(entry.shape) if entry.shape is not None else None,
                        "dtype": str(entry.dtype),
                        "fmt": str(entry.fmt),
                        "chunk_size": chunk_size,
                        "layout_version": 1,
                    }
                )
        return manifest

    def _aissd_sparse_extra(self, key: str, default: Any = None) -> Any:
        extra = getattr(self.config, "extra_config", {}) or {}
        return extra.get(key, extra.get(f"lmcache.{key}", default))

    def _aissd_sparse_lib_path(self) -> str:
        # Library path is configuration, not runtime data/LBA state.
        value = self._aissd_sparse_extra("aissd_sparse_kv_lib", None)
        if value:
            return str(value)
        return os.environ.get("AISSD_SPARSE_KV_LIB", "libaissd_sparse_kv_client.so")

    def _aissd_sparse_gds_lib_path(self) -> str:
        value = self._aissd_sparse_extra("aissd_sparse_kv_gds_lib", None)
        if value:
            return str(value)
        return os.environ.get("AISSD_SPARSE_KV_GDS_LIB", "libaissd_sparse_kv_gds.so")

    def _aissd_sparse_gds_dir(self) -> str:
        value = self._aissd_sparse_extra("aissd_sparse_kv_gds_dir", None)
        return str(value) if value else self.gds_path

    def _load_sparse_candidate_tensor(self, chunk: dict[str, Any]) -> torch.Tensor:
        path = chunk.get("path")
        shape = chunk.get("shape")
        dtype_value = chunk.get("dtype")
        if path is None or shape is None or dtype_value is None:
            raise RuntimeError(f"[aissd-sparse-kv] incomplete candidate chunk manifest: {chunk}")
        dtype = self._parse_sparse_manifest_dtype(dtype_value)
        tensor = torch.empty(torch.Size(shape), dtype=dtype, device=self.dst_device)
        file_offset = int(chunk.get("file_offset", _METADATA_MAX_SIZE))
        nbytes = int(chunk.get("nbytes", tensor.numel() * tensor.element_size()))
        ret = self._load_gds(path, file_offset, ctypes.c_void_p(tensor.data_ptr()), nbytes, 0)
        if ret != nbytes:
            raise RuntimeError(
                f"[aissd-sparse-kv] failed to load candidate K chunk path={path} ret={ret} expected={nbytes}"
            )
        return tensor

    def _extract_k_candidate_for_aissd(
        self,
        tensor: torch.Tensor,
        chunk: dict[str, Any],
        layer_id: int,
        chunk_size: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> torch.Tensor:
        fmt_value = chunk.get("fmt")
        fmt = self._parse_sparse_manifest_fmt(fmt_value) if fmt_value is not None else None
        hidden = num_kv_heads * head_dim

        if fmt == MemoryFormat.KV_T2D:
            # [2, T, D]
            if tensor.ndim != 3 or int(tensor.shape[0]) != 2:
                raise RuntimeError(f"[aissd-sparse-kv] KV_T2D expects [2,T,D], got {tuple(tensor.shape)}")
            k = tensor[0]
        elif fmt == MemoryFormat.KV_2LTD:
            # [2, L, T, D].  This is the non-layerwise LMCache format.
            if tensor.ndim != 4 or int(tensor.shape[0]) != 2:
                raise RuntimeError(f"[aissd-sparse-kv] KV_2LTD expects [2,L,T,D], got {tuple(tensor.shape)}")
            if layer_id < 0 or layer_id >= int(tensor.shape[1]):
                raise RuntimeError(
                    f"[aissd-sparse-kv] layer_id={layer_id} is outside KV_2LTD layer dim={tensor.shape[1]}"
                )
            k = tensor[0, layer_id]
        else:
            # Target path supports explicit selector layout in case LMCache writes it directly.
            if tensor.ndim == 3 and int(tensor.shape[1]) == num_kv_heads and int(tensor.shape[2]) == head_dim:
                k = tensor
            elif tensor.ndim == 2 and int(tensor.shape[1]) == hidden:
                k = tensor
            else:
                raise RuntimeError(
                    f"[aissd-sparse-kv] unsupported candidate fmt={fmt_value!r} shape={tuple(tensor.shape)}; "
                    "expected KV_T2D, KV_2LTD, or explicit [T,H,D]/[T,D] K-only layout"
                )

        if k.ndim == 2:
            if int(k.shape[1]) != hidden:
                raise RuntimeError(f"[aissd-sparse-kv] K hidden dim mismatch: got {tuple(k.shape)}, expected D={hidden}")
            k = k.reshape(int(k.shape[0]), num_kv_heads, head_dim)
        elif k.ndim == 3:
            if int(k.shape[1]) != num_kv_heads or int(k.shape[2]) != head_dim:
                raise RuntimeError(
                    f"[aissd-sparse-kv] K head layout mismatch: got {tuple(k.shape)}, "
                    f"expected [T,{num_kv_heads},{head_dim}]"
                )
        else:
            raise RuntimeError(f"[aissd-sparse-kv] extracted K has unsupported rank: {tuple(k.shape)}")

        if int(k.shape[0]) < chunk_size:
            raise RuntimeError(f"[aissd-sparse-kv] K tokens={k.shape[0]} < chunk_size={chunk_size}")
        return k[:chunk_size].contiguous()

    def _materialize_aissd_k_manifest(
        self,
        chunks: list[dict[str, Any]],
        job_id: int,
        layer_id: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        chunk_size: int,
        block_size: int,
        kv_dtype: int,
    ) -> tuple[int, int]:
        k_chunks: list[torch.Tensor] = []
        for chunk in chunks:
            tensor = self._load_sparse_candidate_tensor(chunk)
            k_chunks.append(
                self._extract_k_candidate_for_aissd(
                    tensor=tensor,
                    chunk=chunk,
                    layer_id=layer_id,
                    chunk_size=chunk_size,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                )
            )
        if not k_chunks:
            raise RuntimeError("[aissd-sparse-kv] no candidate K chunks materialized")
        k_tensor = torch.stack(k_chunks, dim=0).contiguous()
        lib = ctypes.CDLL(self._aissd_sparse_gds_lib_path())
        lib.aissd_gds_write_sparse_kv_manifest.argtypes = [
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_int,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint64,
            ctypes.POINTER(_AissdSparseKvManifestWriteResult),
        ]
        lib.aissd_gds_write_sparse_kv_manifest.restype = ctypes.c_int
        out = _AissdSparseKvManifestWriteResult()
        ret = lib.aissd_gds_write_sparse_kv_manifest(
            self._aissd_sparse_gds_dir().encode("utf-8"),
            ctypes.c_void_p(k_tensor.data_ptr()),
            ctypes.c_uint64(k_tensor.numel() * k_tensor.element_size()),
            ctypes.c_int(1),
            ctypes.c_uint32(kv_dtype),
            ctypes.c_uint32(num_q_heads),
            ctypes.c_uint32(num_kv_heads),
            ctypes.c_uint32(head_dim),
            ctypes.c_uint32(chunk_size),
            ctypes.c_uint32(len(chunks)),
            ctypes.c_uint32(block_size),
            ctypes.c_uint64(job_id),
            ctypes.byref(out),
        )
        if ret != 0 or out.status != 0:
            raise RuntimeError(f"[aissd-sparse-kv] write K manifest failed ret={ret} status={out.status}")
        return int(out.k_manifest_lba), int(out.manifest_bytes)

    def _allocate_aissd_result_slots(
        self,
        job_id: int,
        top_n_chunks: int,
        chunk_size: int,
        block_size: int,
    ) -> tuple[int, int]:
        blocks_per_chunk = (chunk_size + block_size - 1) // block_size
        selected_block_count = top_n_chunks * blocks_per_chunk
        # sizeof(AissdSparseKvSelectedHeader)=56 with current packed protocol.
        result_bytes = 56 + top_n_chunks * 4 + top_n_chunks * 4 + selected_block_count * 4
        manifest_bytes = 4096
        lib = ctypes.CDLL(self._aissd_sparse_gds_lib_path())
        if not hasattr(lib, "aissd_gds_alloc_sparse_kv_result"):
            raise RuntimeError("[aissd-sparse-kv] libaissd_sparse_kv_gds.so lacks aissd_gds_alloc_sparse_kv_result")
        lib.aissd_gds_alloc_sparse_kv_result.argtypes = [
            ctypes.c_char_p,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.c_uint64,
            ctypes.POINTER(_AissdSparseKvResultAllocResult),
        ]
        lib.aissd_gds_alloc_sparse_kv_result.restype = ctypes.c_int
        out = _AissdSparseKvResultAllocResult()
        ret = lib.aissd_gds_alloc_sparse_kv_result(
            self._aissd_sparse_gds_dir().encode("utf-8"),
            ctypes.c_uint64(result_bytes),
            ctypes.c_uint64(manifest_bytes),
            ctypes.c_uint32(block_size),
            ctypes.c_uint64(job_id),
            ctypes.byref(out),
        )
        if ret != 0 or out.status != 0:
            raise RuntimeError(f"[aissd-sparse-kv] allocate result slots failed ret={ret} status={out.status}")
        return int(out.result_lba), int(out.result_manifest_lba)

    def select_sparse_kv_chunks(
        self,
        q_manifest: dict[str, Any],
        candidate_manifest: dict[str, Any],
        top_n_chunks: int,
        score_mode: str = "topm_mean",
        sparse_kv_backend: str = "host",
        req_id: Optional[str] = None,
        layer_name: Optional[str] = None,
    ) -> dict[str, Any]:
        """Select top-n chunks.

        backend=host keeps the original host-side deterministic selector.
        backend=ssd-cpu/ssd-npu must use the AI-SSD RPC path; any missing
        manifest/RPC/input requirement is a hard error, not a fallback.
        """
        chunks = list(candidate_manifest.get("chunks", []))
        backend_name = (sparse_kv_backend or "host").strip().lower()
        if backend_name not in _AISSD_BACKEND_CODE:
            raise ValueError(f"Unsupported sparse_kv_backend={backend_name!r}")

        n = top_n_chunks if top_n_chunks and top_n_chunks > 0 else len(chunks)
        if backend_name == "host":
            selected = chunks[:n]
            for rank, chunk in enumerate(selected):
                chunk.setdefault("score", None)
                chunk["selected_rank"] = rank
            return {
                "req_id": req_id,
                "layer_name": layer_name,
                "granularity": "chunk",
                "score_mode": score_mode,
                "top_n_chunks": top_n_chunks,
                "selector": "host_first_n",
                "sparse_kv_backend": "host",
                "q_manifest": q_manifest,
                "selected_chunks": selected,
            }

        if not chunks:
            raise RuntimeError(f"[aissd-sparse-kv] backend={backend_name} got empty candidate manifest")

        q_tensor = q_manifest.get("tensor")
        q_bytes, q_token_count, num_q_heads, head_dim = _tensor_to_c_contiguous_bytes(q_tensor)
        q_dtype = _AISSD_DTYPE_CODE.get(str(getattr(q_tensor, "dtype", "")))
        if q_dtype is None:
            raise RuntimeError(f"[aissd-sparse-kv] unsupported q dtype={getattr(q_tensor, 'dtype', None)}")

        chunk_size = int(candidate_manifest.get("chunk_size") or chunks[0].get("chunk_size") or 0)
        if chunk_size <= 0:
            raise RuntimeError("[aissd-sparse-kv] chunk_size is required")
        block_size = int(candidate_manifest.get("block_size") or os.environ.get("AISSD_SPARSE_KV_BLOCK_SIZE", "16"))
        num_kv_heads = int(candidate_manifest.get("num_kv_heads") or os.environ.get("AISSD_SPARSE_KV_NUM_KV_HEADS", "0"))
        if num_kv_heads <= 0:
            raise RuntimeError("AISSD_SPARSE_KV_NUM_KV_HEADS or candidate_manifest['num_kv_heads'] is required")
        kv_dtype = int(candidate_manifest.get("kv_dtype_code") or os.environ.get("AISSD_SPARSE_KV_KV_DTYPE", "3"))
        top_m = int(candidate_manifest.get("top_m") or os.environ.get("AISSD_SPARSE_KV_TOP_M", "8"))
        score_code = _AISSD_SCORE_CODE.get(score_mode, _AISSD_SCORE_CODE["topm_mean"])

        manifest_block_size = int(candidate_manifest.get("aissd_manifest_block_size") or self._aissd_sparse_extra("aissd_sparse_kv_manifest_block_size", 4096))
        job_id = _stable_u64(f"{req_id}:{layer_name}:{time.time_ns()}")
        req_u64 = _stable_u64(req_id or "")
        layer_u32 = int(candidate_manifest.get("layer_id") or q_manifest.get("layer_id") or 0)

        # Materialize the selector-specific K-only tensor and allocate result slots
        # in the GDS directory automatically.  No runtime LBA is accepted from env.
        k_manifest_lba, k_manifest_bytes = self._materialize_aissd_k_manifest(
            chunks=chunks,
            job_id=job_id,
            layer_id=layer_u32,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            chunk_size=chunk_size,
            block_size=manifest_block_size,
            kv_dtype=kv_dtype,
        )
        result_lba, result_manifest_lba = self._allocate_aissd_result_slots(
            job_id=job_id,
            top_n_chunks=int(top_n_chunks),
            chunk_size=chunk_size,
            block_size=block_size,
        )

        lib_path = self._aissd_sparse_lib_path()
        lib = ctypes.CDLL(lib_path)
        lib.aissd_sparse_kv_rpc_init.argtypes = []
        lib.aissd_sparse_kv_rpc_init.restype = ctypes.c_int
        lib.aissd_sparse_kv_run_manifest_lba.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(_AissdSparseKvRunReq),
            ctypes.POINTER(_AissdSparseKvRunResp),
            ctypes.c_int,
        ]
        lib.aissd_sparse_kv_run_manifest_lba.restype = ctypes.c_int
        init_ret = lib.aissd_sparse_kv_rpc_init()
        if init_ret != 0:
            raise RuntimeError(f"[aissd-sparse-kv] rpc init failed ret={init_ret}")

        req = _AissdSparseKvRunReq()
        req.cmd = 100
        req.version = _AISSD_PROTOCOL_VERSION
        req.job_id = job_id
        req.request_id = req_u64
        req.layer_id = layer_u32
        req.backend = _AISSD_BACKEND_CODE[backend_name]
        req.num_q_heads = num_q_heads
        req.num_kv_heads = num_kv_heads
        req.head_dim = head_dim
        req.chunk_size = chunk_size
        req.block_size = block_size
        req.top_n_chunks = int(top_n_chunks)
        req.top_m = top_m
        req.score_mode = score_code
        req.q_dtype = q_dtype
        req.kv_dtype = kv_dtype
        req.q_token_count = q_token_count
        req.candidate_chunk_count = len(chunks)
        req.k_manifest_lba = int(k_manifest_lba)
        req.k_manifest_bytes = int(k_manifest_bytes)
        req.manifest_block_size = manifest_block_size
        req.result_lba = result_lba
        req.result_manifest_lba = result_manifest_lba
        req.flags = _AISSD_FLAG_Q_INLINE_CMB | _AISSD_FLAG_RESULT_MANIFEST

        q_buf = ctypes.create_string_buffer(q_bytes)
        resp = _AissdSparseKvRunResp()
        timeout_ms = int(os.environ.get("AISSD_SPARSE_KV_TIMEOUT_MS", "300000"))
        ret = lib.aissd_sparse_kv_run_manifest_lba(
            ctypes.cast(q_buf, ctypes.c_void_p),
            ctypes.c_uint32(len(q_bytes)),
            ctypes.byref(req),
            ctypes.byref(resp),
            ctypes.c_int(timeout_ms),
        )
        if ret != 0 or resp.status != 0:
            raise RuntimeError(
                f"[aissd-sparse-kv] RPC failed ret={ret} status={resp.status} "
                f"backend={backend_name} req_id={req_id} layer={layer_name}"
            )

        selected_count = int(resp.selected_chunk_count)
        if selected_count <= 0:
            raise RuntimeError(f"[aissd-sparse-kv] backend={backend_name} returned zero selected chunks")
        if selected_count > len(chunks):
            raise RuntimeError(
                f"[aissd-sparse-kv] selected_count={selected_count} exceeds candidate chunks={len(chunks)}"
            )

        selected: list[dict[str, Any]] = []
        for rank in range(selected_count):
            idx = int(resp.selected_chunk_ids[rank])
            if idx < 0 or idx >= len(chunks):
                raise RuntimeError(f"[aissd-sparse-kv] invalid selected chunk index={idx}")
            chunk = dict(chunks[idx])
            chunk["score"] = float(resp.selected_chunk_scores[rank])
            chunk["selected_rank"] = rank
            chunk["aissd_result_lba"] = int(resp.result_lba)
            chunk["aissd_result_manifest_lba"] = int(resp.result_manifest_lba)
            chunk["aissd_selected_block_ids"] = [
                int(resp.selected_block_ids[i])
                for i in range(min(int(resp.selected_block_count), _AISSD_MAX_SELECTED_BLOCKS))
            ]
            selected.append(chunk)

        logger.info(
            "[aissd-sparse-kv] backend=%s req_id=%s layer=%s candidates=%d selected=%d result_manifest_lba=%d",
            backend_name,
            req_id,
            layer_name,
            len(chunks),
            len(selected),
            int(resp.result_manifest_lba),
        )
        return {
            "req_id": req_id,
            "layer_name": layer_name,
            "granularity": "chunk",
            "score_mode": score_mode,
            "top_n_chunks": top_n_chunks,
            "selector": f"aissd_{backend_name}",
            "sparse_kv_backend": backend_name,
            "q_manifest": q_manifest,
            "selected_chunks": selected,
            "aissd_result_lba": int(resp.result_lba),
            "aissd_result_manifest_lba": int(resp.result_manifest_lba),
            "aissd_result_bytes": int(resp.result_bytes),
        }


    def _parse_sparse_manifest_dtype(self, dtype_value: Any) -> torch.dtype:
        """Parse dtype strings emitted by get_sparse_kv_candidate_manifest()."""
        if isinstance(dtype_value, torch.dtype):
            return dtype_value
        dtype_str = str(dtype_value)
        if dtype_str.startswith("torch."):
            name = dtype_str.split(".", 1)[1]
            if hasattr(torch, name):
                return getattr(torch, name)
        if dtype_str in torch_dtypes_inverse:
            return torch_dtypes_inverse[dtype_str]
        raise ValueError(f"Unsupported sparse manifest dtype={dtype_value!r}")

    def _parse_sparse_manifest_fmt(self, fmt_value: Any) -> MemoryFormat:
        """Parse MemoryFormat strings emitted by get_sparse_kv_candidate_manifest()."""
        if isinstance(fmt_value, MemoryFormat):
            return fmt_value
        fmt_str = str(fmt_value)
        if fmt_str.startswith("MemoryFormat."):
            name = fmt_str.split(".", 1)[1]
            if hasattr(MemoryFormat, name):
                return getattr(MemoryFormat, name)
        return MemoryFormat(fmt_str)

    def _load_sparse_manifest_chunk_to_memory(
        self,
        path: str,
        file_offset: int,
        nbytes: int,
        memory_obj: Optional[MemoryObj],
    ) -> Optional[MemoryObj]:
        """Read one selected chunk described by a sparse manifest into GPU memory."""
        if memory_obj is None or not memory_obj.is_valid():
            return None

        if self.gds_base_pointer is None:
            tensor = memory_obj.tensor
            assert tensor is not None
            if self._debug_asserts:
                assert tensor.is_cuda
                assert torch.device(self.dst_device) == torch.device(tensor.device)
            addr = ctypes.c_void_p(tensor.data_ptr())
            dev_offset = 0
        else:
            addr = ctypes.c_void_p(self.gds_base_pointer)
            dev_offset = memory_obj.metadata.address

        ret = self._load_gds(path, file_offset, addr, nbytes, dev_offset)
        if ret != nbytes:
            logger.error(
                "[sparse-kv-load] GDS read failed path=%s ret=%s expected=%s offset=%s",
                path,
                ret,
                nbytes,
                file_offset,
            )
            memory_obj.ref_count_down()
            return None
        return memory_obj

    def load_sparse_kv_selected_chunks(
        self,
        selected_chunks: list[dict[str, Any]],
    ) -> list[tuple[MemoryObj, int, int, dict[str, Any]]]:
        """Load selected chunk-level KV entries from GDS.

        Route 1 writes selected chunks back to vLLM's paged KV cache through the
        normal GPUConnector. Therefore each selected chunk is still loaded as a
        full LMCache chunk MemoryObj; sparse attention later decides which
        loaded chunks are visible.
        """
        loaded: list[tuple[MemoryObj, int, int, dict[str, Any]]] = []
        total_bytes = 0
        start_time = time.perf_counter()

        for chunk in selected_chunks:
            path = chunk.get("path")
            shape = chunk.get("shape")
            dtype_value = chunk.get("dtype")
            fmt_value = chunk.get("fmt")
            if path is None or shape is None or dtype_value is None or fmt_value is None:
                logger.warning("[sparse-kv-load] skip incomplete selected chunk: %s", chunk)
                continue

            try:
                dtype = self._parse_sparse_manifest_dtype(dtype_value)
                fmt = self._parse_sparse_manifest_fmt(fmt_value)
                torch_shape = torch.Size(shape)
                file_offset = int(chunk.get("file_offset", _METADATA_MAX_SIZE))
                nbytes = int(chunk.get("nbytes", 0))
                token_start = int(chunk["token_start"])
                token_end = int(chunk["token_end"])
            except Exception:
                logger.exception("[sparse-kv-load] invalid selected chunk manifest: %s", chunk)
                continue

            memory_obj = self.memory_allocator.allocate(torch_shape, dtype, fmt=fmt)
            if memory_obj is None:
                logger.warning(
                    "[sparse-kv-load] allocation failed path=%s shape=%s dtype=%s fmt=%s",
                    path,
                    list(torch_shape),
                    dtype,
                    fmt,
                )
                continue

            memory_obj = self._load_sparse_manifest_chunk_to_memory(
                path=path,
                file_offset=file_offset,
                nbytes=nbytes,
                memory_obj=memory_obj,
            )
            if memory_obj is None:
                continue

            total_bytes += memory_obj.get_size()
            loaded.append((memory_obj, token_start, token_end, chunk))

        duration_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info(
            "[sparse-kv-load] backend=GdsBackend selected_chunks=%d loaded_chunks=%d bytes=%d duration_ms=%.3f",
            len(selected_chunks),
            len(loaded),
            total_bytes,
            duration_ms,
        )
        return loaded

    def get_non_blocking(
        self,
        key: CacheEngineKey,
        location: Optional[str] = None,
    ) -> Optional[Future]:
        # TODO: Using a dummy wrapper around prefetch for now.
        if not self.submit_prefetch_task(key):
            return None
        return Future()

    def batched_get_blocking(
        self,
        keys: List[CacheEngineKey],
    ) -> List[Optional[MemoryObj]]:
        if self.use_thread_pool:
            logger.debug("Using batched_get_blocking with thread pool implementation")
            return self._batched_get_blocking_by_thread_pool_impl(keys)
        else:
            return super().batched_get_blocking(keys)

    def _batched_get_blocking_by_thread_pool_impl(
        self,
        keys: List[CacheEngineKey],
    ) -> list[MemoryObj | None]:
        paths: list[str | None] = []
        dtypes: list[torch.dtype | None] = []
        shapes: list[torch.Size | None] = []
        fmts: list[MemoryFormat | None] = []
        with self.hot_lock:
            for key in keys:
                entry = self.hot_cache.get(key)
                if entry is None:
                    logger.error(f"Lookup failed during get_blocking for {key}")
                    paths.append(None)
                    dtypes.append(None)
                    shapes.append(None)
                    fmts.append(None)
                    continue
                paths.append(entry.path)
                dtypes.append(entry.dtype)
                shapes.append(entry.shape)
                fmts.append(entry.fmt)

        memory_objs: list[MemoryObj | None] = []
        gds_reads, gds_read_bytes = 0, 0
        for dtype, shape, path, fmt in zip(dtypes, shapes, paths, fmts, strict=True):
            if path is None:
                memory_objs.append(None)
                continue
            memory_obj = self.memory_allocator.allocate(shape, dtype, fmt=fmt)
            if memory_obj is None:
                logger.error(f"Memory allocation failed during get_blocking for {path}")
            else:
                gds_reads += 1
                gds_read_bytes += memory_obj.get_size()
            memory_objs.append(memory_obj)

        start_time = time.perf_counter()
        assert self._thread_pool is not None
        results = list(
            self._thread_pool.map(
                self._load_bytes_from_disk_with_memory, keys, paths, memory_objs
            )
        )
        total_time = time.perf_counter() - start_time
        logger.info(
            f"Time taken for batched_get_blocking: {total_time:.3f}s |"
            f" {gds_read_bytes / 1024 / 1024}MiB | {gds_reads} ops."
        )
        return results

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def _save_gds(
        self,
        path: str,
        tmp: str,
        kv_chunk: torch.Tensor,
        fmt: MemoryFormat,
        base_pointer: int,
        device_offset: int,
    ):
        if base_pointer is None:
            addr = ctypes.c_void_p(kv_chunk.data_ptr())
            dev_offset = 0
        else:
            addr = ctypes.c_void_p(base_pointer)
            dev_offset = device_offset
        tmp_path = path + tmp
        offset = _METADATA_MAX_SIZE
        # TODO: We can add the chunk's metadata here, e.g. Tensor parallelism shard
        # and pipeline parallelism index.
        metadata = pack_metadata(
            kv_chunk, fmt=fmt, lmcache_version=str(_METADATA_VERSION)
        )
        try:
            with open(tmp_path, "wb") as f:
                f.write(metadata)
            if self.gds_module:
                with self.gds_module.CuFile(
                    tmp_path, "r+", use_direct_io=self.use_direct_io
                ) as f:
                    f.write(
                        addr, kv_chunk.nbytes, file_offset=offset, dev_offset=dev_offset
                    )
            elif self.cudart:
                # mmap the file
                fd = os.open(tmp_path, os.O_RDWR)
                nbytes = kv_chunk.nbytes
                os.ftruncate(fd, nbytes + offset)
                mm = mmap.mmap(
                    fd, nbytes + offset, prot=mmap.PROT_WRITE, flags=mmap.MAP_SHARED
                )
                os.close(fd)

                # get mapped file address
                arr = np.frombuffer(mm, dtype=np.uint8)
                buf_addr = arr.__array_interface__["data"][0]

                assert addr.value is not None
                res = self.cudart.cudaMemcpy(
                    ctypes.c_void_p(buf_addr + offset),
                    ctypes.c_void_p(int(addr.value) + device_offset),
                    ctypes.c_size_t(nbytes),
                    ctypes.c_int(2),
                )
                if res:
                    raise RuntimeError(f"cudaMemcpy failed {res}")
                del arr
                mm.close()

        except Exception as e:
            logger.error(f"Error saving {tmp_path}: {e}", exc_info=True)
            raise e
        os.rename(tmp_path, path)
        return metadata

    def _load_gds(
        self,
        gds_path: str,
        file_offset: int,
        gpu_pointer: ctypes.c_void_p,
        size_in_bytes: int,
        dev_offset: int,
    ) -> int:
        """Read data from disk into a GPU buffer"""
        try:
            if self.gds_module:
                with self.gds_module.CuFile(
                    gds_path, "r", use_direct_io=self.use_direct_io
                ) as f:
                    return f.read(
                        gpu_pointer,
                        size_in_bytes,
                        file_offset=file_offset,
                        dev_offset=dev_offset,
                    )
            elif self.cudart:
                fd = os.open(gds_path, os.O_RDONLY)
                file_size = os.fstat(fd).st_size

                # Check if file is large enough for the requested read
                if file_size < file_offset + size_in_bytes:
                    os.close(fd)
                    logger.error(
                        f"File {gds_path} is too small: size={file_size}, "
                        f"but need at least {file_offset + size_in_bytes} bytes "
                        f"(offset={file_offset}, requested={size_in_bytes})"
                    )
                    return -1

                mm = mmap.mmap(
                    fd,
                    file_size,
                    prot=mmap.PROT_READ,
                    flags=mmap.MAP_PRIVATE | mmap.MAP_POPULATE,  # type: ignore [attr-defined]
                )
                os.close(fd)

                arr = np.frombuffer(mm, dtype=np.uint8)
                addr = arr.__array_interface__["data"][0]

                assert gpu_pointer.value is not None
                res = self.cudart.cudaMemcpy(
                    ctypes.c_void_p(int(gpu_pointer.value) + dev_offset),
                    ctypes.c_void_p(addr + file_offset),
                    ctypes.c_size_t(size_in_bytes),
                    ctypes.c_int(1),
                )

                if res != 0:
                    raise RuntimeError(f"cudaMemcpy failed with code {res}")
                del arr
                mm.close()
                return size_in_bytes
            else:
                raise RuntimeError(
                    "Both gds_module and cudart are None, this should not happen"
                )
        except Exception as e:
            # return -1 on any exception, and log the error.
            # The caller will handle the error by removing the cache entry and
            # returning None.
            logger.error(f"GDS read failed for {gds_path}: {e}", exc_info=True)
            return -1

    def pin(self, key: CacheEngineKey) -> bool:
        # NOTE (ApostaC): Since gds doesn't have eviction now, we don't need
        # to implement pin and unpin
        return False

    def unpin(self, key: CacheEngineKey) -> bool:
        # NOTE (ApostaC): Since gds doesn't have eviction now, we don't need
        # to implement pin and unpin
        return False

    def remove(self, key: CacheEngineKey, force: bool = True):
        raise NotImplementedError("Remote backend does not support remove now.")

    def initialize_allocator(
        self, config: LMCacheEngineConfig, metadata: LMCacheMetadata
    ) -> Union[CuFileMemoryAllocator, HipFileMemoryAllocator]:
        assert config.gds_buffer_size is not None
        allocator_cls = (
            HipFileMemoryAllocator
            if self.gds_backend == "hipfile"
            else CuFileMemoryAllocator
        )
        return allocator_cls(config.gds_buffer_size * 1024**2)

    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
        busy_loop: bool = True,
    ) -> Optional[MemoryObj]:
        """
        Allocate a memory object of shape and dtype
        """
        if eviction:
            logger.warning("GDS Backend does not support eviction")

        logger.debug(f"Allocating memory with busy loop: {busy_loop}")

        max_attempts = self.max_alloc_attempts if busy_loop else 1
        num_attempts = 0

        # try up to max_attempts
        while True:
            memory_obj = self.memory_allocator.allocate(shapes, dtypes, fmt)
            if memory_obj is not None:  # success
                return memory_obj

            num_attempts += 1
            if num_attempts < max_attempts:  # keep trying until max attempts is reached
                logger.debug(
                    f"Unable to allocate memory object after {num_attempts} "
                    f"attempt(s) of GDS backend allocate(). "
                    f"Waiting {self.alloc_attempt_delay_secs} seconds before retrying."
                )
                if self.alloc_attempt_delay_secs > 0:
                    time.sleep(self.alloc_attempt_delay_secs)
            else:  # break to failure case after max attempts is reached
                break

        logger.warning(
            f"GDS allocation failed after {num_attempts} attempt(s). Returning None."
        )
        if not self.memory_allocator.memcheck():
            logger.error(
                "GDS allocation failed and memory allocator "
                "is inconsistent. This is a bug in the memory allocator."
            )
        return None

    def batched_allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
        busy_loop: bool = True,
    ) -> Optional[list[MemoryObj]]:
        """
        Batched allocate `batch_size` memory objects of shape and dtype
        """
        if eviction:
            logger.warning("GDS Backend does not support eviction")

        logger.debug(
            f"Batched allocating memory in GDS backend with busy loop: {busy_loop}"
        )

        max_attempts = self.max_alloc_attempts if busy_loop else 1
        num_attempts = 0

        # try up to max_attempts
        while True:
            memory_objs = self.memory_allocator.batched_allocate(
                shapes, dtypes, batch_size, fmt
            )
            if memory_objs is not None:  # success
                return memory_objs

            num_attempts += 1
            if num_attempts < max_attempts:  # keep trying until max attempts is reached
                logger.debug(
                    f"Unable to allocate memory object after {num_attempts} "
                    f"attempt(s) of GDS backend batched_allocate(). "
                    f"Waiting {self.alloc_attempt_delay_secs} seconds before retrying."
                )
                if self.alloc_attempt_delay_secs > 0:
                    time.sleep(self.alloc_attempt_delay_secs)
            else:  # break to failure case after max attempts is reached
                break

        logger.warning(
            f"GDS batched allocation failed after {num_attempts} "
            f"attempt(s). Returning None."
        )
        if not self.memory_allocator.memcheck():
            logger.error(
                "GDS batched allocation failed and memory allocator "
                "is inconsistent. This is a bug in the memory allocator."
            )
        return None

    def get_allocator_backend(self):
        return self

    def get_memory_allocator(self):
        return self.memory_allocator

    def close(self) -> None:
        # Wait for initial metadata scan to complete
        try:
            self._scan_metadata_future.result(timeout=30)
        except Exception as e:
            logger.warning(
                f"Exception while waiting for metadata scan: {e}",
                exc_info=True,
            )
        # Wait for pending metadata write tasks to finish before tearing down
        # the allocator and thread pool..
        if self.save_metadata_tasks:

            async def _drain_tasks() -> None:
                await asyncio.gather(*self.save_metadata_tasks, return_exceptions=True)
                self.save_metadata_tasks.clear()

            try:
                drain: Future = asyncio.run_coroutine_threadsafe(
                    _drain_tasks(),
                    self.loop,
                )
                drain.result(timeout=30)
            except Exception as e:
                logger.warning(
                    f"Exception while draining metadata write tasks: {e}",
                    exc_info=True,
                )
        self.memory_allocator.close()
        if self._thread_pool is not None:
            self._thread_pool.shutdown(wait=True)
        logger.info("GDS backend closed.")
