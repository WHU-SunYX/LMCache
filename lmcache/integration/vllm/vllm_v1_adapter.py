# SPDX-License-Identifier: Apache-2.0
# Standard
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generator, Optional, Union
import math
import os
import time
import fcntl
import struct

# Third Party
from vllm.config import (
    VllmConfig,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.parallel_state import (
    get_pp_group,
)
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import RequestStatus
from vllm.version import __version__ as VLLM_VERSION
import torch

# First Party
# Use LMCache's own math utilities instead of vllm's
# (avoids dependency on vllm internal changes like https://github.com/vllm-project/vllm/pull/27188)
from lmcache import utils
from lmcache.integration.vllm.utils import (
    ENGINE_NAME,
    apply_mm_hashes_to_token_ids,
    extract_mm_features,
    lmcache_get_or_create_config,
)
from lmcache.integration.vllm.vllm_service_factory import VllmServiceFactory
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor, PrometheusLogger
from lmcache.utils import CacheStoreEvent, _lmcache_nvtx_annotate, cdiv
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.compute.blend import LMCBlenderBuilder
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.config_base import validate_and_set_config_value
from lmcache.v1.manager import LMCacheManager

if TYPE_CHECKING:
    # Third Party
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.multimodal.inputs import PlaceholderRange
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.sched.output import NewRequestData
    from vllm.v1.request import Request

    # First Party
    from lmcache.v1.lookup_client.abstract_client import LookupClientInterface

logger = init_logger(__name__)


class AISSDFiemapFragmentedError(ValueError):
    """Raised when a candidate range is too fragmented after optional merge."""

    def __init__(
        self,
        message: str,
        *,
        path: str,
        file_offset: int,
        nbytes: int,
        raw_extents: int,
        merged_extents: int,
        max_extents: int,
        scan_extents: int,
    ) -> None:
        super().__init__(message)
        self.path = path
        self.file_offset = int(file_offset)
        self.nbytes = int(nbytes)
        self.raw_extents = int(raw_extents)
        self.merged_extents = int(merged_extents)
        self.max_extents = int(max_extents)
        self.scan_extents = int(scan_extents)


def _env_flag(name: str, default: str = "0") -> bool:
    value = os.environ.get(name, default)
    return str(value).lower() not in ("0", "false", "no", "off")


def _sparse_kv_debug_enabled() -> bool:
    # High-frequency sparse KV / connector / Python-route logs.
    return _env_flag("VLLM_SPARSE_KV_DEBUG", "0")


def _sparse_ctx_ptr_debug_enabled() -> bool:
    # data_ptr diagnostics for CUDA graph context lifetime debugging.
    return _env_flag("VLLM_SPARSE_CTX_PTR_DEBUG", "0")


def _sparse_attn_debug_counters_enabled() -> bool:
    # CUDA counter readback can synchronize GPU work; keep disabled by default.
    return _env_flag("VLLM_SPARSE_ATTN_DEBUG_COUNTERS", "0")


def _sparse_fa_replay_debug_enabled() -> bool:
    # Lightweight CUDA-graph replay marker for FA-varlen sparse attention.
    # Disabled by default because draining the device marker synchronizes.
    return _env_flag("VLLM_SPARSE_FA_REPLAY_DEBUG", "0")


def _aissd_extent_stats_enabled() -> bool:
    # Candidate-range FIEMAP/extent distribution statistics.  Keep separate
    # from VLLM_SPARSE_KV_DEBUG because these stats are useful in production
    # sizing tests without enabling high-frequency debug logs.
    return _env_flag("AISSD_SPARSE_KV_EXTENT_STATS", "0")


def _aissd_selector_stats_enabled() -> bool:
    # End-to-end selector timing. Enables HOST Python/CPU-side logs; the C++
    # client and SSD runner use the same environment variable.
    return _env_flag("AISSD_SPARSE_KV_SELECTOR_STATS", "0")


def _aissd_extent_merge_enabled() -> bool:
    # Merge physically contiguous FIEMAP records before enforcing the protocol
    # extent limit.  This is metadata normalization, not fallback.
    return _env_flag("AISSD_SPARSE_KV_EXTENT_MERGE", "1")


def _aissd_fragment_policy() -> str:
    policy = os.environ.get("AISSD_SPARSE_KV_FRAGMENT_POLICY", "fail")
    policy = str(policy).strip().lower()
    if policy not in ("fail", "skip"):
        raise RuntimeError(
            "AISSD_SPARSE_KV_FRAGMENT_POLICY must be 'fail' or 'skip' "
            f"for the no-fallback debug path, got {policy!r}"
        )
    return policy


# Global worker-side sparse KV connector registry.
# SparseSSDAttentionImpl fetches the worker connector context from here.
# This is not the old Attention.forward Python selection hook; it only provides
# access to graph-stable persistent step-context tensors.
_SPARSE_KV_CONNECTOR: Optional["LMCacheConnectorV1Impl"] = None


def set_sparse_kv_connector(connector: Optional["LMCacheConnectorV1Impl"]) -> None:
    global _SPARSE_KV_CONNECTOR
    _SPARSE_KV_CONNECTOR = connector


def get_sparse_kv_connector() -> Optional["LMCacheConnectorV1Impl"]:
    connector = _SPARSE_KV_CONNECTOR
    if connector is None:
        return None
    spec = getattr(connector, "sparse_kv_spec", None)
    if spec is not None and getattr(spec, "enabled", False):
        return connector
    return None


@dataclass
class LoadSpec:
    # Number of tokens cached in vLLM
    vllm_cached_tokens: int
    # Number of tokens that are cached in LMCache
    lmcache_cached_tokens: int
    # Whether the scheduler allow us to load the tokens
    can_load: bool


@dataclass
class SaveSpec:
    # Skip already saved tokens
    skip_leading_tokens: int
    # Whether the scheduler allow us to save the tokens
    can_save: bool


@dataclass
class SparseKVSpec:
    enabled: bool = False
    granularity: str = "chunk"
    top_n_chunks: int = 0
    score_mode: str = "topm_mean"
    disable_full_load: bool = False
    # Whether to route attention to SparseSSDAttentionImpl when sparse metadata
    # is available.  This is intentionally separate from enable_sparse_kv_cache:
    # sparse KV selection/load can run in dry-run mode while full attention
    # remains the correctness baseline.
    enable_sparse_attention: bool = False
    sparse_kv_backend: str = "host"


@dataclass
class DisaggSpec:
    req_id: str
    receiver_id: str
    receiver_host: str
    receiver_init_port: int
    receiver_alloc_port: int
    is_last_prefill: bool = False
    num_transferred_tokens: int = 0
    total_chunks: int = 0
    receiver_query_port: Optional[list[int]] = None


tmp_disagg_tracker: dict[str, DisaggSpec] = {}


def extract_request_configs(sampling_params: SamplingParams) -> Optional[dict]:
    request_configs = None
    if sampling_params and sampling_params.extra_args is not None:
        if kv_transfer_params := sampling_params.extra_args.get("kv_transfer_params"):
            for k, v in kv_transfer_params.items():
                if k.startswith("lmcache."):
                    if request_configs is None:
                        request_configs = {}
                    request_configs[k] = v
    return request_configs


@dataclass
class RequestTracker:
    # Request id
    req_id: str

    # Total prompt token length
    prompt_len: int

    # The token ids that has been scheduled so far
    token_ids: list[int]

    # The block ids that has been allocated so far
    # NOTE: allocated blocks could be more than the number of tokens
    allocated_block_ids: list[int]

    # The number of tokens that has been saved
    num_saved_tokens: int = 0

    # Disagg spec for the request
    disagg_spec: Optional[DisaggSpec] = None

    # Multimodal hashes and positions
    mm_hashes: Optional[list[str]] = None
    mm_positions: Optional[list["PlaceholderRange"]] = None

    # The configs of the request, includes tags and other configs
    request_configs: Optional[dict] = None

    # Whether the request is in decode phase
    is_decode_phase = False

    # Whether the request cache should be saved
    skip_save: bool = False

    # The number of tokens that are cached in LMCache for this request
    num_lmcache_cached_tokens: int = 0

    @_lmcache_nvtx_annotate
    @staticmethod
    def from_new_request(
        lmcache_config: LMCacheEngineConfig,
        new_request: "NewRequestData",
        num_tokens_to_compute: int,
        lmcache_cached_tokens: int,
        skip_save: bool,
    ) -> "RequestTracker":
        """Create the request tracker from a new request.

        Args:
            lmcache_config (LMCacheEngineConfig): the LMCache engine config.
            new_request (NewRequestData): the new request data.
            num_tokens_to_compute (int): the number of tokens that will
                be 'computed', including the `num_computed_tokens` (vLLM's
                local cache hit) and new tokens that will be scheduled.
            lmcache_cached_tokens (int): the number of tokens that are
                cached in LMCache.
            request_priority (int): the priority of the request
            skip_save (bool): whether the request cache should be saved
        """
        # vLLM 0.9.0 update: request.block_ids changed from list[int] to
        # tuple[list[int]]
        # Need to check the type of request.block_ids

        unfolded_block_ids = []

        if not isinstance(new_request.block_ids[0], list):
            unfolded_block_ids = new_request.block_ids.copy()
        else:
            # According to the vLLM code
            # (https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/
            # sched/scheduler.py#L943),
            # only one KVCacheGroup is supported in connector for now.

            # TODO: Please support multiple KVCacheGroup in connector.
            # NOTE: Also, `update` method in RequestTracker should be
            # updated accordingly.
            unfolded_block_ids = new_request.block_ids[0].copy()

        # NOTE: Initialized in `update_state_after_alloc`
        disagg_spec = tmp_disagg_tracker.pop(new_request.req_id, None)

        request_configs = extract_request_configs(new_request.sampling_params)

        mm_hashes, mm_positions = extract_mm_features(new_request, modify=True)

        return RequestTracker(
            req_id=new_request.req_id,
            prompt_len=len(new_request.prompt_token_ids),
            token_ids=new_request.prompt_token_ids[:num_tokens_to_compute].copy(),
            allocated_block_ids=unfolded_block_ids,
            num_saved_tokens=lmcache_cached_tokens,
            disagg_spec=disagg_spec,
            mm_hashes=mm_hashes,
            mm_positions=mm_positions,
            skip_save=skip_save,
            request_configs=request_configs,
            num_lmcache_cached_tokens=lmcache_cached_tokens,
        )

    def update(
        self,
        new_token_ids: list[int],
        new_block_ids: Union[Optional[tuple[list[int], ...]], list[int]],
        preempted: bool = False,
        lmcache_cached_tokens: int = 0,
        vllm_cached_tokens: int = 0,
        all_token_ids: Optional[list[int]] = None,
    ) -> None:
        """Update the request tracker when a running request is
        scheduled again

        vllm_cached_tokens: the number of tokens that are cached in vLLM
        is only used for preempted requests
        all_token_ids: the full token list from the vLLM request, used to
        restore token_ids for preempted requests to ensure chunk keys match
        """

        if new_block_ids is None:
            # https://github.com/vllm-project/vllm/commit/
            # b029de9902aa3ac58806c8c17776c7074175b6db#
            # diff-cafd89ce8a698a56acb24ada62831cbc7a980782f78a52d1742ba238031f296cL94
            new_block_ids = []
        elif len(new_block_ids) == 0:
            new_block_ids = []
        elif isinstance(new_block_ids, tuple):
            new_block_ids = new_block_ids[0]
        elif isinstance(new_block_ids, list):
            # If input is a list, flatten it to handle potential nesting.
            # This also correctly processes already-flat lists.
            new_block_ids = [
                i
                for elem in new_block_ids
                for i in (elem if isinstance(elem, list) else [elem])
            ]
        else:
            raise ValueError(f"Unsupported new_block_ids type {type(new_block_ids)}")

        if preempted:
            assert all_token_ids is not None, (
                f"Preempted request {self.req_id} has no all_token_ids"
            )
            # the block ids will change after preemption
            self.allocated_block_ids = new_block_ids
            # reset the number of saved tokens
            self.num_saved_tokens = lmcache_cached_tokens
            num_computed_tokens = max(lmcache_cached_tokens, vllm_cached_tokens)

            # FIX: For preempted requests, restore token_ids from the full
            # token list to ensure chunk keys match what was used during
            # lookup. The lookup uses request.all_token_ids, so we need the
            # same tokens for retrieve.
            num_tokens_needed = max(
                num_computed_tokens + len(new_token_ids),
                lmcache_cached_tokens,
            )
            self.token_ids = all_token_ids[:num_tokens_needed]
        else:
            self.allocated_block_ids.extend(new_block_ids)
            self.token_ids.extend(new_token_ids)

        # When a request is scheduled again, and the number of new tokens
        # is 1 (excluding chunked prefill), the request is in decode phase.
        # TODO: Need to further exclude the case of chunked prefill with 1 token.
        if len(new_token_ids) == 1:
            self.is_decode_phase = True


@dataclass
class ReqMeta:
    # Request id
    req_id: str
    # Request tokens
    token_ids: list[int]  # torch.Tensor
    # Slot mapping
    slot_mapping: torch.Tensor

    # Whether is last prefill or not
    is_last_prefill: bool = False

    # Skip save or not
    save_spec: Optional[SaveSpec] = None
    # load_spec
    load_spec: Optional[LoadSpec] = None
    # disagg spec
    disagg_spec: Optional[DisaggSpec] = None
    # the configs of the request
    request_configs: Optional[dict] = None
    # Optional AI-SSD sparse KV selector configuration. When enabled, the
    # worker-side attention hook builds a q_manifest + candidate manifest for
    # SSD-CPU/NPU chunk-level selection.
    sparse_kv_spec: Optional[SparseKVSpec] = None

    @staticmethod
    def from_request_tracker(
        tracker: RequestTracker,
        block_size: int,
        lmcache_chunk_size: int = 256,
        load_spec: Optional[LoadSpec] = None,
        discard_partial_chunks: bool = True,
        save_decode_cache: bool = False,
    ) -> Optional["ReqMeta"]:
        """Create the request metadata from a request tracker.

        Args:
            tracker (RequestTracker): the request tracker.
            block_size (int): the block size in vLLM.
            lmcache_chunk_size (int): the chunk size for LMCache.
            load_spec (Optional[LoadSpec]): the load spec for KV cache loading.
            discard_partial_chunks (bool): whether to discard partial chunks.
            save_decode_cache (bool): whether to save the cache in decode phase.

        Returns:
            the request metadata if we need to perform load/save
            operations, None otherwise.
        """
        input_token_ids = tracker.token_ids
        input_token_len = len(input_token_ids)

        is_last_prefill = False
        if input_token_len >= tracker.prompt_len:
            is_last_prefill = True

        # For save operation: do not save if the following condition is met
        # 1. has already been saved before (num_saved_tokens > 0)
        # 2. number of unsaved tokens is not reached the chunk boundary
        # 3. if save_decode_cache is False and it is in decode phase

        skip_leading_tokens = tracker.num_saved_tokens
        chunk_boundary = (
            cdiv(tracker.num_saved_tokens + 1, lmcache_chunk_size) * lmcache_chunk_size
        )

        # NOTE(vladnosiv): for disagg, you cannot skip saving, as saving is a transfer
        # Check if request_configs has lmcache.skip_save set to True
        request_skip = (tracker.request_configs or {}).get("lmcache.skip_save", False)

        skip_save = tracker.disagg_spec is None and (
            tracker.skip_save
            or (tracker.num_saved_tokens > 0 and input_token_len < chunk_boundary)
            or (tracker.is_decode_phase and not save_decode_cache)
            or request_skip
        )

        if skip_save and load_spec is None:
            return None

        # Calculate number of tokens to save based on discard_partial_chunks
        # setting

        # NOTE(vladnosiv): for the input_token_len chunk prefill,
        # we are required to discard partial chunks,
        # as new tokens will be added in the next iteration.
        if not is_last_prefill or discard_partial_chunks:
            num_tokens_to_save = (
                input_token_len // lmcache_chunk_size * lmcache_chunk_size
            )
        else:
            num_tokens_to_save = input_token_len

        # If we need to save, update the number of saved tokens
        if not skip_save:
            tracker.num_saved_tokens = num_tokens_to_save
        save_spec = SaveSpec(skip_leading_tokens, not skip_save)

        # Calculate the token ids and slot mappings for load and save
        token_ids = input_token_ids[:num_tokens_to_save]

        # If the request has multimodal hashes, apply them to the token ids
        if tracker.mm_hashes:
            # TODO: Optimize this
            token_ids = torch.tensor(token_ids)
            assert tracker.mm_positions is not None, (
                "tracker got mm_hashes but no mm_positions"
            )
            apply_mm_hashes_to_token_ids(
                token_ids, tracker.mm_hashes, tracker.mm_positions
            )
            token_ids = token_ids.tolist()

        num_blocks = len(tracker.allocated_block_ids)

        if len(token_ids) > num_blocks * block_size:
            logger.error(
                "The number of tokens is more than the number of blocks"
                " for request %s. "
                "Something might be wrong in scheduling logic!",
                tracker.req_id,
            )
            logger.error(
                "Num tokens: %d, num blocks: %d, block size: %d",
                len(token_ids),
                num_blocks,
                block_size,
            )

        block_ids = torch.tensor(tracker.allocated_block_ids, dtype=torch.long)
        block_offsets = torch.arange(0, block_size, dtype=torch.long)
        slot_mapping = (
            block_offsets.reshape((1, block_size))
            + block_ids.reshape((num_blocks, 1)) * block_size
        )

        slot_mapping = slot_mapping.flatten()[: len(token_ids)]
        assert slot_mapping.dtype == torch.long  # TODO: this could be removed

        # For load operation: log if the request is scheduled to load
        if load_spec is not None and load_spec.can_load:
            logger.debug(
                "Scheduled to load %d tokens (%d cached in vLLM) for request %s",
                load_spec.lmcache_cached_tokens,
                load_spec.vllm_cached_tokens,
                tracker.req_id,
            )

        # For disagg requests, compute total_chunks for sender admission control.
        if tracker.disagg_spec is not None and tracker.disagg_spec.total_chunks == 0:
            # Only compute once (on first batch)
            total_chunks_for_req = math.ceil(tracker.prompt_len / lmcache_chunk_size)
            tracker.disagg_spec.total_chunks = total_chunks_for_req

        # Note: We keep load_spec even when can_load=False to pass metrics to worker
        return ReqMeta(
            req_id=tracker.req_id,
            token_ids=token_ids,
            slot_mapping=slot_mapping,
            is_last_prefill=is_last_prefill,
            save_spec=save_spec,
            load_spec=load_spec,
            disagg_spec=tracker.disagg_spec,
            request_configs=tracker.request_configs,
        )


@dataclass
class LMCacheConnectorMetadata(KVConnectorMetadata):
    requests: list[ReqMeta] = field(default_factory=list)

    @_lmcache_nvtx_annotate
    def add_request(self, req_meta: ReqMeta) -> None:
        """Add a request to the metadata.

        Args:
            req_meta (ReqMeta): the request metadata.
        """
        self.requests.append(req_meta)


class LMCacheConnectorV1Impl:
    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        parent: KVConnectorBase_V1,
    ):
        self._parent = parent
        self._vllm_config = vllm_config
        self._role = role
        self.device = vllm_config.device_config.device
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self.worker_count = vllm_config.parallel_config.tensor_parallel_size

        # Load and configure LMCache config
        config = lmcache_get_or_create_config()
        assert isinstance(config, LMCacheEngineConfig), (
            "LMCache v1 configuration is should be passed for vLLM v1."
        )
        self._apply_extra_config(config, vllm_config)
        self.config = config

        service_factory = VllmServiceFactory(config, vllm_config, role.name.lower())
        self._manager = LMCacheManager(config, service_factory, connector=self)

        # Start services managed by LMCacheManager
        self._manager.start_services()

        # Initialize connector-specific state
        self._init_connector_state(role, vllm_config, config)

        # Register the worker-side sparse KV connector for the vLLM attention
        # hook. The scheduler-side connector is not used by Attention.forward.
        if role == KVConnectorRole.WORKER and getattr(self, "sparse_kv_spec", SparseKVSpec()).enabled:
            set_sparse_kv_connector(self)
            logger.info(
                "[sparse-kv] registered LMCache worker connector for sparse attention context"
            )

        # Setup metrics for monitoring data structures
        self._setup_metrics()

        logger.info(
            "LMCache initialized for role %s with version %s, "
            "vllm version %s, lmcache cache_engine metadata: %s",
            role,
            utils.get_version(),
            VLLM_VERSION,
            getattr(self.lmcache_engine, "metadata", None),
        )

    def _apply_extra_config(
        self, config: LMCacheEngineConfig, vllm_config: "VllmConfig"
    ) -> None:
        """Apply extra config from vLLM to LMCache config."""
        kv_connector_extra_config = (
            vllm_config.kv_transfer_config.kv_connector_extra_config
        )
        if kv_connector_extra_config:
            for key, value in kv_connector_extra_config.items():
                if key.startswith("lmcache."):
                    config_key = key[8:]  # Remove "lmcache." prefix
                    if validate_and_set_config_value(config, config_key, value):
                        logger.info(
                            "Updated config %s from vLLM extra config",
                            config_key,
                        )

    def _init_connector_state(
        self,
        role: KVConnectorRole,
        vllm_config: "VllmConfig",
        config: LMCacheEngineConfig,
    ) -> None:
        """Initialize connector-specific state variables."""
        self.async_loading = config.enable_async_loading
        self.layerwise_retrievers: list[
            Generator[Optional[torch.Tensor], None, None]
        ] = []
        self._layerwise_save_storers: dict[
            str, Generator[Optional[torch.Tensor], None, None]
        ] = {}
        self._stats_monitor = LMCStatsMonitor.GetOrCreate()

        # Role-specific initialization
        if role == KVConnectorRole.SCHEDULER:
            self._unfinished_requests: dict[str, "Request"] = {}
        else:
            self.use_layerwise = config.use_layerwise
            self.enable_blending = config.enable_blending

            if self.enable_blending:
                assert self.lmcache_engine is not None
                assert self.lmcache_engine.gpu_connector is not None, (
                    "GPU connector must be available for blending"
                )
                self.blender = LMCBlenderBuilder.get_or_create(
                    ENGINE_NAME,
                    self.lmcache_engine,
                    self.lmcache_engine.gpu_connector,
                    config,
                )

        # Legacy compatibility check
        self._check_legacy_register_kv_caches()

        self.kv_caches: dict[str, torch.Tensor] = {}
        self._block_size = vllm_config.cache_config.block_size
        self.load_specs: dict[str, LoadSpec] = {}
        self.kv_cache_manager: Optional["KVCacheManager"] = None
        self._request_trackers: dict[str, RequestTracker] = {}

        self._discard_partial_chunks = (
            vllm_config.kv_transfer_config.get_from_extra_config(
                "discard_partial_chunks", False
            )
            or not config.save_unfull_chunk
        )

        self._lmcache_chunk_size = config.chunk_size

        self.skip_last_n_tokens = vllm_config.kv_transfer_config.get_from_extra_config(
            "skip_last_n_tokens", 0
        )

        self.num_layers = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        try:
            self._sparse_num_kv_heads = int(
                vllm_config.model_config.get_num_kv_heads(vllm_config.parallel_config)
            )
        except Exception:
            self._sparse_num_kv_heads = 0
        try:
            self._sparse_head_size = int(vllm_config.model_config.get_head_size())
        except Exception:
            self._sparse_head_size = 0
        self.current_layer = 0

        self.force_skip_save = bool(os.environ.get("LMCACHE_FORCE_SKIP_SAVE", False))
        self._requests_priority: dict[str, int] = {}
        self._invalid_block_ids: set[int] = set()

        extra_cfg = {}
        extra_cfg.update(getattr(config, "extra_config", {}) or {})
        kv_extra = getattr(vllm_config.kv_transfer_config, "kv_connector_extra_config", None)
        if kv_extra:
            extra_cfg.update(kv_extra)

        def _extra_bool(name: str, default: bool = False) -> bool:
            value = extra_cfg.get(name, extra_cfg.get(f"lmcache.{name}", default))
            if isinstance(value, str):
                return value.lower() in ("1", "true", "yes", "on")
            return bool(value)

        def _extra_int(name: str, default: int = 0) -> int:
            value = extra_cfg.get(name, extra_cfg.get(f"lmcache.{name}", default))
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        def _extra_str(name: str, default: str) -> str:
            value = extra_cfg.get(name, extra_cfg.get(f"lmcache.{name}", default))
            return str(value)

        self.sparse_kv_spec = SparseKVSpec(
            enabled=_extra_bool("enable_sparse_kv_cache", False),
            granularity=_extra_str("sparse_kv_granularity", "chunk"),
            top_n_chunks=_extra_int("sparse_kv_top_n_chunks", 0),
            score_mode=_extra_str("sparse_kv_score_mode", "topm_mean"),
            disable_full_load=_extra_bool("sparse_kv_disable_full_load", False),
            enable_sparse_attention=_extra_bool("enable_sparse_attention", False),
            sparse_kv_backend=_extra_str("sparse_kv_backend", "host"),
        )
        if self.sparse_kv_spec.sparse_kv_backend not in ("host", "ssd-cpu", "ssd-npu"):
            raise ValueError(f"Unsupported lmcache.sparse_kv_backend={self.sparse_kv_spec.sparse_kv_backend!r}")
        if self.sparse_kv_spec.enabled:
            logger.info("LMCache sparse KV enabled: %s", self.sparse_kv_spec)

            # AISSD ssd-cpu/ssd-npu q-aware selection still must execute
            # outside CUDA graph replay, because it performs HOST<->SSD CMB/RPC
            # after the real per-layer Q tensor is available.  Do NOT mutate the
            # global vLLM compilation_config here: doing so disables CUDA graphs
            # for the whole engine and is only useful for bring-up.  The runtime
            # vLLM GPU model runner patch checks this same sparse config and
            # sets cudagraph_mode=NONE only for real AISSD steps.
            if self.sparse_kv_spec.sparse_kv_backend in ("ssd-cpu", "ssd-npu"):
                if _env_flag("AISSD_SPARSE_KV_ALLOW_CUDAGRAPH", "0"):
                    logger.warning(
                        "[warning][aissd-selector-cudagraph-allowed] backend=%s "
                        "override=AISSD_SPARSE_KV_ALLOW_CUDAGRAPH=1 risk=CUDA graph "
                        "replay may skip aissd_sparse_kv_select unless the runner "
                        "uses per-step eager fallback",
                        self.sparse_kv_spec.sparse_kv_backend,
                    )
                else:
                    logger.warning(
                        "[warning][aissd-selector-step-eager-required] backend=%s "
                        "reason=HOST/SSD selector needs Python/C++ bridge with real Q; "
                        "global CUDA graph capture remains enabled, but real AISSD "
                        "steps must dispatch with CUDAGraphMode.NONE in the vLLM "
                        "GPU model runner. override=AISSD_SPARSE_KV_ALLOW_CUDAGRAPH=1",
                        self.sparse_kv_spec.sparse_kv_backend,
                    )

        # Sparse attention uses CUDA graphs, so the device tensor addresses in
        # the step context must stay stable after capture.  Do not allocate a
        # tiny 1x1 empty context during capture and then reallocate larger
        # tensors for real requests: graph replay would keep reading the old
        # addresses.  Pre-size the persistent context to the configured runtime
        # envelope.
        scheduler_config = getattr(vllm_config, "scheduler_config", None)
        model_config = getattr(vllm_config, "model_config", None)
        compilation_config = getattr(vllm_config, "compilation_config", None)
        # CUDA graph capture can use padded batch sizes larger than
        # --max-num-seqs (for example capture size 32 while max_num_seqs=16).
        # Sparse FA-varlen passes block_table/seq_lens rows indexed by the
        # captured query batch size, so the persistent context must cover the
        # larger of runtime max_num_seqs and max_cudagraph_capture_size.
        self._sparse_max_reqs = max(
            1,
            int(getattr(scheduler_config, "max_num_seqs", 1) or 1),
            int(getattr(compilation_config, "max_cudagraph_capture_size", 0) or 0),
        )
        self._sparse_max_slots = max(
            1,
            int(getattr(model_config, "max_model_len", 0) or 0)
            or int(getattr(scheduler_config, "max_model_len", 0) or 0)
            or int(getattr(scheduler_config, "max_num_batched_tokens", 1) or 1),
        )
        blocks_per_chunk = cdiv(
            max(1, int(self._lmcache_chunk_size)),
            max(1, int(self._block_size)),
        )
        self._sparse_max_selected_blocks = max(
            1,
            int(getattr(self.sparse_kv_spec, "top_n_chunks", 0) or 0)
            * int(blocks_per_chunk),
        )
        # Monotonic host-side metadata for the persistent sparse context.
        # Device tensors are graph-visible, but Python-side routing/logging must
        # not decide from a stale attn_metadata object that was populated during
        # CUDA graph capture.  The attention backend compares these host fields
        # and prefers the connector's newest context.
        self._sparse_step_generation = 0
        self._sparse_active_context_pending = False
        self._sparse_persistent_step_context: Optional[dict[str, Any]] = None
        self._sparse_current_step_context: Optional[dict[str, Any]] = None

        # Pre-allocate the graph-visible sparse context before CUDA graph
        # capture can reach Attention.forward().  The captured graph must bind
        # the same device tensor addresses that runtime prepare_sparse_kv_step()
        # updates in-place.  This is not an eager fallback; it is the normal
        # CUDA-graph-compatible lifetime model.
        if getattr(self.sparse_kv_spec, "enabled", False):
            try:
                self._sparse_current_step_context = self._ensure_sparse_step_buffers(
                    self._sparse_max_reqs,
                    self._sparse_max_slots,
                    self._sparse_max_selected_blocks,
                )
            except Exception:
                logger.exception("[sparse-kv-step] failed to preallocate persistent sparse context")
                raise

    def _check_legacy_register_kv_caches(self) -> None:
        """Check for legacy connector without register_kv_caches implementation."""
        if self.lmcache_engine is None:
            return

        child_class = self._parent.__class__
        parent_class = KVConnectorBase_V1
        child_method = getattr(child_class, "register_kv_caches", None)
        parent_method = getattr(parent_class, "register_kv_caches", None)

        if child_method is None or parent_method is None:
            implements = False
        else:
            implements = child_method is not parent_method

        if not implements:
            logger.warning(
                "Please use the latest lmcache connector, otherwise some "
                "features may not work, such as DSA"
            )
            self._manager.post_init()

    # ==================== Property Accessors ====================

    @property
    def lmcache_engine(self) -> Optional[LMCacheEngine]:
        """Get the LMCache engine instance from manager."""
        return self._manager.lmcache_engine

    @property
    def lmcache_engine_metadata(self):
        """Get the LMCache engine metadata from manager."""
        return self._manager.lmcache_engine_metadata

    @property
    def lookup_client(self) -> Optional["LookupClientInterface"]:
        """Get the lookup client from manager."""
        return self._manager.lookup_client

    @property
    def lookup_server(self):
        """Get the lookup server from manager."""
        return self._manager.lookup_server

    def _setup_metrics(self) -> None:
        """Setup metrics for monitoring data structures in the connector."""
        metadata = self._manager.lmcache_engine_metadata
        if metadata is None:
            logger.warning(
                "LMCache metadata is not initialized, "
                "connector metrics will not be collected"
            )
            return
        prometheus_logger = PrometheusLogger.GetOrCreate(
            metadata,
            config=self.config,
        )

        # Set up metrics for scheduler-specific and general data structures
        metrics_map = {
            "_unfinished_requests": "scheduler_unfinished_requests_count",
            "load_specs": "connector_load_specs_count",
            "_request_trackers": "connector_request_trackers_count",
            "kv_caches": "connector_kv_caches_count",
            "layerwise_retrievers": "connector_layerwise_retrievers_count",
            "_invalid_block_ids": "connector_invalid_block_ids_count",
            "_requests_priority": "connector_requests_priority_count",
        }

        for attr_name, metric_name in metrics_map.items():
            if hasattr(self, attr_name):
                metric = getattr(prometheus_logger, metric_name)
                # Use a default argument in the lambda to capture
                # the current value of `attr_name`
                # to avoid issues with late binding in closures.
                metric.set_function(lambda name=attr_name: len(getattr(self, name)))

    def get_inference_info(self) -> dict:
        """Get inference information including vLLM config and related details.

        Returns:
            dict: Dictionary containing inference information
        """
        # Get vLLM config information
        vllm_config = self._vllm_config

        # Use vLLM config's string representation and add specific configs
        inference_info = {
            "vllm_version": VLLM_VERSION,
            "lmcache_version": utils.get_version(),
            "vllm_config": str(vllm_config),
            "model_config": {
                "model": getattr(vllm_config.model_config, "model", None),
                "dtype": str(getattr(vllm_config.model_config, "dtype", None)),
                "max_model_len": getattr(
                    vllm_config.model_config, "max_model_len", None
                ),
                "vocab_size": getattr(vllm_config.model_config, "vocab_size", None),
                "num_layers": getattr(
                    vllm_config.model_config, "get_num_layers", lambda _: None
                )(vllm_config.parallel_config),
                "num_attention_heads": getattr(
                    vllm_config.model_config, "get_num_attention_heads", lambda _: None
                )(vllm_config.parallel_config),
                "num_kv_heads": getattr(
                    vllm_config.model_config, "get_num_kv_heads", lambda _: None
                )(vllm_config.parallel_config),
                "head_size": getattr(
                    vllm_config.model_config, "get_head_size", lambda: None
                )(),
            },
            "cache_config": {
                "block_size": getattr(vllm_config.cache_config, "block_size", None),
                "cache_dtype": str(
                    getattr(vllm_config.cache_config, "cache_dtype", None)
                ),
                "gpu_memory_utilization": getattr(
                    vllm_config.cache_config, "gpu_memory_utilization", None
                ),
                "swap_space": getattr(vllm_config.cache_config, "swap_space", None),
                "enable_prefix_caching": getattr(
                    vllm_config.cache_config, "enable_prefix_caching", None
                ),
            },
        }

        return inference_info

    def get_inference_version(self) -> str:
        """Get vLLM version information.

        Returns:
            str: vLLM version string
        """
        return VLLM_VERSION

    # TODO(chunxiaozheng): in the latest lmcache_connector, we use `register_kv_caches`
    #  to init self.kv_caches, we keep it in order to be compatible with old versions
    #  and will be removed in the future.
    @_lmcache_nvtx_annotate
    def _init_kv_caches_from_forward_context(self, forward_context: "ForwardContext"):
        for layer_name in forward_context.no_compile_layers:
            attn_layer = forward_context.no_compile_layers[layer_name]
            if not hasattr(attn_layer, "kv_cache"):
                logger.debug("The layer %s does not have kv_cache, skip it", layer_name)
                continue

            if layer_name not in self.kv_caches:
                self.kv_caches[layer_name] = attn_layer.kv_cache[
                    forward_context.virtual_engine
                ]

    ####################
    # Worker side APIs
    ####################
    @_lmcache_nvtx_annotate
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        logger.info("Registering KV caches")
        # TODO(chunxiaozheng): `_init_kv_caches_from_forward_context` is
        #  not called, we should consider removing it.
        assert len(self.kv_caches) == 0 and len(kv_caches) > 0
        self.kv_caches = kv_caches
        self._manager.post_init()

    @_lmcache_nvtx_annotate
    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        """Start loading the KV cache from the connector buffer to vLLM's
        paged KV buffer.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation
        """
        self.current_layer = 0

        if len(self.kv_caches) == 0:
            logger.warning(
                "Please update LMCacheConnector, "
                "use register_kv_caches to init kv_caches"
            )
            self._init_kv_caches_from_forward_context(forward_context)

        metadata = self._parent._get_connector_metadata()
        assert isinstance(metadata, LMCacheConnectorMetadata)
        if _sparse_kv_debug_enabled():
            logger.info(
                "[lmcache-kv-iface] start_load_kv metadata_requests=%d",
                len(metadata.requests),
            )
        for _i, _req in enumerate(metadata.requests):
            _ls = getattr(_req, "load_spec", None)
            _ss = getattr(_req, "sparse_kv_spec", None)
            if _sparse_kv_debug_enabled():
                logger.info(
                    "[lmcache-kv-iface] start_load_kv request[%d] req_id=%s "
                    "token_ids=%d slot_mapping=%s load_spec=%s can_load=%s "
                    "vllm_cached=%s lmcache_cached=%s save_can=%s sparse=%s "
                    "disable_full_load=%s",
                    _i,
                    getattr(_req, "req_id", None),
                    len(getattr(_req, "token_ids", []) or []),
                    tuple(getattr(getattr(_req, "slot_mapping", None), "shape", [])),
                    _ls is not None,
                    getattr(_ls, "can_load", None),
                    getattr(_ls, "vllm_cached_tokens", None),
                    getattr(_ls, "lmcache_cached_tokens", None),
                    getattr(getattr(_req, "save_spec", None), "can_save", None),
                    bool(getattr(_ss, "enabled", False)) if _ss is not None else False,
                    bool(getattr(_ss, "disable_full_load", False)) if _ss is not None else False,
                )

        assert len(self.kv_caches) > 0
        kvcaches = list(self.kv_caches.values())

        # NOTE: LMCache retrieve does not require attention metadata.  Some
        # vLLM execution paths call start_load_kv() before attn_metadata is
        # attached to the ForwardContext.  Do not return here, otherwise the
        # worker can receive metadata_requests>0 but never actually invoke
        # lmcache_engine.retrieve().  Sparse attention metadata is produced
        # later by sparse_select_kv_layer(), when Attention.forward provides
        # the real attn_metadata object.
        attn_metadata = getattr(forward_context, "attn_metadata", None)
        if attn_metadata is None:
            if _sparse_kv_debug_enabled():
                logger.info(
                    "[lmcache-kv-iface] start_load_kv attn_metadata=None; "
                    "continue retrieve because retrieve does not require attn_metadata"
                )

        assert self.lmcache_engine is not None

        self.layerwise_retrievers = []
        # Runtime information needed by the q-aware sparse selected-load path.
        # Reset once per model forward step.
        self._sparse_runtime_requests: dict[str, dict[str, Any]] = {}
        self._sparse_loaded_chunk_hashes: set[tuple[str, str]] = set()

        last_idx = -1
        for idx, request in enumerate(metadata.requests):
            if request.load_spec is None or not request.load_spec.can_load:
                continue
            last_idx = idx

        for idx, request in enumerate(metadata.requests):
            # Update metrics for all requests that have a load_spec
            if request.load_spec is not None:
                self._stats_monitor.update_interval_vllm_hit_tokens(
                    request.load_spec.vllm_cached_tokens
                )
                self._stats_monitor.update_interval_prompt_tokens(
                    len(request.token_ids)
                )

            if request.load_spec is None or not request.load_spec.can_load:
                if _sparse_kv_debug_enabled():
                    logger.info(
                        "[lmcache-kv-iface] start_load_kv skip request req_id=%s "
                        "load_spec=%s can_load=%s",
                        getattr(request, "req_id", None),
                        request.load_spec is not None,
                        getattr(request.load_spec, "can_load", None)
                        if request.load_spec is not None else None,
                    )
                continue

            tokens = request.token_ids
            # TODO: have a pre-allocated buffer to hold the slot_mappings
            slot_mapping = request.slot_mapping.to(self.device)
            assert len(tokens) == len(slot_mapping)

            token_mask = torch.ones(len(tokens), dtype=torch.bool)
            masked_token_count = (
                request.load_spec.vllm_cached_tokens
                // self._lmcache_chunk_size
                * self._lmcache_chunk_size
            )
            token_mask[:masked_token_count] = False

            lmcache_cached_tokens = request.load_spec.lmcache_cached_tokens
            num_load_tokens = max(
                0, lmcache_cached_tokens - request.load_spec.vllm_cached_tokens
            )
            true_token_count = int(token_mask[:lmcache_cached_tokens].sum().item())
            if _sparse_kv_debug_enabled():
                logger.info(
                    "[lmcache-kv-iface] start_load_kv begin request req_id=%s "
                    "idx=%d token_ids=%d slot_mapping=%d lmcache_cached=%d "
                    "vllm_cached=%d masked_prefix=%d token_mask_true=%d "
                    "expected_load_tokens=%d use_layerwise=%s",
                    request.req_id,
                    idx,
                    len(tokens),
                    int(slot_mapping.numel()),
                    lmcache_cached_tokens,
                    request.load_spec.vllm_cached_tokens,
                    masked_token_count,
                    true_token_count,
                    num_load_tokens,
                    self.use_layerwise,
                )
            sparse_spec = getattr(request, "sparse_kv_spec", None)
            if sparse_spec is not None and sparse_spec.enabled:
                self._sparse_runtime_requests[request.req_id] = {
                    "tokens": tokens[:lmcache_cached_tokens],
                    "token_mask": token_mask[:lmcache_cached_tokens],
                    "slot_mapping": slot_mapping[:lmcache_cached_tokens],
                    "lmcache_cached_tokens": lmcache_cached_tokens,
                    "vllm_cached_tokens": request.load_spec.vllm_cached_tokens,
                    "request_configs": request.request_configs,
                    "sparse_spec": sparse_spec,
                }
                if _sparse_kv_debug_enabled():
                    logger.info(
                        "[sparse-kv-load] runtime cached req_id=%s tokens=%d "
                        "slot_mapping=%d vllm_cached=%d lmcache_cached=%d",
                        request.req_id,
                        len(tokens[:lmcache_cached_tokens]),
                        int(slot_mapping[:lmcache_cached_tokens].numel()),
                        request.load_spec.vllm_cached_tokens,
                        lmcache_cached_tokens,
                    )
            else:
                if _sparse_kv_debug_enabled():
                    logger.info(
                        "[sparse-kv-load] runtime not cached req_id=%s sparse_spec=%s",
                        request.req_id,
                        sparse_spec,
                    )
            sparse_prod_enabled = (
                sparse_spec is not None
                and sparse_spec.enabled
                and getattr(sparse_spec, "enable_sparse_attention", False)
            )
            skip_full_retrieve = bool(
                sparse_prod_enabled
                or (
                    sparse_spec is not None
                    and sparse_spec.enabled
                    and sparse_spec.disable_full_load
                )
            )
            if skip_full_retrieve:
                if _sparse_kv_debug_enabled():
                    logger.info(
                        "[req_id=%s] sparse production path enabled; skipping full "
                        "LMCache retrieve. KV load must be performed by the "
                        "SPARSE_SSD custom op / selected-load path. "
                        "disable_full_load=%s enable_sparse_attention=%s",
                        request.req_id,
                        bool(getattr(sparse_spec, "disable_full_load", False)),
                        bool(getattr(sparse_spec, "enable_sparse_attention", False)),
                    )
                continue

            if self.use_layerwise:
                if idx == last_idx:
                    sync = True
                else:
                    sync = False
                # NOTE(Jiayi): Perform blending before layerwise prefix caching
                if self.enable_blending:
                    # TODO(Jiayi): Need to make prefix caching and blending compatible
                    self.blender.blend(
                        tokens[:lmcache_cached_tokens],
                        token_mask[:lmcache_cached_tokens],
                        kvcaches=kvcaches,
                        slot_mapping=slot_mapping[:lmcache_cached_tokens],
                        vllm_cached_tokens=request.load_spec.vllm_cached_tokens,
                    )
                else:
                    layerwise_retriever = self.lmcache_engine.retrieve_layer(
                        tokens[:lmcache_cached_tokens],
                        token_mask[:lmcache_cached_tokens],
                        kvcaches=kvcaches,
                        slot_mapping=slot_mapping[:lmcache_cached_tokens],
                        vllm_cached_tokens=request.load_spec.vllm_cached_tokens,
                        sync=sync,
                    )
                    # NOTE: retrieve for two layers at the first layer
                    next(layerwise_retriever)
                    next(layerwise_retriever)
                    self.layerwise_retrievers.append(layerwise_retriever)
            else:
                if _sparse_kv_debug_enabled():
                    logger.info(
                        "[lmcache-kv-iface] retrieve begin req_id=%s tokens=%d "
                        "mask_true=%d slot_mapping=%d vllm_cached=%d",
                        request.req_id,
                        len(tokens[:lmcache_cached_tokens]),
                        int(token_mask[:lmcache_cached_tokens].sum().item()),
                        int(slot_mapping[:lmcache_cached_tokens].numel()),
                        request.load_spec.vllm_cached_tokens,
                    )
                _retrieve_t0 = time.perf_counter()
                try:
                    ret_token_mask = self.lmcache_engine.retrieve(
                        tokens[:lmcache_cached_tokens],
                        token_mask[:lmcache_cached_tokens],
                        kvcaches=kvcaches,
                        slot_mapping=slot_mapping[:lmcache_cached_tokens],
                        vllm_cached_tokens=request.load_spec.vllm_cached_tokens,
                        request_configs=request.request_configs,
                        req_id=request.req_id,
                    )
                except Exception:
                    logger.exception(
                        "[lmcache-kv-iface] retrieve exception req_id=%s",
                        request.req_id,
                    )
                    raise
                _retrieve_ms = (time.perf_counter() - _retrieve_t0) * 1000.0

                # Check the result
                num_retrieved_tokens = ret_token_mask.sum().item()
                if _sparse_kv_debug_enabled():
                    logger.info(
                        "[lmcache-kv-iface] retrieve end req_id=%s "
                        "retrieved_tokens=%d expected_tokens=%d ret_mask_len=%d "
                        "duration_ms=%.3f",
                        request.req_id,
                        int(num_retrieved_tokens),
                        int(lmcache_cached_tokens - request.load_spec.vllm_cached_tokens),
                        int(ret_token_mask.numel()) if hasattr(ret_token_mask, "numel") else -1,
                        _retrieve_ms,
                    )
                num_expected_tokens = (
                    lmcache_cached_tokens - request.load_spec.vllm_cached_tokens
                )
                if num_retrieved_tokens < num_expected_tokens:
                    logger.error(
                        "Request %s"
                        "The number of retrieved tokens is less than the "
                        "expected number of tokens! This should not happen!",
                        request.req_id,
                    )
                    logger.error(
                        "Num retrieved tokens: %d, num expected tokens: %d",
                        num_retrieved_tokens,
                        num_expected_tokens,
                    )
                    """
                    Report failed block IDs in case of partial failure.
                    """
                    missing_blocks = self.record_failed_blocks(
                        request.req_id,
                        token_mask[:lmcache_cached_tokens],
                        ret_token_mask,
                        slot_mapping[:lmcache_cached_tokens],
                    )
                    self._invalid_block_ids.update(missing_blocks)

        # Publish graph-visible sparse metadata for this model step.  This must
        # happen in the KVConnector pre-forward path, after start_load_kv() has
        # built _sparse_runtime_requests and before the compiled attention graph
        # reads the persistent tensors.
        if getattr(self, "sparse_kv_spec", SparseKVSpec()).enabled:
            try:
                self.prepare_sparse_kv_step(forward_context=forward_context)
            except Exception:
                logger.exception("[sparse-kv-step] prepare failed in start_load_kv")
                raise

    def record_failed_blocks(
        self,
        request_id: str,
        expected_mask: torch.Tensor,
        ret_mask: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> set[int]:
        """Record block IDs associated with failed load attempts.

        Args:
            request_id: request id from vLLM.
            expected_mask: Boolean tensor indicating which tokens were expected to
                be loaded from LMCache. True means the token should be loaded,
                False means the token is already cached in vLLM and does not need
                to be loaded from LMCache.
            ret_mask: Boolean tensor indicating which tokens were actually
                successfully retrieved from LMCache. True means the token was
                successfully loaded. For example, if 256 tokens are expected to be
                loaded, but only 192 tokens are successfully loaded, then the
                ret_mask will be a tensor of 256 items like [T, T, ..., F, F, ...]
                where the first 192 elements are True and the last 64 elements
                are False.
            slot_mapping: Tensor indicating slot IDs for each token. The block
                ID is computed by dividing the slot ID by the block size.

        Example:
            expected_mask = [F, T, T, T] meaning the 1st is in vLLM cache
            ret_mask = [F, T, F, F] meaning failure from loading the 3rd
            missing_mask = expected_mask & ~ret_mask = [F, F, T, T]
            missing_indices = [2, 3]
            then missing_blocks is calculated from slot_mapping and missing_indices

        Returns:
            set[int]: Set of block IDs that failed to load.
        """

        if expected_mask.numel() == 0:
            return set()

        expected_mask_cpu = expected_mask.to(device="cpu", dtype=torch.bool)
        ret_mask_cpu = ret_mask.to(device="cpu", dtype=torch.bool)

        if ret_mask_cpu.shape[0] != expected_mask_cpu.shape[0]:
            logger.debug("expected_mask_cpu.shape[0] != ret_mask_cpu.shape[0]")
            return set()

        missing_mask = expected_mask_cpu & ~ret_mask_cpu
        if not torch.any(missing_mask):
            return set()

        missing_indices = torch.nonzero(missing_mask, as_tuple=False).view(-1)
        if missing_indices.numel() == 0:
            return set()

        slot_mapping_cpu = slot_mapping.to(device="cpu", dtype=torch.long)
        if slot_mapping_cpu.shape[0] > missing_mask.shape[0]:
            slot_mapping_cpu = slot_mapping_cpu[: missing_mask.shape[0]]

        missing_blocks_tensor = torch.unique(
            slot_mapping_cpu[missing_indices] // self._block_size
        )
        missing_blocks = {int(block.item()) for block in missing_blocks_tensor}

        if not missing_blocks:
            return set()

        logger.warning(
            "Request %s failed to load %d tokens across %d blocks",
            request_id,
            missing_indices.numel(),
            len(missing_blocks),
        )
        return missing_blocks



    def _sparse_runtime_device(self) -> torch.device:
        """Return the concrete device used by graph-visible sparse tensors.

        vLLM/LMCache may store self.device as "cuda" while allocated tensors
        report cuda:0.  Treat those as the same concrete device; otherwise the
        persistent context would be incorrectly rejected as "device changed".
        """
        device = torch.device(self.device)
        if device.type == "cuda" and device.index is None:
            try:
                return torch.device("cuda", torch.cuda.current_device())
            except Exception:
                return torch.device("cuda", 0)
        return device

    def _same_sparse_device(self, tensor_device: torch.device, target_device: torch.device) -> bool:
        tensor_device = torch.device(tensor_device)
        target_device = torch.device(target_device)
        if tensor_device.type != target_device.type:
            return False
        if tensor_device.type != "cuda":
            return tensor_device == target_device
        # Normalize "cuda" and "cuda:0" to the concrete current CUDA device.
        try:
            current = torch.cuda.current_device()
        except Exception:
            current = 0
        tensor_idx = current if tensor_device.index is None else int(tensor_device.index)
        target_idx = current if target_device.index is None else int(target_device.index)
        return tensor_idx == target_idx

    @staticmethod
    def _sparse_tensor_ptr(tensor: Any) -> str:
        if isinstance(tensor, torch.Tensor):
            try:
                return hex(int(tensor.data_ptr()))
            except Exception:
                return "unavailable"
        return "None"

    def _log_sparse_context_ptr(self, tag: str, ctx: Optional[dict[str, Any]]) -> None:
        if not _sparse_ctx_ptr_debug_enabled():
            return
        if not isinstance(ctx, dict):
            logger.info("[sparse-ctx-ptr][%s] ctx=None", tag)
            return
        logger.info(
            "[sparse-ctx-ptr][%s] ctx_id=%s generation=%s host_reqs=%s "
            "host_selected_blocks=%s active_reqs_ptr=%s req_token_lens_ptr=%s "
            "slot_mapping_table_ptr=%s selected_block_table_ptr=%s "
            "selected_block_lens_ptr=%s selected_ready_flags_ptr=%s "
            "debug_counters_ptr=%s fa_replay_marker_ptr=%s "
            "fa_block_table_ptr=%s fa_seq_lens_ptr=%s "
            "fa_query_start_loc_ptr=%s max_reqs=%s max_slots=%s max_selected_blocks=%s",
            tag,
            hex(id(ctx)),
            ctx.get("context_generation"),
            ctx.get("host_active_reqs"),
            ctx.get("host_selected_blocks"),
            self._sparse_tensor_ptr(ctx.get("active_reqs")),
            self._sparse_tensor_ptr(ctx.get("req_token_lens")),
            self._sparse_tensor_ptr(ctx.get("slot_mapping_table")),
            self._sparse_tensor_ptr(ctx.get("selected_block_table")),
            self._sparse_tensor_ptr(ctx.get("selected_block_lens")),
            self._sparse_tensor_ptr(ctx.get("selected_ready_flags")),
            self._sparse_tensor_ptr(ctx.get("debug_counters")),
            self._sparse_tensor_ptr(ctx.get("fa_replay_debug_marker")),
            self._sparse_tensor_ptr(ctx.get("fa_block_table")),
            self._sparse_tensor_ptr(ctx.get("fa_seq_lens")),
            self._sparse_tensor_ptr(ctx.get("fa_query_start_loc")),
            ctx.get("max_reqs"),
            ctx.get("max_slots"),
            ctx.get("max_selected_blocks"),
        )


    @staticmethod
    def _aissd_dtype_code(dtype_value: Any) -> int:
        s = str(dtype_value)
        if "bfloat16" in s or s == "BF16":
            return 3
        if "float16" in s or "half" in s or s == "F16":
            return 2
        if "float32" in s or s == "F32":
            return 1
        if "int8" in s or s == "I8":
            return 4
        raise RuntimeError(f"Unsupported AISSD sparse KV dtype={dtype_value!r}")

    @staticmethod
    def _aissd_fmt_code(fmt_value: Any) -> int:
        s = str(fmt_value)
        if "KV_2LTD" in s:
            return 1
        if "KV_T2D" in s:
            return 2
        if "KV_2TD" in s:
            return 3
        if "K_ONLY" in s:
            return 100
        # Conservative default for current LMCache vLLM layout: [L][NB,2,BS,NH,HS]
        # persisted chunks usually report KV_2TD/KV_T2D. Unknown layouts must not
        # silently succeed because SSD would decode wrong K.
        raise RuntimeError(f"Unsupported AISSD sparse KV fmt={fmt_value!r}")

    @staticmethod
    def _aissd_merge_extents(extents: list[tuple[int, int]], block_size: int) -> list[tuple[int, int]]:
        """Merge adjacent physical extents.

        FIEMAP may split a physically-contiguous range into multiple records due
        to filesystem bookkeeping.  The AISSD wire protocol has a bounded number
        of extents per candidate, so compact the representation before deciding
        a chunk is too fragmented for native-extent selection.
        """
        merged: list[tuple[int, int]] = []
        for lba, nbytes in extents:
            if nbytes <= 0:
                continue
            if merged:
                prev_lba, prev_bytes = merged[-1]
                prev_end_lba = prev_lba + ((prev_bytes + block_size - 1) // block_size)
                if prev_end_lba == lba:
                    merged[-1] = (prev_lba, prev_bytes + nbytes)
                    continue
            merged.append((int(lba), int(nbytes)))
        return merged

    @staticmethod
    def _aissd_fiemap_extents(
        path: str,
        file_offset: int,
        nbytes: int,
        block_size: int,
    ) -> tuple[list[tuple[int, int]], dict[str, int]]:
        """Return (compacted extents, stats) for a file byte range.

        AISSD_SPARSE_KV_EXTENT_MERGE controls whether physically contiguous
        FIEMAP records are merged before enforcing AISSD_SPARSE_KV_MAX_EXTENTS.
        Merge is metadata normalization, not fallback.

        AISSD_SPARSE_KV_FRAGMENT_POLICY is enforced by the caller.  This helper
        raises AISSDFiemapFragmentedError when the range still exceeds the
        protocol extent limit after optional merge.
        """
        FS_IOC_FIEMAP = 0xC020660B
        FIEMAP_FLAG_SYNC = 0x00000001
        FIEMAP_EXTENT_LAST = 0x00000001
        max_proto_extents = int(os.environ.get("AISSD_SPARSE_KV_MAX_EXTENTS", "64"))
        max_scan_extents = int(os.environ.get("AISSD_SPARSE_KV_FIEMAP_SCAN_EXTENTS", "1024"))
        max_scan_extents = max(max_proto_extents, max_scan_extents)
        merge_enabled = _aissd_extent_merge_enabled()

        header_size = 32
        extent_size = 56
        buf = bytearray(header_size + max_scan_extents * extent_size)
        struct.pack_into(
            "QQIIII",
            buf,
            0,
            int(file_offset),
            int(nbytes),
            FIEMAP_FLAG_SYNC,
            0,
            max_scan_extents,
            0,
        )
        fd = os.open(path, os.O_RDONLY)
        try:
            fcntl.ioctl(fd, FS_IOC_FIEMAP, buf, True)
        finally:
            os.close(fd)

        # struct fiemap layout (linux/fiemap.h):
        #   __u64 fm_start;           // offset 0
        #   __u64 fm_length;          // offset 8
        #   __u32 fm_flags;           // offset 16
        #   __u32 fm_mapped_extents;  // offset 20
        #   __u32 fm_extent_count;    // offset 24
        #   __u32 fm_reserved;        // offset 28
        # The previous code accidentally read offset 16, i.e. fm_flags.
        # With FIEMAP_FLAG_SYNC=1 this made mapped look like 1 forever,
        # so only the first extent was parsed and long ranges falsely failed
        # with "spans more than host scan limit".
        mapped = int(struct.unpack_from("I", buf, 20)[0])
        if mapped <= 0:
            raise RuntimeError(f"FIEMAP returned no extents for {path}:{file_offset}+{nbytes}")

        raw: list[tuple[int, int]] = []
        requested_end = int(file_offset) + int(nbytes)
        saw_last = False
        covered_end = int(file_offset)
        for i in range(min(mapped, max_scan_extents)):
            off = header_size + i * extent_size
            logical, physical, length = struct.unpack_from("QQQ", buf, off)
            flags = struct.unpack_from("I", buf, off + 40)[0]
            logical = int(logical)
            physical = int(physical)
            length = int(length)
            if length <= 0:
                continue
            overlap_start = max(logical, int(file_offset))
            overlap_end = min(logical + length, requested_end)
            if overlap_end <= overlap_start:
                continue
            physical_start = physical + (overlap_start - logical)
            take = overlap_end - overlap_start
            if physical_start % block_size != 0:
                raise RuntimeError(f"FIEMAP physical address is not block aligned: {physical_start}")
            raw.append((int(physical_start // block_size), int(take)))
            covered_end = max(covered_end, overlap_end)
            if flags & FIEMAP_EXTENT_LAST:
                saw_last = True
            if covered_end >= requested_end:
                break

        if merge_enabled:
            compacted = LMCacheConnectorV1Impl._aissd_merge_extents(raw, block_size)
        else:
            compacted = raw

        stats = {
            "raw_extents": int(len(raw)),
            "compacted_extents": int(len(compacted)),
            "mapped_extents": int(mapped),
            "max_extents": int(max_proto_extents),
            "scan_extents": int(max_scan_extents),
            "merge_enabled": int(bool(merge_enabled)),
        }

        if covered_end < requested_end and not saw_last:
            raise AISSDFiemapFragmentedError(
                f"FIEMAP range spans more than host scan limit {max_scan_extents} records for {path}",
                path=path,
                file_offset=file_offset,
                nbytes=nbytes,
                raw_extents=len(raw),
                merged_extents=len(compacted),
                max_extents=max_proto_extents,
                scan_extents=max_scan_extents,
            )

        if len(compacted) > max_proto_extents:
            raise AISSDFiemapFragmentedError(
                f"FIEMAP range has {len(compacted)} compacted extents "
                f"(raw={len(raw)}, merge={int(bool(merge_enabled))}); "
                f"protocol max is {max_proto_extents} for {path}",
                path=path,
                file_offset=file_offset,
                nbytes=nbytes,
                raw_extents=len(raw),
                merged_extents=len(compacted),
                max_extents=max_proto_extents,
                scan_extents=max_scan_extents,
            )
        return compacted, stats

    def _record_aissd_extent_stats(
        self,
        *,
        req_id: str,
        chunk_index: int,
        path: str,
        file_offset: int,
        nbytes: int,
        raw_extents: int,
        compacted_extents: int,
        max_extents: int,
        skipped: bool = False,
        failed: bool = False,
    ) -> None:
        if not _aissd_extent_stats_enabled():
            return
        samples = getattr(self, "_aissd_extent_count_samples", None)
        if samples is None:
            samples = []
            self._aissd_extent_count_samples = samples
        samples.append(int(compacted_extents))
        # Bound memory while still giving enough recent samples for P99.
        max_samples = int(os.environ.get("AISSD_SPARSE_KV_EXTENT_STATS_MAX_SAMPLES", "100000"))
        if len(samples) > max_samples:
            del samples[: len(samples) - max_samples]

        interval = max(1, int(os.environ.get("AISSD_SPARSE_KV_EXTENT_STATS_INTERVAL", "256")))
        should_log = failed or skipped or (len(samples) % interval == 0)
        if not should_log:
            return

        vals = sorted(samples)
        n = len(vals)

        def pct(p: int) -> int:
            if n <= 0:
                return 0
            idx = min(n - 1, int((p / 100.0) * (n - 1)))
            return int(vals[idx])

        logger.info(
            "[aissd-fiemap-stats] samples=%d p50=%d p90=%d p95=%d p99=%d max=%d "
            "last_req=%s last_chunk=%s last_raw=%d last_compacted=%d max_extents=%d "
            "skipped=%s failed=%s path=%s offset=%d nbytes=%d",
            n,
            pct(50),
            pct(90),
            pct(95),
            pct(99),
            int(vals[-1]) if vals else 0,
            req_id,
            chunk_index,
            int(raw_extents),
            int(compacted_extents),
            int(max_extents),
            bool(skipped),
            bool(failed),
            path,
            int(file_offset),
            int(nbytes),
        )

    def _build_aissd_candidate_tensors_for_step(
        self,
        req_ids: list[str],
        runtime_items: list[dict[str, Any]],
        max_candidates: int,
        blocks_per_chunk: int,
    ) -> dict[str, torch.Tensor]:
        """Build CPU native-extent candidate tensors for the AISSD selector op.

        This runs in LMCache/vLLM pre_forward/start_load_kv path, before model
        graph execution and before Q is available.  It does not select chunks;
        it only describes candidate native LMCache files/extents.  The compiled
        AISSD selector op consumes these tensors together with the real per-layer
        Q tensor and updates selected_block_table in-place.
        """
        req_n = len(req_ids)
        max_extents = int(os.environ.get("AISSD_SPARSE_KV_MAX_EXTENTS", "64"))
        max_dims = 8
        candidate_count = torch.zeros(req_n, dtype=torch.int32, device="cpu")
        chunk_ids = torch.full((req_n, max_candidates), -1, dtype=torch.int32, device="cpu")
        block_ids = torch.full((req_n, max_candidates, blocks_per_chunk), -1, dtype=torch.int32, device="cpu")
        block_lens = torch.zeros((req_n, max_candidates), dtype=torch.int32, device="cpu")
        token_start = torch.zeros((req_n, max_candidates), dtype=torch.int32, device="cpu")
        token_end = torch.zeros((req_n, max_candidates), dtype=torch.int32, device="cpu")
        dtype = torch.zeros((req_n, max_candidates), dtype=torch.int32, device="cpu")
        fmt = torch.zeros((req_n, max_candidates), dtype=torch.int32, device="cpu")
        ndim = torch.zeros((req_n, max_candidates), dtype=torch.int32, device="cpu")
        shape = torch.zeros((req_n, max_candidates, max_dims), dtype=torch.int64, device="cpu")
        extent_count = torch.zeros((req_n, max_candidates), dtype=torch.int32, device="cpu")
        extent_lba = torch.zeros((req_n, max_candidates, max_extents), dtype=torch.int64, device="cpu")
        extent_bytes = torch.zeros((req_n, max_candidates, max_extents), dtype=torch.int64, device="cpu")
        raw_block_size = int(os.environ.get("AISSD_SPARSE_KV_MANIFEST_BLOCK_SIZE", "4096"))

        for r, runtime in enumerate(runtime_items):
            tokens = runtime.get("tokens", [])
            token_mask = runtime.get("token_mask", None)
            slot_mapping = runtime.get("slot_mapping", None)
            request_configs = runtime.get("request_configs")
            if not tokens or token_mask is None or slot_mapping is None:
                continue
            slot_cpu = slot_mapping.detach().to("cpu") if isinstance(slot_mapping, torch.Tensor) else slot_mapping
            mask_cpu = token_mask.detach().to("cpu") if isinstance(token_mask, torch.Tensor) else token_mask
            manifest = self.lmcache_engine.build_sparse_kv_candidate_manifest(
                tokens=tokens,
                mask=mask_cpu,
                request_configs=request_configs,
                req_id=req_ids[r],
                layer_name=None,
                slot_mapping=slot_cpu,
                chunk_size=self._lmcache_chunk_size,
            )
            chunks = list(manifest.get("chunks", []))

            # The current SSD-NPU qK npubin/protocol path is compiled for at
            # most 128 real candidates.  Keep the persistent HOST tensor
            # capacity (max_candidates, normally 256) stable for graph/context
            # lifetime, but cap the actual candidate_count sent to SSD.
            #
            # Truncation policy:
            #   1) If candidate chunks carry a score-like field, keep top cap
            #      by score, then restore the original logical order.
            #   2) Otherwise keep the most recent tail chunks.  For decode,
            #      the recent context is usually the safest no-score fallback.
            npu_candidate_cap = int(os.environ.get("AISSD_SPARSE_KV_NPU_CANDIDATE_CAP", "128"))
            npu_candidate_cap = max(1, min(int(npu_candidate_cap), int(max_candidates)))
            original_candidate_count = len(chunks)
            if original_candidate_count > npu_candidate_cap:
                score_keys = (
                    "score",
                    "candidate_score",
                    "lookup_score",
                    "priority",
                    "rank_score",
                )
                scored: list[tuple[int, float, dict[str, Any]]] = []
                has_score = False
                for i, ch in enumerate(chunks):
                    score_val = None
                    for key in score_keys:
                        if key in ch and ch.get(key) is not None:
                            score_val = ch.get(key)
                            break
                    if score_val is not None:
                        try:
                            scored.append((i, float(score_val), ch))
                            has_score = True
                            continue
                        except (TypeError, ValueError):
                            pass
                    scored.append((i, float("-inf"), ch))

                if has_score:
                    selected = sorted(scored, key=lambda x: (x[1], x[0]), reverse=True)[:npu_candidate_cap]
                    selected.sort(key=lambda x: x[0])
                    chunks = [ch for _, _, ch in selected]
                    cap_strategy = "score_top"
                else:
                    chunks = chunks[-npu_candidate_cap:]
                    cap_strategy = "recent_tail"

                logger.warning(
                    "[warning][aissd-selector-host-cap] req=%s layer=step "
                    "original_candidates=%d capped_candidates=%d dropped=%d "
                    "strategy=%s tensor_capacity=%d risk=sparse-KV recall may "
                    "drop; consider multi-batch npubin up to c256 or a better "
                    "candidate ranking policy",
                    req_ids[r],
                    original_candidate_count,
                    len(chunks),
                    original_candidate_count - len(chunks),
                    cap_strategy,
                    int(max_candidates),
                )

            kept = 0
            skipped_fragmented = 0
            for src_c, chunk in enumerate(chunks):
                if kept >= max_candidates:
                    break
                c = kept
                # IMPORTANT: the SSD native-extents protocol treats
                # candidate.chunk_index as a candidate-local ordinal and
                # sparse_kv_runner validates chunk_index == candidate array
                # position.  After HOST-side truncation (for example keeping
                # recent_tail chunks 8..135), preserving the original manifest
                # chunk_index would make SSD return -14 before qK/NPU runs.
                # Keep token_start/token_end and block metadata from the real
                # chunk, but send a compact local index 0..candidate_count-1.
                chunk_ids[r, c] = int(c)
                ts = int(chunk.get("token_start", 0))
                te = int(chunk.get("token_end", ts + self._lmcache_chunk_size))
                token_start[r, c] = ts
                token_end[r, c] = te
                dtype[r, c] = self._aissd_dtype_code(chunk.get("dtype"))
                fmt[r, c] = self._aissd_fmt_code(chunk.get("fmt"))
                shp = list(chunk.get("shape") or [])[:max_dims]
                ndim[r, c] = len(shp)
                for d, val in enumerate(shp):
                    shape[r, c, d] = int(val)
                # Convert this chunk's slot range into vLLM paged block IDs.
                ss = int(chunk.get("slot_start", ts))
                se = int(chunk.get("slot_end", te))
                block_start = max(0, ss // int(self._block_size))
                block_end = max(block_start, (max(ss, se - 1) // int(self._block_size)) + 1)
                local_blocks = list(range(block_start, min(block_end, block_start + blocks_per_chunk)))
                block_lens[r, c] = len(local_blocks)
                for b, bid in enumerate(local_blocks):
                    block_ids[r, c, b] = int(bid)
                path = str(chunk.get("path"))
                file_offset = int(chunk.get("file_offset", 4096))
                nbytes = int(chunk.get("nbytes", 0))
                if not path or nbytes <= 0:
                    raise RuntimeError(f"Invalid AISSD candidate chunk path/nbytes: {chunk}")
                try:
                    exts, extent_stats = self._aissd_fiemap_extents(
                        path, file_offset, nbytes, raw_block_size
                    )
                    self._record_aissd_extent_stats(
                        req_id=req_ids[r],
                        chunk_index=int(chunk.get("chunk_index", src_c)),
                        path=path,
                        file_offset=file_offset,
                        nbytes=nbytes,
                        raw_extents=int(extent_stats.get("raw_extents", len(exts))),
                        compacted_extents=int(extent_stats.get("compacted_extents", len(exts))),
                        max_extents=max_extents,
                    )
                except AISSDFiemapFragmentedError as exc:
                    policy = _aissd_fragment_policy()
                    self._record_aissd_extent_stats(
                        req_id=req_ids[r],
                        chunk_index=int(chunk.get("chunk_index", src_c)),
                        path=path,
                        file_offset=file_offset,
                        nbytes=nbytes,
                        raw_extents=int(getattr(exc, "raw_extents", 0)),
                        compacted_extents=int(getattr(exc, "merged_extents", 0)),
                        max_extents=max_extents,
                        skipped=(policy == "skip"),
                        failed=(policy == "fail"),
                    )
                    if policy == "fail":
                        raise RuntimeError(
                            "AISSD native-extent candidate is too fragmented "
                            f"after merge; policy=fail req={req_ids[r]} "
                            f"chunk={chunk.get('chunk_index', src_c)} "
                            f"raw_extents={getattr(exc, 'raw_extents', 0)} "
                            f"compacted_extents={getattr(exc, 'merged_extents', 0)} "
                            f"max_extents={max_extents} path={path} "
                            f"offset={file_offset} nbytes={nbytes}"
                        ) from exc

                    # policy=skip: A single fragmented LMCache file range should
                    # not abort the whole request. Skip just this candidate and
                    # let AISSD select among the remaining native-readable chunks.
                    skipped_fragmented += 1
                    if _sparse_kv_debug_enabled() or _aissd_extent_stats_enabled():
                        logger.warning(
                            "[aissd-native-extents] skip fragmented candidate "
                            "req=%s chunk=%s raw_extents=%s compacted_extents=%s "
                            "max_extents=%s path=%s offset=%s nbytes=%s",
                            req_ids[r],
                            chunk.get("chunk_index", src_c),
                            getattr(exc, "raw_extents", 0),
                            getattr(exc, "merged_extents", 0),
                            max_extents,
                            path,
                            file_offset,
                            nbytes,
                        )
                    # Clear the partially-filled slot for this candidate.
                    chunk_ids[r, c] = -1
                    block_ids[r, c].fill_(-1)
                    block_lens[r, c] = 0
                    token_start[r, c] = 0
                    token_end[r, c] = 0
                    dtype[r, c] = 0
                    fmt[r, c] = 0
                    ndim[r, c] = 0
                    shape[r, c].zero_()
                    continue
                extent_count[r, c] = len(exts)
                for e, (lba, n) in enumerate(exts):
                    extent_lba[r, c, e] = int(lba)
                    extent_bytes[r, c, e] = int(n)
                kept += 1
            candidate_count[r] = kept
            if kept == 0 and chunks:
                raise RuntimeError(
                    f"All AISSD native-extent candidates were skipped for req={req_ids[r]} "
                    f"because LMCache files are too fragmented; skipped={skipped_fragmented}. "
                    "Defragment/rewrite the GDS cache directory or set AISSD_SPARSE_KV_FRAGMENT_POLICY=fail to debug the first fragmented candidate."
                )
            if skipped_fragmented and _sparse_kv_debug_enabled():
                logger.warning(
                    "[aissd-native-extents] req=%s kept=%d skipped_fragmented=%d",
                    req_ids[r], kept, skipped_fragmented,
                )

        return {
            "aissd_candidate_count": candidate_count,
            "aissd_candidate_chunk_ids": chunk_ids,
            "aissd_candidate_block_ids": block_ids,
            "aissd_candidate_block_lens": block_lens,
            "aissd_candidate_token_start": token_start,
            "aissd_candidate_token_end": token_end,
            "aissd_candidate_dtype": dtype,
            "aissd_candidate_fmt": fmt,
            "aissd_candidate_ndim": ndim,
            "aissd_candidate_shape": shape,
            "aissd_candidate_extent_count": extent_count,
            "aissd_candidate_extent_lba": extent_lba,
            "aissd_candidate_extent_bytes": extent_bytes,
        }

    def _sparse_kv_dtype_nbytes(self) -> int:
        """Return bytes/element for KV cache tensors, best-effort."""
        try:
            if self.kv_caches:
                t = next(iter(self.kv_caches.values()))
                if isinstance(t, torch.Tensor):
                    return int(t.element_size())
        except Exception:
            pass
        return 2

    def _sparse_kv_bytes_per_token_all_layers(self) -> int:
        """Bytes for one token of K+V across all layers in LMCache chunks."""
        num_layers = max(1, int(getattr(self, "num_layers", 1) or 1))
        num_kv_heads = max(1, int(getattr(self, "_sparse_num_kv_heads", 0) or 0))
        head_size = max(1, int(getattr(self, "_sparse_head_size", 0) or 0))
        elem_bytes = max(1, int(self._sparse_kv_dtype_nbytes()))
        return int(num_layers * 2 * num_kv_heads * head_size * elem_bytes)

    def _sparse_estimated_selected_kv_bytes(self, selected_tokens: int) -> int:
        return int(max(0, int(selected_tokens)) * self._sparse_kv_bytes_per_token_all_layers())

    def _ensure_sparse_step_buffers(
        self,
        max_reqs: int = 1,
        max_slots: int = 1,
        max_selected_blocks: int = 1,
    ) -> dict[str, Any]:
        """Return persistent device tensors for graph-compatible sparse KV state.

        CUDA graph replay can only see the same device tensor addresses that were
        present during capture.  Therefore this function allocates once and then
        refuses real capacity growth.  Runtime steps update the same tensors
        in-place; get_sparse_kv_step_context() never clears or replaces them.
        """
        spec = getattr(self, "sparse_kv_spec", SparseKVSpec())
        device = self._sparse_runtime_device()
        max_reqs = max(1, int(max_reqs))
        max_slots = max(1, int(max_slots))
        max_selected_blocks = max(1, int(max_selected_blocks))
        ctx = getattr(self, "_sparse_persistent_step_context", None)

        if ctx is not None:
            existing_device = ctx["req_token_lens"].device
            if not self._same_sparse_device(existing_device, device):
                raise RuntimeError(
                    "Sparse persistent context device changed; refusing to replace "
                    "CUDA-graph-visible tensors. "
                    f"existing={existing_device}, requested={device}, self.device={self.device}"
                )
            if int(ctx.get("max_reqs", 0)) < max_reqs:
                raise RuntimeError(
                    "Sparse persistent context max_reqs is too small; refusing to "
                    "replace CUDA-graph-visible tensors. "
                    f"existing={ctx.get('max_reqs')}, requested={max_reqs}. "
                    "Increase --max-num-seqs before engine start."
                )
            if int(ctx.get("max_slots", 0)) < max_slots:
                raise RuntimeError(
                    "Sparse persistent context max_slots is too small; refusing to "
                    "replace CUDA-graph-visible tensors. "
                    f"existing={ctx.get('max_slots')}, requested={max_slots}. "
                    "Increase --max-model-len/--max-num-batched-tokens before engine start."
                )
            if int(ctx.get("max_selected_blocks", 0)) < max_selected_blocks:
                raise RuntimeError(
                    "Sparse persistent context max_selected_blocks is too small; refusing to "
                    "replace CUDA-graph-visible tensors. "
                    f"existing={ctx.get('max_selected_blocks')}, requested={max_selected_blocks}. "
                    "Increase lmcache.sparse_kv_top_n_chunks before engine start."
                )
            return ctx

        ctx = {
            "req_ids": [],
            "active_reqs": torch.zeros((), dtype=torch.int32, device=device),
            "req_token_lens": torch.zeros(max_reqs, dtype=torch.int32, device=device),
            "req_vllm_cached_tokens": torch.zeros(max_reqs, dtype=torch.int32, device=device),
            "req_lmcache_cached_tokens": torch.zeros(max_reqs, dtype=torch.int32, device=device),
            "req_slot_lens": torch.zeros(max_reqs, dtype=torch.int32, device=device),
            "slot_mapping_table": torch.full((max_reqs, max_slots), -1, dtype=torch.long, device=device),
            "selected_block_table": torch.full((max_reqs, max_selected_blocks), -1, dtype=torch.int32, device=device),
            "selected_block_lens": torch.zeros(max_reqs, dtype=torch.int32, device=device),
            "selected_ready_flags": torch.zeros(max_reqs, dtype=torch.int32, device=device),
            # FlashAttention-varlen sparse path metadata.  These tensors are also
            # graph-visible and are updated in-place by prepare_sparse_kv_step().
            # Inactive padded rows use block 0 with seqlen 1; their outputs are
            # ignored by vLLM, while active rows are overwritten with selected
            # sparse block tables and selected token lengths.
            "fa_block_table": torch.zeros((max_reqs, max_selected_blocks), dtype=torch.int32, device=device),
            "fa_seq_lens": torch.ones(max_reqs, dtype=torch.int32, device=device),
            "fa_query_start_loc": torch.arange(max_reqs + 1, dtype=torch.int32, device=device),
            "fa_max_seq_len": int(max_selected_blocks) * int(self._block_size),
            # Captured by sparse_flash_attention; prepare_sparse_kv_step drains it.
            "debug_counters": torch.zeros(8, dtype=torch.long, device=device),
            # Captured by FA-varlen sparse path when VLLM_SPARSE_FA_REPLAY_DEBUG=1.
            # The attention backend writes a few metadata values into this marker
            # on graph replay; prepare_sparse_kv_step drains it before publishing
            # the next step.  It is tiny and always allocated to avoid CUDA graph
            # capture-time allocation.
            "fa_replay_debug_marker": torch.zeros(8, dtype=torch.int32, device=device),
            # AISSD selector metadata is CPU-side native extent metadata built in
            # prepare_sparse_kv_step(); the C++ op consumes it with the real Q.
            "aissd_selector_backend": str(getattr(spec, "sparse_kv_backend", "host")),
            "aissd_top_m": int(os.environ.get("AISSD_SPARSE_KV_TOP_M", "8")),
            "aissd_score_mode_code": 1 if str(getattr(spec, "score_mode", "topm_mean")) == "topm_mean" else 2,
            "aissd_manifest_block_size": int(os.environ.get("AISSD_SPARSE_KV_MANIFEST_BLOCK_SIZE", "4096")),
            "aissd_timeout_ms": int(os.environ.get("AISSD_SPARSE_KV_TIMEOUT_MS", "300000")),
            # Empty CPU-side AISSD candidate tensors for bootstrap/CUDA graph
            # capture. Real request steps replace these with tensors built from
            # native LMCache file extents. Keeping the keys present lets the
            # sparse attention backend distinguish "bootstrap/no active request"
            # from a real AISSD metadata construction bug.
            "aissd_candidate_count": torch.zeros(max_reqs, dtype=torch.int32, device="cpu"),
            "aissd_candidate_chunk_ids": torch.full((max_reqs, 1), -1, dtype=torch.int32, device="cpu"),
            "aissd_candidate_block_ids": torch.full((max_reqs, 1, 1), -1, dtype=torch.int32, device="cpu"),
            "aissd_candidate_block_lens": torch.zeros((max_reqs, 1), dtype=torch.int32, device="cpu"),
            "aissd_candidate_token_start": torch.zeros((max_reqs, 1), dtype=torch.int32, device="cpu"),
            "aissd_candidate_token_end": torch.zeros((max_reqs, 1), dtype=torch.int32, device="cpu"),
            "aissd_candidate_dtype": torch.zeros((max_reqs, 1), dtype=torch.int32, device="cpu"),
            "aissd_candidate_fmt": torch.zeros((max_reqs, 1), dtype=torch.int32, device="cpu"),
            "aissd_candidate_ndim": torch.zeros((max_reqs, 1), dtype=torch.int32, device="cpu"),
            "aissd_candidate_shape": torch.zeros((max_reqs, 1, 8), dtype=torch.int64, device="cpu"),
            "aissd_candidate_extent_count": torch.zeros((max_reqs, 1), dtype=torch.int32, device="cpu"),
            "aissd_candidate_extent_lba": torch.zeros((max_reqs, 1, int(os.environ.get("AISSD_SPARSE_KV_MAX_EXTENTS", "64"))), dtype=torch.int64, device="cpu"),
            "aissd_candidate_extent_bytes": torch.zeros((max_reqs, 1, int(os.environ.get("AISSD_SPARSE_KV_MAX_EXTENTS", "64"))), dtype=torch.int64, device="cpu"),
            "max_reqs": max_reqs,
            "max_slots": max_slots,
            "max_selected_blocks": max_selected_blocks,
            "block_size": int(self._block_size),
            "chunk_size": int(self._lmcache_chunk_size),
            "top_n_chunks": int(getattr(spec, "top_n_chunks", 0)),
            "score_mode": str(getattr(spec, "score_mode", "topm_mean")),
            "disable_full_load": bool(getattr(spec, "disable_full_load", False)),
            "host_active_reqs": 0,
            "host_selected_blocks": 0,
            "sparse_selected_blocks": 0,
            "sparse_selected_tokens": 0,
            "sparse_selected_kv_bytes": 0,
            "sparse_selected_kv_bytes_source": "none",
            "sparse_selected_load_ms": 0.0,
            "sparse_selected_load_bytes": 0,
            "sparse_selected_loaded_chunks": 0,
            "context_generation": 0,
        }
        self._sparse_persistent_step_context = ctx
        self._sparse_current_step_context = ctx
        logger.info(
            "[sparse-kv-step] allocated persistent buffers max_reqs=%d "
            "max_slots=%d max_selected_blocks=%d block_size=%d chunk_size=%d "
            "device=%s self_device=%s",
            max_reqs,
            max_slots,
            max_selected_blocks,
            int(self._block_size),
            int(self._lmcache_chunk_size),
            device,
            self.device,
        )
        self._log_sparse_context_ptr("alloc", ctx)
        return ctx

    def get_sparse_kv_step_context(self, create_if_missing: bool = True) -> Optional[dict[str, Any]]:
        """Return stable sparse step tensors for SparseSSDAttentionImpl.

        During CUDA graph capture there may be no real requests yet.  In that
        case we still return an empty persistent context so the graph captures
        the production sparse custom-op call with stable tensor addresses.
        """
        ctx = getattr(self, "_sparse_current_step_context", None)
        if ctx is not None:
            return ctx
        if not create_if_missing:
            return None
        spec = getattr(self, "sparse_kv_spec", SparseKVSpec())
        if not getattr(spec, "enabled", False):
            return None
        ctx = self._ensure_sparse_step_buffers(
            getattr(self, "_sparse_max_reqs", 1),
            getattr(self, "_sparse_max_slots", 1),
            getattr(self, "_sparse_max_selected_blocks", 1),
        )
        # Do not clear graph-visible tensors in this accessor.  Attention.forward
        # may call it during CUDA graph capture; any zero_()/fill_() here would
        # be captured and replayed before sparse_flash_attention, wiping the
        # runtime active context prepared by KVConnector.pre_forward().  Only
        # prepare_sparse_kv_step() is allowed to publish/clear step metadata.
        self._sparse_current_step_context = ctx
        self._log_sparse_context_ptr("get", ctx)
        return ctx

    def _build_sparse_selected_blocks_from_slots(
        self,
        slot_mapping: torch.Tensor,
        vllm_cached_tokens: int,
        lmcache_cached_tokens: int,
        top_n_chunks: int,
    ) -> list[int]:
        """Build a compact selected block list for the production custom op.

        The actual Q-aware ranking is intended to move into SSD-CPU/NPU or the
        sparse attention op once per-chunk summaries are available.  This
        function only turns a selected token/chunk range into physical vLLM
        paged-cache block ids and keeps the graph-visible metadata stable.
        It selects the most recent top-N LMCache chunks in the load window,
        which is a useful production-safe default for decode while avoiding
        full-context block tables.
        """
        if not isinstance(slot_mapping, torch.Tensor) or slot_mapping.numel() == 0:
            return []
        start_token = max(0, int(vllm_cached_tokens))
        end_token = max(start_token, min(int(lmcache_cached_tokens), int(slot_mapping.numel())))
        if end_token <= start_token:
            return []

        chunk_size = max(1, int(self._lmcache_chunk_size))
        first_chunk = start_token // chunk_size
        last_chunk = (end_token - 1) // chunk_size
        chunk_ids = list(range(first_chunk, last_chunk + 1))
        if top_n_chunks and top_n_chunks > 0:
            # Prefer recent chunks for decode/prefix reuse workloads.  This is a
            # deterministic policy, not a debug fallback; Q-aware ranking can
            # replace the chunk_ids selection without changing the downstream
            # metadata/op ABI.
            chunk_ids = chunk_ids[-int(top_n_chunks):]

        selected_blocks: list[int] = []
        seen: set[int] = set()
        slot_cpu = slot_mapping.detach().to("cpu", dtype=torch.long)
        for chunk_id in chunk_ids:
            t0 = max(start_token, chunk_id * chunk_size)
            t1 = min(end_token, (chunk_id + 1) * chunk_size)
            if t1 <= t0:
                continue
            slots = slot_cpu[t0:t1]
            for block_id in (slots // int(self._block_size)).tolist():
                block_id = int(block_id)
                if block_id >= 0 and block_id not in seen:
                    seen.add(block_id)
                    selected_blocks.append(block_id)
        return selected_blocks


    def _sparse_debug_counters_enabled(self) -> bool:
        return _sparse_attn_debug_counters_enabled()

    def _sparse_fa_replay_debug_enabled(self) -> bool:
        return _sparse_fa_replay_debug_enabled()

    def _drain_sparse_fa_replay_marker(
        self,
        step_context: Optional[dict[str, Any]],
        reason: str,
    ) -> None:
        """Log CUDA-graph replay marker from the FA-varlen sparse path.

        Python attention forward is not re-entered during graph replay, so this
        marker is written by graph-captured tensor copy ops in
        SparseSSDAttentionImpl._forward_sparse_fa_varlen().  Draining here
        proves whether replay saw runtime-updated fa_seq_lens/fa_block_table.
        """
        if not self._sparse_fa_replay_debug_enabled():
            return
        if not isinstance(step_context, dict):
            return
        marker = step_context.get("fa_replay_debug_marker")
        if not isinstance(marker, torch.Tensor) or marker.numel() < 8:
            return
        try:
            vals = [int(x) for x in marker.detach().cpu().tolist()[:8]]
        except Exception:
            logger.debug(
                "[sparse-attn-fa-replay] failed to read debug marker",
                exc_info=True,
            )
            return

        (
            fa_seq_len0,
            fa_block0,
            fa_block1,
            fa_query_start1,
            marker_active_reqs,
            marker_ready0,
            marker_q_tokens,
            marker_selected_len0,
        ) = vals

        # Ignore completely empty markers.  Dummy capture rows typically have
        # seq_len0=1/block0=0 and active_reqs=0; keep those suppressed so the
        # useful active replay evidence is easy to see.
        if (
            marker_active_reqs == 0
            and marker_selected_len0 == 0
            and fa_seq_len0 <= 1
            and fa_block0 == 0
            and fa_block1 == 0
        ):
            marker.zero_()
            return

        logger.info(
            "[sparse-attn-fa-replay] reason=%s generation=%s "
            "host_reqs=%s host_selected_blocks=%s fa_seq_len0=%d "
            "fa_block0=%d fa_block1=%d fa_query_start1=%d "
            "marker_active_reqs=%d marker_ready0=%d marker_q_tokens=%d "
            "marker_selected_len0=%d",
            reason,
            step_context.get("context_generation"),
            step_context.get("host_active_reqs"),
            step_context.get("host_selected_blocks"),
            fa_seq_len0,
            fa_block0,
            fa_block1,
            fa_query_start1,
            marker_active_reqs,
            marker_ready0,
            marker_q_tokens,
            marker_selected_len0,
        )
        marker.zero_()

    def _drain_sparse_debug_counters(
        self,
        step_context: Optional[dict[str, Any]],
        reason: str,
    ) -> None:
        """Log CUDA-graph replay sparse-attention counters from last step.

        When CUDA graphs are enabled, SparseSSDAttentionImpl.forward() is not
        re-entered on every replay, so Python-side per-layer logging only sees
        warmup/capture calls.  The sparse_flash_attention op therefore writes
        counters into a stable device tensor captured by the graph.  The KV
        connector runs before the next model step and can safely drain that
        tensor here, proving whether the replayed graph saw active_reqs and
        visited selected KV blocks/tokens.
        """
        if not self._sparse_debug_counters_enabled():
            return
        if not isinstance(step_context, dict):
            return
        counters = step_context.get("debug_counters")
        if not isinstance(counters, torch.Tensor) or counters.numel() < 8:
            return
        try:
            vals = [int(x) for x in counters.detach().cpu().tolist()[:8]]
        except Exception:
            logger.debug(
                "[sparse-attn-kernel-replay] failed to read debug counters",
                exc_info=True,
            )
            return

        (
            launched_qh_blocks,
            active_qh_blocks,
            selected_block_visits,
            selected_token_visits,
            inactive_qh_blocks,
            invalid_block_refs,
            kernel_active_reqs,
            kernel_max_selected_len,
        ) = vals

        # Avoid flooding logs with pure empty capture/warmup counters.  Active
        # replay evidence is active_qh_blocks>0 or selected KV visits>0.
        if (
            active_qh_blocks == 0
            and selected_block_visits == 0
            and selected_token_visits == 0
            and kernel_active_reqs == 0
        ):
            counters.zero_()
            return

        logger.info(
            "[sparse-attn-kernel-replay] reason=%s generation=%s "
            "host_reqs=%s host_selected_blocks=%s launched_qh_blocks=%d "
            "active_qh_blocks=%d inactive_qh_blocks=%d "
            "selected_block_visits=%d selected_token_visits=%d "
            "invalid_block_refs=%d kernel_active_reqs=%d "
            "kernel_max_selected_len=%d",
            reason,
            step_context.get("context_generation"),
            step_context.get("host_active_reqs"),
            step_context.get("host_selected_blocks"),
            launched_qh_blocks,
            active_qh_blocks,
            inactive_qh_blocks,
            selected_block_visits,
            selected_token_visits,
            invalid_block_refs,
            kernel_active_reqs,
            kernel_max_selected_len,
        )
        counters.zero_()

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def prepare_sparse_kv_step(
        self,
        forward_context: "ForwardContext",
        scheduler_output: Optional["SchedulerOutput"] = None,
        **kwargs: Any,
    ) -> Optional[dict[str, Any]]:
        """Publish production sparse-KV step context before model execution.

        This is deliberately a step-level hook called from vLLM's KV connector
        pre_forward path, after start_load_kv() has collected per-request
        token/slot/runtime information but before the model graph runs.  It does
        not perform Q-aware selection and it does not trigger IO.  A production
        SparseSSDAttention custom op can consume these stable tensors/handles
        together with the per-layer Q tensor inside the compiled attention path.

        This replaces the earlier Python attention-hook design, which was not
        production-safe because CUDA graph replay does not re-enter Python on
        every real request.
        """
        spec = getattr(self, "sparse_kv_spec", SparseKVSpec())
        if not getattr(spec, "enabled", False):
            return None
        t_prepare0 = time.perf_counter()
        t_candidate0 = 0.0
        t_candidate1 = 0.0

        # Drain counters written by the previous CUDA graph replay before we
        # publish/zero metadata for the next step.  This is the only reliable
        # Python-visible place to observe graph replay because the attention
        # backend forward() is not re-entered on replay.
        self._drain_sparse_fa_replay_marker(
            getattr(self, "_sparse_current_step_context", None),
            reason="before_prepare",
        )
        self._drain_sparse_debug_counters(
            getattr(self, "_sparse_current_step_context", None),
            reason="before_prepare",
        )
        self._log_sparse_context_ptr("prepare-entry", getattr(self, "_sparse_current_step_context", None))

        runtime_requests = getattr(self, "_sparse_runtime_requests", {}) or {}
        attn_metadata = getattr(forward_context, "attn_metadata", None)

        req_ids: list[str] = []
        token_lens: list[int] = []
        vllm_cached: list[int] = []
        lmcache_cached: list[int] = []
        slot_lens: list[int] = []
        max_slots = 0
        max_selected_blocks = 1
        slot_rows: list[torch.Tensor] = []
        selected_block_rows: list[list[int]] = []
        runtime_items: list[dict[str, Any]] = []

        for req_id, runtime in runtime_requests.items():
            slot_mapping = runtime.get("slot_mapping")
            if not isinstance(slot_mapping, torch.Tensor) or slot_mapping.numel() == 0:
                continue
            req_ids.append(str(req_id))
            runtime_items.append(runtime)
            token_lens.append(int(len(runtime.get("tokens", []) or [])))
            vllm_cached.append(int(runtime.get("vllm_cached_tokens", 0)))
            lmcache_cached.append(int(runtime.get("lmcache_cached_tokens", 0)))
            slot_lens.append(int(slot_mapping.numel()))
            max_slots = max(max_slots, int(slot_mapping.numel()))
            slot_row = slot_mapping.to(self.device, dtype=torch.long)
            slot_rows.append(slot_row)
            selected_blocks = self._build_sparse_selected_blocks_from_slots(
                slot_mapping=slot_mapping,
                vllm_cached_tokens=int(runtime.get("vllm_cached_tokens", 0)),
                lmcache_cached_tokens=int(runtime.get("lmcache_cached_tokens", 0)),
                top_n_chunks=int(getattr(runtime.get("sparse_spec", spec), "top_n_chunks", getattr(spec, "top_n_chunks", 0))),
            )
            selected_block_rows.append(selected_blocks)
            max_selected_blocks = max(max_selected_blocks, len(selected_blocks))

        if not req_ids:
            existing_context = getattr(self, "_sparse_current_step_context", None)
            if (
                getattr(self, "_sparse_active_context_pending", False)
                and existing_context is not None
                and int(existing_context.get("host_active_reqs", 0) or 0) > 0
            ):
                # A later graph-capture/small-step prepare with no runtime
                # requests must not overwrite the active context that was just
                # published by start_load_kv().  Otherwise Attention.forward
                # observes reqs=0 even though start_load_kv logged
                # prepared reqs>0 / selected_blocks>0.
                step_context = existing_context
                if _sparse_kv_debug_enabled():
                    logger.info(
                        "[sparse-kv-step] no eligible runtime requests; keeping "
                        "active persistent context generation=%s reqs=%s "
                        "selected_blocks=%s runtime_requests=%d attn_metadata=%s",
                        step_context.get("context_generation"),
                        step_context.get("host_active_reqs"),
                        step_context.get("host_selected_blocks"),
                        len(runtime_requests),
                        type(attn_metadata).__name__ if attn_metadata is not None else "None",
                    )
            else:
                if _sparse_kv_debug_enabled():
                    logger.info(
                        "[sparse-kv-step] no eligible runtime requests; publishing empty "
                        "persistent context runtime_requests=%d attn_metadata=%s",
                        len(runtime_requests),
                        type(attn_metadata).__name__ if attn_metadata is not None else "None",
                    )
                step_context = self._ensure_sparse_step_buffers(
                    getattr(self, "_sparse_max_reqs", 1),
                    getattr(self, "_sparse_max_slots", 1),
                    getattr(self, "_sparse_max_selected_blocks", 1),
                )
                step_context["active_reqs"].zero_()
                step_context["req_ids"] = []
                step_context["req_token_lens"].zero_()
                step_context["req_vllm_cached_tokens"].zero_()
                step_context["req_lmcache_cached_tokens"].zero_()
                step_context["req_slot_lens"].zero_()
                step_context["slot_mapping_table"].fill_(-1)
                step_context["selected_block_table"].fill_(-1)
                step_context["selected_block_lens"].zero_()
                step_context["selected_ready_flags"].zero_()
                step_context["fa_block_table"].zero_()
                step_context["fa_seq_lens"].fill_(1)
                step_context["fa_query_start_loc"].copy_(
                    torch.arange(
                        int(step_context["fa_query_start_loc"].numel()),
                        dtype=torch.int32,
                        device=self.device,
                    )
                )
                step_context["host_active_reqs"] = 0
                step_context["host_selected_blocks"] = 0
                step_context["sparse_selected_blocks"] = 0
                step_context["sparse_selected_tokens"] = 0
                step_context["sparse_selected_kv_bytes"] = 0
                step_context["sparse_selected_kv_bytes_source"] = "none"
                step_context["sparse_selected_load_ms"] = 0.0
                step_context["sparse_selected_load_bytes"] = 0
                step_context["sparse_selected_loaded_chunks"] = 0
                self._sparse_active_context_pending = False
        else:
            step_context = self._ensure_sparse_step_buffers(
                max(len(req_ids), int(getattr(self, "_sparse_max_reqs", 1))),
                max(max_slots, int(getattr(self, "_sparse_max_slots", 1))),
                max(max_selected_blocks, int(getattr(self, "_sparse_max_selected_blocks", 1))),
            )
            step_context["active_reqs"].fill_(len(req_ids))
            step_context["req_ids"] = req_ids
            # Clear whole buffers first so stale rows from previous larger batches
            # cannot leak into the graph-captured custom op.
            step_context["req_token_lens"].zero_()
            step_context["req_vllm_cached_tokens"].zero_()
            step_context["req_lmcache_cached_tokens"].zero_()
            step_context["req_slot_lens"].zero_()
            step_context["slot_mapping_table"].fill_(-1)
            step_context["selected_block_table"].fill_(-1)
            step_context["selected_block_lens"].zero_()
            step_context["selected_ready_flags"].zero_()
            step_context["fa_block_table"].zero_()
            step_context["fa_seq_lens"].fill_(1)
            step_context["fa_query_start_loc"].copy_(
                torch.arange(
                    int(step_context["fa_query_start_loc"].numel()),
                    dtype=torch.int32,
                    device=self.device,
                )
            )
            n = len(req_ids)
            step_context["req_token_lens"][:n].copy_(
                torch.tensor(token_lens, dtype=torch.int32, device=self.device)
            )
            step_context["req_vllm_cached_tokens"][:n].copy_(
                torch.tensor(vllm_cached, dtype=torch.int32, device=self.device)
            )
            step_context["req_lmcache_cached_tokens"][:n].copy_(
                torch.tensor(lmcache_cached, dtype=torch.int32, device=self.device)
            )
            step_context["req_slot_lens"][:n].copy_(
                torch.tensor(slot_lens, dtype=torch.int32, device=self.device)
            )
            for row, slots in enumerate(slot_rows):
                step_context["slot_mapping_table"][row, : slots.numel()].copy_(slots)
            backend_name = str(getattr(spec, "sparse_kv_backend", "host"))
            step_context["aissd_selector_backend"] = backend_name
            if backend_name in ("ssd-cpu", "ssd-npu"):
                blocks_per_chunk = max(1, cdiv(max(1, int(self._lmcache_chunk_size)), max(1, int(self._block_size))))
                # Keep the persistent AISSD candidate tensor capacity stable.
                # The compiled SSD-NPU qK model currently accepts at most 128
                # real candidates, but the graph-visible HOST tensors may be
                # larger.  Do not derive capacity from max_slots, otherwise
                # different scheduling shapes (for example 136 chunks) change
                # tensor shapes and can perturb CUDA-graph/context behavior.
                max_candidates = max(1, int(os.environ.get("AISSD_SPARSE_KV_CANDIDATE_TENSOR_CAP", "256")))
                t_candidate0 = time.perf_counter()
                aissd_tensors = self._build_aissd_candidate_tensors_for_step(
                    req_ids=req_ids,
                    runtime_items=runtime_items,
                    max_candidates=max_candidates,
                    blocks_per_chunk=blocks_per_chunk,
                )
                t_candidate1 = time.perf_counter()
                step_context.update(aissd_tensors)
                step_context["aissd_selector_layer_reuse"] = bool(
                    _env_flag("AISSD_SPARSE_KV_LAYER_REUSE", "1")
                )
                step_context["aissd_selector_done_generation"] = -1
                step_context["aissd_selector_done_layer"] = ""
                # The AISSD selector op will fill selected_block_table/lens/ready
                # with q-aware results before FA-varlen consumes them.
                selected_blocks_sum = 0
            else:
                for row, block_ids in enumerate(selected_block_rows):
                    if not block_ids:
                        continue
                    count = min(len(block_ids), int(step_context["selected_block_table"].shape[1]))
                    block_tensor = torch.tensor(block_ids[:count], dtype=torch.int32, device=self.device)
                    step_context["selected_block_table"][row, :count].copy_(block_tensor)
                    step_context["selected_block_lens"][row] = count
                    # FlashAttention-varlen consumes a block_table + seqlen.  Keep
                    # selected blocks in ascending logical-token order so FA sees a
                    # compacted KV sequence with correct order.  count*block_size is
                    # exact for chunk-aligned selected blocks in the current path.
                    step_context["fa_block_table"][row, :count].copy_(block_tensor)
                    step_context["fa_seq_lens"][row] = int(count) * int(self._block_size)
                    # Host-side selected-load readiness is represented as a stable
                    # device flag.  A future async GDS path should clear this before
                    # launch and set it after DMA completion; current synchronous
                    # selected metadata marks it ready for the attention op.
                    step_context["selected_ready_flags"][row] = 1
                selected_blocks_sum = sum(min(len(x), int(step_context["selected_block_table"].shape[1])) for x in selected_block_rows)
            self._sparse_step_generation += 1
            self._sparse_active_context_pending = True
            step_context["host_active_reqs"] = int(n)
            step_context["host_selected_blocks"] = int(selected_blocks_sum)
            # Selected-KV bandwidth numerator.  For host-side selected metadata,
            # selected_blocks_sum is exact.  For AISSD selector mode the q-aware
            # op fills selected blocks later, so use a deterministic upper-bound
            # estimate: active_reqs * top_n_chunks * blocks_per_chunk.
            if backend_name in ("ssd-cpu", "ssd-npu"):
                est_blocks_per_req = max(1, int(getattr(spec, "top_n_chunks", 0) or 0)) * max(1, blocks_per_chunk)
                metric_selected_blocks = int(n) * int(est_blocks_per_req)
                bytes_source = "estimated_from_topn_blocks"
            else:
                metric_selected_blocks = int(selected_blocks_sum)
                bytes_source = "host_selected_blocks"
            metric_selected_tokens = int(metric_selected_blocks) * int(self._block_size)
            step_context["sparse_selected_blocks"] = int(metric_selected_blocks)
            step_context["sparse_selected_tokens"] = int(metric_selected_tokens)
            step_context["sparse_selected_kv_bytes"] = int(self._sparse_estimated_selected_kv_bytes(metric_selected_tokens))
            step_context["sparse_selected_kv_bytes_source"] = bytes_source
            # These are overwritten by the Python selected-load path when actual
            # selected GDS/LMCache load happens before attention.  AISSD q-aware
            # mode currently keeps this as 0 unless selected-load instrumentation
            # populates it.
            step_context["sparse_selected_load_ms"] = 0.0
            step_context["sparse_selected_load_bytes"] = 0
            step_context["sparse_selected_loaded_chunks"] = 0
            step_context["context_generation"] = int(self._sparse_step_generation)
            self._log_sparse_context_ptr("prepare-active", step_context)

        self._log_sparse_context_ptr("prepare-final", step_context)
        targets = [] if attn_metadata is None else [attn_metadata]
        if attn_metadata is not None:
            common_meta = getattr(attn_metadata, "common_metadata", None)
            if common_meta is not None and common_meta is not attn_metadata:
                targets.append(common_meta)
        for target in targets:
            try:
                setattr(target, "sparse_kv_step_context", step_context)
                setattr(target, "sparse_kv_step_ready", True)
            except Exception:
                logger.debug(
                    "[sparse-kv-step] failed to attach step context to %s",
                    type(target).__name__,
                    exc_info=True,
                )

        self._sparse_current_step_context = step_context
        if _sparse_kv_debug_enabled():
            logger.info(
                "[sparse-kv-step] prepared reqs=%d max_slots=%d block_size=%d "
                "chunk_size=%d top_n_chunks=%d selected_blocks=%d "
                "generation=%s host_reqs=%s disable_full_load=%s attn_metadata=%s",
                len(req_ids),
                max_slots,
                int(self._block_size),
                int(self._lmcache_chunk_size),
                int(getattr(spec, "top_n_chunks", 0)),
                int(step_context.get("host_selected_blocks", 0) or 0),
                step_context.get("context_generation"),
                step_context.get("host_active_reqs"),
                bool(getattr(spec, "disable_full_load", False)),
                type(attn_metadata).__name__ if attn_metadata is not None else "None",
            )
        if _aissd_selector_stats_enabled():
            t_prepare1 = time.perf_counter()
            candidate_ms = ((t_candidate1 - t_candidate0) * 1000.0) if t_candidate1 else 0.0
            candidate_count_obj = step_context.get("aissd_candidate_count")
            try:
                candidate_counts = (candidate_count_obj.detach().cpu().tolist()
                                    if isinstance(candidate_count_obj, torch.Tensor)
                                    else [])
            except Exception:
                candidate_counts = []
            logger.info(
                "[aissd-selector-host-prepare] generation=%s reqs=%d "
                "candidate_counts=%s layer_reuse=%s build_candidates_ms=%.3f "
                "prepare_total_ms=%.3f",
                step_context.get("context_generation"),
                len(req_ids),
                candidate_counts[:len(req_ids)] if candidate_counts else [],
                step_context.get("aissd_selector_layer_reuse"),
                candidate_ms,
                (t_prepare1 - t_prepare0) * 1000.0,
            )
        return step_context

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def sparse_select_kv_layer(
        self,
        layer_name: str,
        query: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> Optional[dict[str, Any]]:
        """Build q + candidate chunk manifests for SSD-CPU/NPU selection.

        This is the host-side entry point called from vLLM Attention.forward after
        q is available. It does not perform SSD-side qK/top-n yet;
        GdsBackend.select_sparse_kv_chunks() currently provides a stub that will
        be replaced by the SSD-CPU/NPU RPC in the next step.
        """
        spec = getattr(self, "sparse_kv_spec", SparseKVSpec())
        if not spec.enabled:
            logger.debug("[sparse-kv] sparse_select_kv_layer called but disabled")
            return None
        if spec.granularity != "chunk":
            logger.warning(
                "Sparse KV granularity %s is not supported in this first version; "
                "falling back to full attention",
                spec.granularity,
            )
            return None

        if self.lmcache_engine is None:
            logger.warning("[sparse-kv] no LMCache engine on worker connector")
            return None

        # Prefer bound connector metadata when it is available, but do not
        # require it.  In vLLM compile/CUDA-graph paths the attention hook can
        # execute after start_load_kv() has cached runtime request information
        # while connector metadata is not currently bound on _parent.  Falling
        # back to _sparse_runtime_requests keeps sparse selection usable in that
        # path.
        metadata_requests = []
        metadata = None
        if self._parent.has_connector_metadata():
            metadata = self._parent._get_connector_metadata()
            if isinstance(metadata, LMCacheConnectorMetadata):
                metadata_requests = list(metadata.requests)
            else:
                logger.warning(
                    "[sparse-kv] unexpected connector metadata type=%s for layer=%s; "
                    "falling back to runtime requests",
                    type(metadata).__name__,
                    layer_name,
                )
        else:
            if _sparse_kv_debug_enabled():
                logger.info(
                    "[sparse-kv] connector metadata is not bound for layer=%s; "
                    "falling back to runtime requests",
                    layer_name,
                )

        runtime_requests = getattr(self, "_sparse_runtime_requests", {}) or {}
        request_items: list[dict[str, Any]] = []

        # Metadata path: preserves per-request load_spec and request_configs.
        for request in metadata_requests:
            if request.load_spec is None or not request.load_spec.can_load:
                continue
            request_spec = getattr(request, "sparse_kv_spec", None) or spec
            if not request_spec.enabled:
                continue

            lmcache_cached_tokens = request.load_spec.lmcache_cached_tokens
            vllm_cached_tokens = request.load_spec.vllm_cached_tokens
            if lmcache_cached_tokens <= vllm_cached_tokens:
                continue

            tokens = request.token_ids[:lmcache_cached_tokens]
            if not tokens:
                continue
            token_mask = torch.ones(len(tokens), dtype=torch.bool)
            masked_token_count = (
                vllm_cached_tokens // self._lmcache_chunk_size * self._lmcache_chunk_size
            )
            token_mask[:masked_token_count] = False
            request_items.append(
                {
                    "req_id": request.req_id,
                    "tokens": tokens,
                    "token_mask": token_mask,
                    "slot_mapping_for_manifest": request.slot_mapping[:lmcache_cached_tokens],
                    "request_configs": request.request_configs,
                    "request_spec": request_spec,
                    "runtime": runtime_requests.get(request.req_id),
                    "source": "metadata",
                }
            )

        # Runtime fallback path: used when metadata is absent, or when metadata
        # did not yield eligible requests.  These entries are cached by
        # start_load_kv() and include the exact tokens/mask/slot_mapping needed
        # for selected chunk load.
        if not request_items and runtime_requests:
            for req_id, runtime in runtime_requests.items():
                request_spec = runtime.get("sparse_spec", spec)
                if request_spec is None or not getattr(request_spec, "enabled", False):
                    continue
                tokens = runtime.get("tokens", [])
                token_mask = runtime.get("token_mask", None)
                slot_mapping = runtime.get("slot_mapping", None)
                if not tokens or token_mask is None or slot_mapping is None:
                    continue
                slot_mapping_for_manifest = slot_mapping
                if isinstance(slot_mapping_for_manifest, torch.Tensor):
                    slot_mapping_for_manifest = slot_mapping_for_manifest.detach().to("cpu")
                request_items.append(
                    {
                        "req_id": req_id,
                        "tokens": tokens,
                        "token_mask": token_mask.detach().to("cpu")
                        if isinstance(token_mask, torch.Tensor) else token_mask,
                        "slot_mapping_for_manifest": slot_mapping_for_manifest,
                        "request_configs": runtime.get("request_configs"),
                        "request_spec": request_spec,
                        "runtime": runtime,
                        "source": "runtime",
                    }
                )

        if not request_items:
            if _sparse_kv_debug_enabled():
                logger.info(
                    "[sparse-kv] layer=%s q_shape=%s no eligible sparse runtime requests "
                    "metadata_bound=%s runtime_requests=%d",
                    layer_name,
                    tuple(query.shape),
                    self._parent.has_connector_metadata(),
                    len(runtime_requests),
                )
            return None

        q_manifest: dict[str, Any] = {
            "layer_name": layer_name,
            "shape": list(query.shape),
            "dtype": str(query.dtype),
            "device": str(query.device),
            "data_ptr": int(query.data_ptr()) if query.device.type == "cuda" else None,
            "score_scale": 1.0 / math.sqrt(query.shape[-1]) if query.ndim > 0 else None,
            "score_mode": spec.score_mode,
            "granularity": spec.granularity,
        }

        if _sparse_kv_debug_enabled():
            logger.info(
                "[sparse-kv] hook-enter layer=%s q_shape=%s requests=%d "
                "metadata_bound=%s runtime_requests=%d",
                layer_name,
                tuple(query.shape),
                len(request_items),
                self._parent.has_connector_metadata(),
                len(runtime_requests),
            )

        selected_by_request: list[dict[str, Any]] = []
        for item in request_items:
            req_id = item["req_id"]
            tokens = item["tokens"]
            token_mask = item["token_mask"]
            slot_mapping = item["slot_mapping_for_manifest"]
            request_configs = item.get("request_configs")
            request_spec = item["request_spec"]
            runtime = item.get("runtime")

            candidate_manifest = self.lmcache_engine.build_sparse_kv_candidate_manifest(
                tokens=tokens,
                mask=token_mask,
                request_configs=request_configs,
                req_id=req_id,
                layer_name=layer_name,
                slot_mapping=slot_mapping,
                chunk_size=self._lmcache_chunk_size,
            )
            selected_manifest = self.lmcache_engine.select_sparse_kv_chunks(
                q_manifest=q_manifest,
                candidate_manifest=candidate_manifest,
                top_n_chunks=request_spec.top_n_chunks,
                score_mode=request_spec.score_mode,
                req_id=req_id,
                layer_name=layer_name,
            )
            selected_load_mask = None
            selected_load_tokens = 0
            selected_loaded_chunks = 0
            selected_load_ms = 0.0
            selected_load_bytes = 0
            # Route 1: load selected chunks back into vLLM's paged KV cache via
            # the normal GPUConnector. In dry-run mode (disable_full_load=false),
            # this duplicates a subset of the full retrieve for measurement. In
            # real sparse mode, it is the replacement for full retrieve, but
            # sparse attention metadata/kernel must be enabled for correctness.
            selected_chunks = list(selected_manifest.get("selected_chunks", []))
            if runtime is not None and selected_chunks:
                # Avoid loading the same selected chunk multiple times across
                # layers in one forward step. LMCache non-layerwise chunks carry
                # all layers, so one selected load writes all layer KV for that
                # token chunk into the paged KV cache.
                filtered_chunks = []
                for chunk in selected_chunks:
                    chunk_hash = str(chunk.get("chunk_hash", chunk.get("key", "")))
                    load_key = (req_id, chunk_hash)
                    if load_key in self._sparse_loaded_chunk_hashes:
                        continue
                    self._sparse_loaded_chunk_hashes.add(load_key)
                    filtered_chunks.append(chunk)

                if filtered_chunks:
                    selected_manifest_to_load = dict(selected_manifest)
                    selected_manifest_to_load["selected_chunks"] = filtered_chunks
                    selected_loaded_chunks = len(filtered_chunks)
                    selected_load_bytes = sum(int(chunk.get("nbytes", 0) or 0) for chunk in filtered_chunks)
                    t_selected_load0 = time.perf_counter()
                    selected_load_mask = self.lmcache_engine.load_sparse_kv_selected_chunks(
                        selected_manifest_to_load,
                        kvcaches=list(self.kv_caches.values()),
                        slot_mapping=runtime["slot_mapping"],
                        vllm_cached_tokens=runtime["vllm_cached_tokens"],
                        request_configs=runtime.get("request_configs"),
                        req_id=req_id,
                        tokens_len=runtime["lmcache_cached_tokens"],
                    )
                    selected_load_ms = (time.perf_counter() - t_selected_load0) * 1000.0
                    selected_load_tokens = int(selected_load_mask.sum().item())
                    if selected_load_bytes <= 0:
                        selected_load_bytes = int(self._sparse_estimated_selected_kv_bytes(selected_load_tokens))
                    if _aissd_selector_stats_enabled():
                        bw = (float(selected_load_bytes) / (selected_load_ms / 1000.0) / 1.0e9) if selected_load_ms > 0 and selected_load_bytes > 0 else 0.0
                        logger.info(
                            "[sparse-kv-selected-load] req_id=%s layer=%s selected_chunks=%d "
                            "loaded_tokens=%d bytes=%d load_ms=%.3f bw_GBps=%.6f",
                            req_id,
                            layer_name,
                            selected_loaded_chunks,
                            selected_load_tokens,
                            selected_load_bytes,
                            selected_load_ms,
                            bw,
                        )

            selected_by_request.append(
                {
                    "req_id": req_id,
                    "candidate_manifest": candidate_manifest,
                    "selected_manifest": selected_manifest,
                    "sparse_kv_spec": request_spec,
                    "selected_load_tokens": selected_load_tokens,
                    "selected_loaded_chunks": selected_loaded_chunks,
                    "selected_load_ms": selected_load_ms,
                    "selected_load_bytes": selected_load_bytes,
                }
            )

        if not selected_by_request:
            if _sparse_kv_debug_enabled():
                logger.info(
                    "[sparse-kv] layer=%s q_shape=%s no eligible load requests",
                    layer_name,
                    tuple(query.shape),
                )
            return None

        result = {
            "enabled": True,
            "granularity": spec.granularity,
            "layer_name": layer_name,
            "q_manifest": q_manifest,
            "requests": selected_by_request,
        }
        total_candidates = sum(
            len(item.get("candidate_manifest", {}).get("chunks", []))
            for item in selected_by_request
        )
        total_selected = sum(
            len(item.get("selected_manifest", {}).get("selected_chunks", []))
            for item in selected_by_request
        )
        total_selected_loaded_chunks = sum(
            int(item.get("selected_loaded_chunks", 0)) for item in selected_by_request
        )
        total_selected_loaded_tokens = sum(
            int(item.get("selected_load_tokens", 0)) for item in selected_by_request
        )
        total_selected_load_ms = sum(
            float(item.get("selected_load_ms", 0.0)) for item in selected_by_request
        )
        total_selected_load_bytes = sum(
            int(item.get("selected_load_bytes", 0)) for item in selected_by_request
        )
        self._populate_sparse_attention_metadata(
            attn_metadata=attn_metadata,
            result=result,
            query_device=query.device,
            layer_name=layer_name,
        )
        step_context = getattr(self, "_sparse_current_step_context", None)
        if isinstance(step_context, dict) and total_selected_load_bytes > 0:
            step_context["sparse_selected_load_ms"] = float(total_selected_load_ms)
            step_context["sparse_selected_load_bytes"] = int(total_selected_load_bytes)
            step_context["sparse_selected_loaded_chunks"] = int(total_selected_loaded_chunks)
            step_context["sparse_selected_tokens"] = int(total_selected_loaded_tokens)
            step_context["sparse_selected_kv_bytes"] = int(total_selected_load_bytes)
            step_context["sparse_selected_kv_bytes_source"] = "actual_selected_load_bytes"
        if _aissd_selector_stats_enabled() and total_selected_load_bytes > 0:
            total_bw = (float(total_selected_load_bytes) / (total_selected_load_ms / 1000.0) / 1.0e9) if total_selected_load_ms > 0 else 0.0
            logger.info(
                "[sparse-kv-selected-load-summary] layer=%s requests=%d selected_loaded_chunks=%d "
                "selected_loaded_tokens=%d bytes=%d load_ms=%.3f bw_GBps=%.6f",
                layer_name,
                len(selected_by_request),
                total_selected_loaded_chunks,
                total_selected_loaded_tokens,
                total_selected_load_bytes,
                total_selected_load_ms,
                total_bw,
            )
        if _sparse_kv_debug_enabled():
            logger.info(
                "[sparse-kv] layer=%s requests=%d q_shape=%s candidate_chunks=%d selected_chunks=%d selected_loaded_chunks=%d selected_loaded_tokens=%d",
                layer_name,
                len(selected_by_request),
                q_manifest["shape"],
                total_candidates,
                total_selected,
                total_selected_loaded_chunks,
                total_selected_loaded_tokens,
            )
        return result


    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def select_and_load_sparse_kv_for_attention(
        self,
        layer_name: str,
        query: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> Optional[dict[str, Any]]:
        """Backend-facing alias for q-aware sparse selected KV load.

        SparseSSDAttentionImpl calls this from the real vLLM attention backend
        path.  Keep sparse_select_kv_layer() as the implementation to preserve
        compatibility with older debug hooks.
        """
        return self.sparse_select_kv_layer(
            layer_name=layer_name,
            query=query,
            attn_metadata=attn_metadata,
            **kwargs,
        )


    def _populate_sparse_attention_metadata(
        self,
        attn_metadata: "AttentionMetadata",
        result: dict[str, Any],
        query_device: torch.device,
        layer_name: str,
    ) -> None:
        """Convert selected chunk manifests into sparse attention metadata.

        Route 1 writes selected KV chunks back into vLLM's normal paged KV cache.
        The sparse attention kernel therefore only needs a selected block table
        that points to the already-populated paged-cache blocks.

        Tensors added here are intentionally simple and backend-agnostic:
          - sparse_block_table_tensor: [num_reqs, max_selected_blocks], block ids
          - sparse_block_lens: [num_reqs], valid block count per request
          - sparse_token_ranges_tensor: [num_reqs, max_selected_chunks, 2]
          - sparse_slot_ranges_tensor: [num_reqs, max_selected_chunks, 2]
          - sparse_chunk_lens: [num_reqs], valid chunk count per request
        """
        requests = result.get("requests", [])
        if not requests:
            return

        block_rows: list[list[int]] = []
        token_range_rows: list[list[tuple[int, int]]] = []
        slot_range_rows: list[list[tuple[int, int]]] = []
        total_selected_chunks = 0

        runtime_requests = getattr(self, "_sparse_runtime_requests", {})
        for item in requests:
            req_id = item.get("req_id")
            runtime = runtime_requests.get(req_id, {})
            slot_mapping = runtime.get("slot_mapping")
            selected_chunks = list(
                item.get("selected_manifest", {}).get("selected_chunks", [])
            )
            total_selected_chunks += len(selected_chunks)

            block_ids: list[int] = []
            seen_blocks: set[int] = set()
            token_ranges: list[tuple[int, int]] = []
            slot_ranges: list[tuple[int, int]] = []

            for chunk in selected_chunks:
                token_start = int(chunk.get("token_start", 0))
                token_end = int(chunk.get("token_end", token_start))
                if token_end <= token_start:
                    continue
                token_ranges.append((token_start, token_end))

                slot_start: int | None = None
                slot_end: int | None = None
                if isinstance(slot_mapping, torch.Tensor) and slot_mapping.numel() > 0:
                    start = max(0, min(token_start, int(slot_mapping.numel())))
                    end = max(start, min(token_end, int(slot_mapping.numel())))
                    slots = slot_mapping[start:end]
                    if slots.numel() > 0:
                        slot_start = int(slots[0].item())
                        slot_end = int(slots[-1].item()) + 1
                        # Preserve block order instead of relying on torch.unique sorting.
                        blocks_cpu = (slots.detach().to("cpu", dtype=torch.long) // self._block_size).tolist()
                        for block_id in blocks_cpu:
                            block_id = int(block_id)
                            if block_id not in seen_blocks:
                                seen_blocks.add(block_id)
                                block_ids.append(block_id)
                if slot_start is None:
                    # Fallback for manifests that already carry slot ranges.
                    raw_slot_start = chunk.get("slot_start", None)
                    raw_slot_end = chunk.get("slot_end", None)
                    if raw_slot_start is not None and raw_slot_end is not None:
                        slot_start = int(raw_slot_start)
                        slot_end = int(raw_slot_end)
                        first_block = slot_start // self._block_size
                        last_block = max(slot_start, slot_end - 1) // self._block_size
                        for block_id in range(first_block, last_block + 1):
                            if block_id not in seen_blocks:
                                seen_blocks.add(block_id)
                                block_ids.append(block_id)
                if slot_start is not None and slot_end is not None:
                    slot_ranges.append((slot_start, slot_end))

            block_rows.append(block_ids)
            token_range_rows.append(token_ranges)
            slot_range_rows.append(slot_ranges)

        num_reqs = len(block_rows)
        max_blocks = max((len(row) for row in block_rows), default=0)
        max_chunks = max((len(row) for row in token_range_rows), default=0)
        if max_blocks == 0 or max_chunks == 0:
            if _sparse_kv_debug_enabled():
                logger.info(
                    "[sparse-kv-meta] layer=%s no selected sparse blocks/chunks",
                    layer_name,
                )
            return

        table = torch.full(
            (num_reqs, max_blocks),
            -1,
            dtype=torch.int32,
            device=query_device,
        )
        lens = torch.zeros(num_reqs, dtype=torch.int32, device=query_device)
        token_ranges_tensor = torch.full(
            (num_reqs, max_chunks, 2),
            -1,
            dtype=torch.int32,
            device=query_device,
        )
        slot_ranges_tensor = torch.full(
            (num_reqs, max_chunks, 2),
            -1,
            dtype=torch.int64,
            device=query_device,
        )
        chunk_lens = torch.zeros(num_reqs, dtype=torch.int32, device=query_device)

        for row_idx, block_ids in enumerate(block_rows):
            if block_ids:
                table[row_idx, : len(block_ids)] = torch.tensor(
                    block_ids, dtype=torch.int32, device=query_device
                )
                lens[row_idx] = len(block_ids)
            tr = token_range_rows[row_idx]
            sr = slot_range_rows[row_idx]
            if tr:
                token_ranges_tensor[row_idx, : len(tr), :] = torch.tensor(
                    tr, dtype=torch.int32, device=query_device
                )
                chunk_lens[row_idx] = len(tr)
            if sr:
                slot_ranges_tensor[row_idx, : len(sr), :] = torch.tensor(
                    sr, dtype=torch.int64, device=query_device
                )

        sparse_meta = {
            "sparse_block_table_tensor": table,
            "sparse_block_lens": lens,
            "sparse_token_ranges_tensor": token_ranges_tensor,
            "sparse_slot_ranges_tensor": slot_ranges_tensor,
            "sparse_chunk_lens": chunk_lens,
        }
        result.update(sparse_meta)

        # Attach both to the backend-specific attention metadata object and to
        # the common metadata object when present. Many vLLM metadata classes are
        # normal Python objects, so dynamic attributes are acceptable here.
        targets = [attn_metadata]
        common_meta = getattr(attn_metadata, "common_metadata", None)
        if common_meta is not None and common_meta is not attn_metadata:
            targets.append(common_meta)
        for target in targets:
            try:
                setattr(target, "sparse_kv_enabled", True)
                setattr(target, "sparse_block_table_tensor", table)
                setattr(target, "sparse_block_lens", lens)
                setattr(target, "sparse_token_ranges_tensor", token_ranges_tensor)
                setattr(target, "sparse_slot_ranges_tensor", slot_ranges_tensor)
                setattr(target, "sparse_chunk_lens", chunk_lens)
                setattr(target, "sparse_selector_result", result)
            except Exception:
                logger.debug(
                    "[sparse-kv-meta] failed to attach metadata to %s",
                    type(target).__name__,
                    exc_info=True,
                )

        if _sparse_kv_debug_enabled():
            logger.info(
                "[sparse-kv-meta] layer=%s reqs=%d selected_chunks=%d max_blocks=%d total_blocks=%d",
                layer_name,
                num_reqs,
                total_selected_chunks,
                max_blocks,
                int(lens.sum().item()),
            )

    @_lmcache_nvtx_annotate
    def wait_for_layer_load(self, layer_name: str) -> None:
        """Blocking until the KV for a specific layer is loaded into vLLM's
        paged buffer.

        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
        """
        if self.layerwise_retrievers:
            logger.debug(f"Waiting for layer {self.current_layer} to be loaded")

        # Wait for the layer to be loaded
        for layerwise_retriever in self.layerwise_retrievers:
            ret_token_mask = next(layerwise_retriever)

            if self.current_layer == self.num_layers - 1:
                assert ret_token_mask is not None
                num_retrieved_tokens = ret_token_mask.sum().item()
                logger.info(f"Retrieved {num_retrieved_tokens} tokens")

        if self.layerwise_retrievers:
            self.current_layer += 1

        return

    @_lmcache_nvtx_annotate
    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        """Start saving the a layer of KV cache from vLLM's paged buffer
        to the connector.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """
        assert self.lmcache_engine is not None

        if not self.use_layerwise:
            return

        if self.kv_role == "kv_consumer":
            # Don't do save if the role is kv_consumer
            return
        if self._parent._connector_metadata is None:
            logger.warning(
                "In connector.save_kv_layer, but the connector metadata is None"
            )
            return
        connector_metadata = self._parent._get_connector_metadata()
        assert isinstance(connector_metadata, LMCacheConnectorMetadata)

        assert len(self.kv_caches) > 0

        kvcaches = list(self.kv_caches.values())
        is_first = True

        for request in connector_metadata.requests:
            save_spec = request.save_spec
            if (
                save_spec is None or not save_spec.can_save
            ) and self.kv_role != "kv_producer":
                continue

            layerwise_storer = self._layerwise_save_storers.get(request.req_id)
            if layerwise_storer is None:
                token_ids = request.token_ids
                assert isinstance(token_ids, list)

                slot_mapping = request.slot_mapping
                assert isinstance(slot_mapping, torch.Tensor)
                assert len(slot_mapping) == len(token_ids)

                # TODO: have a pre-allocated buffer to hold the slot_mappings
                slot_mapping = slot_mapping.to(self.device)

                if self.kv_role == "kv_producer":
                    skip_leading_tokens = 0
                else:
                    assert save_spec is not None
                    skip_leading_tokens = save_spec.skip_leading_tokens

                    if skip_leading_tokens == len(token_ids):
                        continue  # skip this request
                    # Align to lmcache chunk size
                    skip_leading_tokens = (
                        skip_leading_tokens
                        // self._lmcache_chunk_size
                        * self._lmcache_chunk_size
                    )

                store_mask = torch.ones(len(token_ids), dtype=torch.bool)
                store_mask[:skip_leading_tokens] = False

                logger.debug(
                    "Storing KV cache for %d out of %d tokens "
                    "(skip_leading_tokens=%d) for request %s",
                    len(token_ids) - skip_leading_tokens,
                    len(token_ids),
                    skip_leading_tokens,
                    request.req_id,
                )

                # TODO (Jiayi): need to make layerwise storing
                # compatible with disagg spec
                layerwise_storer = self.lmcache_engine.store_layer(
                    token_ids,
                    mask=store_mask,
                    kvcaches=kvcaches,
                    slot_mapping=slot_mapping,
                    offset=skip_leading_tokens,
                    sync=is_first,
                    req_id=request.req_id,
                )
                self._layerwise_save_storers[request.req_id] = layerwise_storer
                if is_first:
                    is_first = False

            next(layerwise_storer)

    @_lmcache_nvtx_annotate
    def wait_for_save(self):
        """Blocking until the KV cache is saved to the connector buffer."""

        connector_metadata = self._parent._get_connector_metadata()
        assert isinstance(connector_metadata, LMCacheConnectorMetadata)

        if self.kv_role == "kv_consumer":
            # Don't do save if the role is kv_consumer
            # But still need to unpin the kv caches according to req_id
            # to balance the pin count from contains()
            assert self.lmcache_engine is not None, (
                "LMCacheEngine must be initialized to unpin requests."
            )
            for request in connector_metadata.requests:
                self.lmcache_engine.lookup_unpin(request.req_id)

            return

        if self.use_layerwise:
            for request in connector_metadata.requests:
                layerwise_storer = self._layerwise_save_storers.pop(
                    request.req_id, None
                )
                if layerwise_storer is not None:
                    next(layerwise_storer)
                # unpin the kv caches according to req_id
                self.lmcache_engine.lookup_unpin(request.req_id)
            return

        assert len(self.kv_caches) > 0
        kvcaches = list(self.kv_caches.values())

        assert self.lmcache_engine is not None

        # Probe decoder cache before store if bidirectional mode is enabled
        bidir_enabled = getattr(self.config, "pd_bidirectional", False)

        for request in connector_metadata.requests:
            # unpin the kv caches according to req_id
            self.lmcache_engine.lookup_unpin(request.req_id)

            save_spec = request.save_spec
            if (
                save_spec is None or not save_spec.can_save
            ) and self.kv_role != "kv_producer":
                continue

            token_ids = request.token_ids

            slot_mapping = request.slot_mapping
            assert isinstance(slot_mapping, torch.Tensor)
            assert len(slot_mapping) == len(token_ids)

            # TODO: have a pre-allocated buffer to hold the slot_mappings
            slot_mapping = slot_mapping.to(self.device)

            skip_leading_tokens = save_spec.skip_leading_tokens
            # shared storage disaggregation will not have a disagg_spec passed in
            if self.kv_role == "kv_producer" and request.disagg_spec:
                skip_leading_tokens = min(
                    skip_leading_tokens, request.disagg_spec.num_transferred_tokens
                )

            if skip_leading_tokens == len(token_ids):
                continue  # skip this request
            # Align to lmcache chunk size
            skip_leading_tokens = (
                skip_leading_tokens
                // self._lmcache_chunk_size
                * self._lmcache_chunk_size
            )

            store_mask = torch.ones(len(token_ids), dtype=torch.bool)
            store_mask[:skip_leading_tokens] = False

            logger.debug(
                "Storing KV cache for %d out of %d tokens "
                "(skip_leading_tokens=%d) for request %s",
                len(token_ids) - skip_leading_tokens,
                len(token_ids),
                skip_leading_tokens,
                request.req_id,
            )

            is_last_prefill = request.is_last_prefill
            if is_last_prefill:
                if request.disagg_spec:
                    request.disagg_spec.is_last_prefill = True
            else:
                if not self.enable_blending:
                    token_len = len(token_ids)
                    aligned_token_len = (
                        token_len // self._lmcache_chunk_size * self._lmcache_chunk_size
                    )
                    token_ids = token_ids[:aligned_token_len]
                    store_mask = store_mask[:aligned_token_len]
                    slot_mapping = slot_mapping[:aligned_token_len]

            # Probe decoder cache before store
            if bidir_enabled and request.disagg_spec is not None:
                try:
                    self._probe_decoder_cache(request, token_ids)
                except Exception as e:
                    logger.warning(
                        "Bidirectional NIXL cache probe failed for %s: %s",
                        request.req_id,
                        e,
                    )

            self.lmcache_engine.store(
                token_ids,
                mask=store_mask,
                kvcaches=kvcaches,
                slot_mapping=slot_mapping,
                offset=skip_leading_tokens,
                transfer_spec=request.disagg_spec,
                request_configs=request.request_configs,
                req_id=request.req_id,
            )

            # Probe decoder cache after store
            if (
                bidir_enabled
                and request.disagg_spec is not None
                and request.disagg_spec.receiver_query_port is not None
            ):
                try:
                    self._probe_decoder_cache(request, token_ids)
                except Exception as e:
                    logger.warning(
                        "Bidirectional NIXL cache probe failed for %s: %s",
                        request.req_id,
                        e,
                    )

            # Update skip_leading_tokens only on last rank to ensure
            # each PP stage stores its own KV cache
            if get_pp_group().is_last_rank:
                # NOTE(Jiayi): We assume all tokens are saved
                save_spec.skip_leading_tokens = len(token_ids)
                if request.disagg_spec:
                    request.disagg_spec.num_transferred_tokens = len(token_ids)

    def _probe_decoder_cache(self, request: ReqMeta, token_ids: list[int]) -> None:
        """Query the decoder's cache to check which blocks are already cached.

        This is the bidirectional NIXL cache probe: the prefiller queries the
        decoder via ZMQ to find out which KV blocks are already in the
        decoder's GPU memory. This validates the cache query channel works
        E2E through the real inference path.

        In the future, this information can be used to skip prefill
        computation for cached blocks.
        """
        sm = self.lmcache_engine.storage_manager  # type: ignore[union-attr]
        if sm is None or sm.allocator_backend is None:
            return
        pd_backend = sm.allocator_backend
        if not hasattr(pd_backend, "query_remote_cache"):
            return
        if not hasattr(pd_backend, "cache_query_sockets"):
            return

        # Get query port from LMCache config (pd_peer_query_port)
        query_ports = self.config.pd_peer_query_port
        if query_ports is None:
            return

        # Build cache keys using the token database's process_tokens
        td = self.lmcache_engine.token_database  # type: ignore[union-attr]
        if td is None:
            return

        chunk_keys = []
        for _start, _end, key in td.process_tokens(
            tokens=token_ids, mask=None, make_key=True
        ):
            chunk_keys.append(key)

        if not chunk_keys:
            return

        # Build receiver_id from disagg_spec
        disagg = request.disagg_spec
        init_port = disagg.receiver_init_port  # type: ignore[union-attr]
        if isinstance(init_port, list):
            init_port = init_port[pd_backend.tp_rank]  # type: ignore[union-attr]
        receiver_id = disagg.receiver_host + str(init_port)  # type: ignore[union-attr]

        # Ensure peer and cache query connections
        alloc_port = disagg.receiver_alloc_port  # type: ignore[union-attr]
        if isinstance(alloc_port, list):
            alloc_port = alloc_port[pd_backend.tp_rank]  # type: ignore[union-attr]
        query_port = query_ports[pd_backend.tp_rank]  # type: ignore[union-attr]

        pd_backend._ensure_peer_connection(  # type: ignore[union-attr]
            receiver_id=receiver_id,
            receiver_host=disagg.receiver_host,  # type: ignore[union-attr]
            receiver_init_port=init_port,
            receiver_alloc_port=alloc_port,
        )
        pd_backend._ensure_cache_query_connection(  # type: ignore[union-attr]
            receiver_id=receiver_id,
            receiver_host=disagg.receiver_host,  # type: ignore[union-attr]
            receiver_query_port=query_port,
        )

        # Query decoder cache
        cache_resp = pd_backend.query_remote_cache(receiver_id, chunk_keys)

        logger.info(
            "Bidirectional NIXL cache probe: req=%s, "
            "queried %d chunks, decoder has %d cached "
            "(%.0f%% hit rate)",
            request.req_id,
            len(chunk_keys),
            len(cache_resp.cached_keys),
            100.0 * len(cache_resp.cached_keys) / len(chunk_keys) if chunk_keys else 0,
        )

    @_lmcache_nvtx_annotate
    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        return None, None

    def get_block_ids_with_load_errors(self) -> set[int]:
        invalid_blocks = self._invalid_block_ids.copy()
        self._invalid_block_ids.clear()
        return invalid_blocks

    @_lmcache_nvtx_annotate
    def shutdown(self):
        """Shutdown the connector by delegating to LMCacheManager."""
        logger.info("Starting LMCacheConnector shutdown...")
        self._manager.stop_services()

    ###################
    # Scheduler side APIs
    ####################

    @_lmcache_nvtx_annotate
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> Optional[int]:
        """
        Check for external KV cache hit.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            the number of tokens that can be loaded from the
            external KV cache beyond what is already computed.
        """
        # Ignore DP attention mock requests
        if request.request_id.startswith("mock_req"):
            return 0
        # to handle preempted requests, we want `get_num_new_matched_tokens` to be
        # idempotent under the condition that `update_state_after_alloc` is NOT called
        # then the two side-effects that must be idempotent are:
        # 1. lookup_client caches a result
        #     uncached in `update_state_after_alloc` if this request can be scheduled
        # 2. cache engine will pin the KV caches for the request
        #     unpinned in `wait_for_save` if this request can be scheduled
        if self.kv_role == "kv_producer" and not hasattr(
            self.lookup_client, "supports_producer_reuse"
        ):
            return 0

        req_id = request.request_id

        # lookup_client is always initialized for scheduler role
        assert self.lookup_client is not None

        if (
            num_external_hit_tokens := self.lookup_client.lookup_cache(lookup_id=req_id)
        ) != -1:
            # -1 means no result cached
            # None or int means ongoing (async) or cached result
            logger.debug(
                f"Found {num_external_hit_tokens} hit tokens for request"
                f" {req_id} in the lookup cache."
            )
        else:
            logger.debug(f"Looking up cache for the first time for request {req_id}!")
            self._requests_priority[req_id] = getattr(request, "priority", 0)

            # token_ids = request.prompt_token_ids
            # all token ids covers the preemption case
            token_ids = request.all_token_ids

            # If the request has multimodal hashes, apply them to the token ids
            mm_hashes, mm_positions = extract_mm_features(request)
            if mm_hashes and mm_positions:
                # TODO(Jiayi): Optimize this
                token_ids = torch.tensor(request.prompt_token_ids)
                apply_mm_hashes_to_token_ids(token_ids, mm_hashes, mm_positions)
                token_ids = token_ids.tolist()

            request_configs = extract_request_configs(request.sampling_params)
            if self.skip_last_n_tokens > 0:
                token_ids = token_ids[: -self.skip_last_n_tokens]

            num_external_hit_tokens = self.lookup_client.lookup(
                token_ids,
                lookup_id=req_id,
                request_configs=request_configs,
            )

        if num_external_hit_tokens is None:
            logger.debug(
                "Reqid: %s, Total tokens %d, Inference Engine computed tokens: %d, "
                "LMCache hit tokens: None.",
                req_id,
                request.num_tokens,
                num_computed_tokens,
            )
            return None

        # When prompt length is divisible by the block size and all
        # blocks are cached, we need to recompute the last token.
        # This will be removed in the future if vLLM's scheduler provides
        # a better support for this case.
        need_to_allocate = num_external_hit_tokens - num_computed_tokens

        # In, full-prompt-hit case, we need to recompute the last token
        if num_external_hit_tokens == request.num_tokens:
            need_to_allocate -= 1

        # Check if hit tokens meet the minimum for retrieve
        # If below minimum, skip retrieve but still record hit tokens
        # for skip_leading_tokens to avoid re-storing existing chunks
        min_retrieve = self.config.min_retrieve_tokens
        below_min_retrieve = min_retrieve > 0 and need_to_allocate < min_retrieve

        if below_min_retrieve:
            logger.info(
                "Reqid: %s, Total tokens %d, Inference Engine computed tokens: %d, "
                "LMCache hit tokens: %d, but need to load: %d < min_retrieve %d, "
                "skip retrieve but record for save skip",
                req_id,
                request.num_tokens,
                num_computed_tokens,
                num_external_hit_tokens,
                max(need_to_allocate, 0),
                min_retrieve,
            )
        else:
            logger.info(
                "Reqid: %s, Total tokens %d, Inference Engine computed tokens: %d, "
                "LMCache hit tokens: %d, need to load: %d",
                req_id,
                request.num_tokens,
                num_computed_tokens,
                num_external_hit_tokens,
                max(need_to_allocate, 0),
            )

        self.load_specs[req_id] = LoadSpec(
            vllm_cached_tokens=num_computed_tokens,
            lmcache_cached_tokens=num_external_hit_tokens,
            can_load=False,
        )

        if below_min_retrieve or need_to_allocate <= 0:
            return 0

        # TODO: Align to vLLM block size. Should test whether it can be removed
        # need_to_allocate = need_to_allocate // self._block_size * \
        #        self._block_size

        # This vLLM scheduler path expects a plain int/None.  Returning a
        # tuple here breaks token-fate tracing, which compares this value with
        # an int.  Keep async-load disabled by using the synchronous retrieve
        # path in start_load_kv().
        if _sparse_kv_debug_enabled():
            logger.info(
                "[lmcache-kv-iface] get_num_new_matched_tokens req_id=%s "
                "return=%d",
                req_id,
                need_to_allocate,
            )
        return need_to_allocate

    @_lmcache_nvtx_annotate
    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: Any = None,
        num_external_tokens: Optional[int] = None,
    ):
        """Update KVConnector state after vLLM KV block allocation.

        vLLM >= 0.19 calls this as:
            update_state_after_alloc(request, blocks, num_external_tokens)
        Older wrappers may still call:
            update_state_after_alloc(request, num_external_tokens)

        The implementation accepts both forms.  `blocks` is not required by
        the LMCache pull path, because ReqMeta reconstructs slot_mapping from
        the request's allocated block ids.
        """

        # Backward compatibility with old wrapper calls where the second
        # positional argument was num_external_tokens.
        if num_external_tokens is None:
            if isinstance(blocks, int):
                num_external_tokens = blocks
                blocks = None
            else:
                num_external_tokens = 0

        # Some connector wrappers may accidentally pass a tuple from the new
        # get_num_new_matched_tokens() API.  Normalize defensively.
        if isinstance(num_external_tokens, tuple):
            num_external_tokens = num_external_tokens[0]
        num_external_tokens = int(num_external_tokens or 0)

        if _sparse_kv_debug_enabled():
            logger.info(
                "[lmcache-kv-iface] update_state_after_alloc req_id=%s "
                "num_external_tokens=%d has_blocks=%s",
                request.request_id,
                num_external_tokens,
                blocks is not None,
            )

        # Clear local status in lookup client when a new request is
        # successfully scheduled.
        assert self.lookup_client is not None
        self.lookup_client.clear_lookup_status(request.request_id)

        kv_transfer_params = (
            request.kv_transfer_params
            if hasattr(request, "kv_transfer_params")
            else None
        )

        if kv_transfer_params is not None and "disagg_spec" in kv_transfer_params:
            req_disagg_spec = kv_transfer_params["disagg_spec"]

            receiver_id = req_disagg_spec["receiver_host"] + str(
                req_disagg_spec["receiver_init_port"]
            )

            disagg_spec = DisaggSpec(
                req_id=req_disagg_spec["req_id"],
                receiver_id=receiver_id,
                receiver_host=req_disagg_spec["receiver_host"],
                receiver_init_port=req_disagg_spec["receiver_init_port"],
                receiver_alloc_port=req_disagg_spec["receiver_alloc_port"],
                receiver_query_port=req_disagg_spec.get("receiver_query_port"),
            )

            tmp_disagg_tracker[request.request_id] = disagg_spec
        self._unfinished_requests[request.request_id] = request

        if request.request_id not in self.load_specs:
            if _sparse_kv_debug_enabled():
                logger.info(
                    "[lmcache-kv-iface] update_state_after_alloc req_id=%s "
                    "has no load_spec",
                    request.request_id,
                )
            # No KV tokens from external KV cache, return
            return

        if num_external_tokens == 0:
            # No need to load anything
            self.load_specs[request.request_id].can_load = False
            if _sparse_kv_debug_enabled():
                logger.info(
                    "[lmcache-kv-iface] update_state_after_alloc req_id=%s "
                    "can_load=False",
                    request.request_id,
                )
            return

        recalc_last = (
            1
            if (
                self.load_specs[request.request_id].lmcache_cached_tokens
                == request.num_tokens
            )
            else 0
        )
        assert (
            num_external_tokens
            == self.load_specs[request.request_id].lmcache_cached_tokens
            - self.load_specs[request.request_id].vllm_cached_tokens
            - recalc_last
        ), (
            f"Mismatch in tokens to load: {num_external_tokens} vs "
            f"{self.load_specs[request.request_id].lmcache_cached_tokens} "
            "(tokens in lmcache) - "
            f"{self.load_specs[request.request_id].vllm_cached_tokens} "
            "(tokens in vllm) - "
            f"{recalc_last} "
            "(full lmcache hits subtracts last token to recalculate logits)"
            f" for request {request.request_id}"
        )

        self.load_specs[request.request_id].can_load = True
        if _sparse_kv_debug_enabled():
            logger.info(
                "[lmcache-kv-iface] update_state_after_alloc req_id=%s can_load=True "
                "vllm_cached_tokens=%d lmcache_cached_tokens=%d",
                request.request_id,
                self.load_specs[request.request_id].vllm_cached_tokens,
                self.load_specs[request.request_id].lmcache_cached_tokens,
            )

    @_lmcache_nvtx_annotate
    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        """Attach the connector metadata to the request object.

        This function should NOT modify other fields in the scheduler_output
        except the `kv_connector_metadata` field.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """

        force_skip_save = self.kv_role == "kv_consumer" or self.force_skip_save

        meta = LMCacheConnectorMetadata()

        for finished_req_id in scheduler_output.finished_req_ids:
            self._request_trackers.pop(finished_req_id, None)
            self._unfinished_requests.pop(finished_req_id, None)

        # We should load KV for:
        # 1. new requests
        # 2. preempted requests (once per recovery)
        # can_load will only be True if `update_state_after_alloc` has been called
        # which only happens when vLLM's KV manager has space to receive KV from LMCache
        for request in scheduler_output.scheduled_new_reqs:
            # Ignore DP attention mock requests
            if request.req_id.startswith("mock_req"):
                continue
            load_spec = self.load_specs.pop(request.req_id, None)
            num_tokens_to_compute = (
                request.num_computed_tokens
                + scheduler_output.num_scheduled_tokens[request.req_id]
            )
            lmcache_cached_tokens = 0
            if load_spec is not None:
                lmcache_cached_tokens = load_spec.lmcache_cached_tokens
            request_priority = self._requests_priority.pop(request.req_id, 0)

            skip_save = force_skip_save or (
                self.config.priority_limit is not None
                and request_priority > self.config.priority_limit
            )

            request_tracker = RequestTracker.from_new_request(
                self.config,
                request,
                num_tokens_to_compute,
                lmcache_cached_tokens,
                skip_save,
            )
            self._request_trackers[request.req_id] = request_tracker

            req_meta = ReqMeta.from_request_tracker(
                request_tracker,
                self._block_size,
                self._lmcache_chunk_size,
                load_spec=load_spec,
                discard_partial_chunks=self._discard_partial_chunks,
                save_decode_cache=self.config.save_decode_cache,
            )
            if req_meta is not None:
                if self.sparse_kv_spec.enabled:
                    req_meta.sparse_kv_spec = self.sparse_kv_spec
                meta.add_request(req_meta)

        cached_reqs = scheduler_output.scheduled_cached_reqs

        # NOTE: For backward compatibility with vllm version < 0.9.2,
        # In the latest vllm version, the type of scheduled_cached_reqs has
        # changed from list to object `CachedRequestData`
        if isinstance(cached_reqs, list):
            for i, req in enumerate(cached_reqs):
                load_spec = self.load_specs.pop(req.req_id, None)
                lmcache_cached_tokens = 0
                vllm_cached_tokens = 0
                if load_spec is not None:
                    lmcache_cached_tokens = load_spec.lmcache_cached_tokens
                    vllm_cached_tokens = load_spec.vllm_cached_tokens
                request_tracker = self._request_trackers[req.req_id]

                # Pass all_token_ids for preempted requests to restore
                # token_ids correctly for chunk key computation
                all_token_ids = None
                if req.resumed_from_preemption:
                    vllm_request = self._unfinished_requests.get(req.req_id)
                    assert vllm_request is not None, (
                        f"Preempted request {req.req_id} not found "
                        "in _unfinished_requests"
                    )
                    all_token_ids = list(vllm_request.all_token_ids)

                request_tracker.update(
                    req.new_token_ids,
                    req.new_block_ids,
                    req.resumed_from_preemption,
                    lmcache_cached_tokens=lmcache_cached_tokens,
                    vllm_cached_tokens=vllm_cached_tokens,
                    all_token_ids=all_token_ids,
                )

                req_meta = ReqMeta.from_request_tracker(
                    request_tracker,
                    self._block_size,
                    self._lmcache_chunk_size,
                    load_spec=load_spec,
                    discard_partial_chunks=self._discard_partial_chunks,
                    save_decode_cache=self.config.save_decode_cache,
                )
                if req_meta is not None:
                    if self.sparse_kv_spec.enabled:
                        req_meta.sparse_kv_spec = self.sparse_kv_spec
                    meta.add_request(req_meta)
            if _sparse_kv_debug_enabled():
                logger.info(
                    "[lmcache-kv-iface] build_connector_meta requests=%d",
                    len(meta.requests),
                )
            return meta

        for i, req_id in enumerate(cached_reqs.req_ids):
            request_tracker = self._request_trackers[req_id]
            num_new_tokens = scheduler_output.num_scheduled_tokens[req_id]
            # TODO: this is a dangerous reference to the request object inside vllm
            if request := self._unfinished_requests.get(req_id):
                num_current_tokens = request.num_computed_tokens
                # tracker_len < num_computed_tokens during decode
                #   (important for save_decode_cache).
                # num_computed_tokens < tracker_len after preemption.
                tracker_len = len(request_tracker.token_ids)
                slice_base = min(num_current_tokens, tracker_len)
                new_token_ids = request.all_token_ids[
                    slice_base : slice_base + num_new_tokens
                ]
            else:
                raise ValueError(
                    f"Request {req_id} is not in _unfinished_requests, "
                    f"but it is scheduled to be cached"
                )
            new_block_ids = cached_reqs.new_block_ids[i]

            load_spec = self.load_specs.pop(req_id, None)
            lmcache_cached_tokens = 0
            vllm_cached_tokens = 0
            if load_spec is not None:
                lmcache_cached_tokens = load_spec.lmcache_cached_tokens
                vllm_cached_tokens = load_spec.vllm_cached_tokens

            # Handle both old and new versions of CachedRequestData
            if hasattr(cached_reqs, "resumed_req_ids"):
                # New version with resumed_req_ids
                preempted = req_id in cached_reqs.resumed_req_ids
            elif hasattr(cached_reqs, "resumed_from_preemption"):
                # Old version with resumed_from_preemption
                preempted = cached_reqs.resumed_from_preemption[i]
            else:
                # This case should not be reached with supported vLLM versions.
                # Raising an error is safer than assuming not preempted.
                raise AttributeError(
                    f"Unable to determine preemption status for request {req_id}. "
                    f"This might be due to an unsupported vLLM version."
                )
            if preempted:
                assert load_spec is not None, (
                    f"Request {req_id} is preempted but was not given a load spec"
                )
                # num_computed_tokens should be reset to 0 during preemption
                # and then set to the number of already cached tokens (maxxing
                # prefix caching and lmcache)
                # this assumption is crucial for the update() call of RequestTracker
                # On full cache hit, get_num_new_matched_tokens subtracts 1
                # to force last-token recomputation. This only affects
                # num_computed_tokens when lmcache has all tokens AND
                # provides more than vLLM's local cache.
                expected = max(lmcache_cached_tokens, load_spec.vllm_cached_tokens)
                full_hit_adj = (
                    lmcache_cached_tokens == len(request.all_token_ids)
                    and lmcache_cached_tokens > load_spec.vllm_cached_tokens
                )
                if full_hit_adj:
                    expected -= 1
                assert request.num_computed_tokens == expected, (
                    f"Preempted request {req_id} has "
                    f"num_computed_tokens {request.num_computed_tokens} "
                    f"but expected {expected} "
                    f"(full_hit_adj={full_hit_adj})"
                )

            # When retrieve fail, vllm will call _handle_invalid_blocks to
            # reset request.num_computed_tokens, this will lead to
            # request_tracker.token_ids being not matched with vllm
            if num_current_tokens < len(request_tracker.token_ids):
                logger.warning(
                    "Request %s rolled back from %d to %d tokens; "
                    "truncating tracker state.",
                    req_id,
                    len(request_tracker.token_ids),
                    num_current_tokens,
                )
                num_token_slots = (
                    len(request_tracker.allocated_block_ids) * self._block_size
                )
                tokens_to_keep = num_current_tokens
                if num_token_slots < num_current_tokens:
                    logger.warning(
                        "Request %s tracker has %d token slots but %d tokens; "
                        "capping token_ids to slot capacity.",
                        req_id,
                        num_token_slots,
                        num_current_tokens,
                    )
                    tokens_to_keep = num_token_slots

                request_tracker.token_ids = list(request.all_token_ids[:tokens_to_keep])
                request_tracker.num_saved_tokens = min(
                    request_tracker.num_saved_tokens, tokens_to_keep
                )

            # Pass all_token_ids for preempted requests to restore
            # token_ids correctly for chunk key computation
            all_token_ids = list(request.all_token_ids) if preempted else None

            request_tracker.update(
                new_token_ids,
                new_block_ids,
                preempted=preempted,
                lmcache_cached_tokens=lmcache_cached_tokens,
                vllm_cached_tokens=vllm_cached_tokens,
                all_token_ids=all_token_ids,
            )

            req_meta = ReqMeta.from_request_tracker(
                request_tracker,
                self._block_size,
                self._lmcache_chunk_size,
                load_spec=load_spec,
                discard_partial_chunks=self._discard_partial_chunks,
                save_decode_cache=self.config.save_decode_cache,
            )
            if req_meta is not None:
                if self.sparse_kv_spec.enabled:
                    req_meta.sparse_kv_spec = self.sparse_kv_spec
                meta.add_request(req_meta)

        if _sparse_kv_debug_enabled():
            logger.info(
                "[lmcache-kv-iface] build_connector_meta requests=%d",
                len(meta.requests),
            )
        return meta

    @_lmcache_nvtx_annotate
    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        # Layerwise save uses request-scoped generators. If request finishes
        # without entering wait_for_save (abort/error/evict path), make sure
        # we release the generator entry to avoid leaking state.
        if getattr(self, "use_layerwise", False) and hasattr(
            self, "_layerwise_save_storers"
        ):
            self._layerwise_save_storers.pop(request.request_id, None)

        # Cleanup if request was aborted
        if request.status == RequestStatus.FINISHED_ABORTED:
            # Notify storage backends of aborted requests
            assert self.lmcache_engine is not None
            sm = self.lmcache_engine.storage_manager
            if sm is not None:
                sm.cancel_request(request.request_id)

            if self.async_loading:
                # Cancel any ongoing async lookup and prefetch tasks on workers
                lookup_id = request.request_id
                assert self.lookup_client is not None
                self.lookup_client.cancel_lookup(lookup_id)  # type: ignore[attr-defined]

        params = (
            request.kv_transfer_params
            if hasattr(request, "kv_transfer_params")
            else None
        )
        return_params = None

        # NOTE: Used to stream back the first token
        # for disagg prefill
        if params is not None and "ret_first_tok" in params:
            return_params = {
                "first_tok": request._output_token_ids[0],
            }

        if self.config.get_extra_config_value(
            "enable_cache_usage_details_in_response", False
        ):
            request_tracker = self._request_trackers.get(request.request_id)
            if request_tracker:
                return_params = return_params or {}
                return_params["num_lmcache_cached_tokens"] = (
                    request_tracker.num_lmcache_cached_tokens
                )

        return False, return_params

    @_lmcache_nvtx_annotate
    def get_kv_events(self) -> Iterable[CacheStoreEvent]:
        if self.lmcache_engine is not None:
            return self.lmcache_engine.get_kv_events()
        return []
