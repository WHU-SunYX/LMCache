# SPDX-License-Identifier: Apache-2.0
"""Request-scoped IndexCache policy parsing and snapshotting.

The policy is immutable for the lifetime of one vLLM request.  New requests may
use a different policy without restarting the engine.  The worker currently
requires all requests that share one model-execution batch to use the same
layer layout; the greedy-search runner therefore uses homogeneous batches.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


_REQUEST_FULL_LAYER_KEYS = (
    "lmcache.aissd_f_layers",
    "lmcache.indexcache_full_layers",
    "aissd_f_layers",
    "indexcache_full_layers",
)
_REQUEST_PATTERN_KEYS = (
    "lmcache.aissd_layer_pattern",
    "lmcache.indexcache_pattern",
    "aissd_layer_pattern",
    "indexcache_pattern",
)
_REQUEST_GENERATION_KEYS = (
    "lmcache.aissd_pattern_generation",
    "lmcache.indexcache_generation",
    "aissd_pattern_generation",
    "indexcache_generation",
)


def _mapping_get(mapping: Optional[Mapping[str, Any]], keys: Sequence[str]) -> Any:
    if not mapping:
        return None
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _parse_int(value: Any, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def parse_full_layers(value: Any, num_layers: int) -> tuple[int, ...]:
    """Parse a 0-based Full-layer list and validate IndexCache invariants."""
    if isinstance(value, str):
        items: Sequence[Any] = [part.strip() for part in value.replace(";", ",").split(",")]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        items = value
    else:
        raise ValueError(
            "full_layers must be a comma-separated string or a sequence of integers"
        )

    layers: list[int] = []
    for item in items:
        if item is None or (isinstance(item, str) and not item.strip()):
            continue
        layer = _parse_int(item, "full layer")
        if layer < 0 or layer >= num_layers:
            raise ValueError(
                f"full layer {layer} is outside [0, {num_layers - 1}]"
            )
        layers.append(layer)

    unique = tuple(sorted(set(layers)))
    if not unique:
        raise ValueError("at least one Full layer is required")
    if unique[0] != 0:
        raise ValueError("layer 0 must be Full so Shared layers have a predecessor")
    return unique


def parse_pattern(value: Any, num_layers: int) -> tuple[int, ...]:
    """Parse an F/S pattern such as ``FSSFS`` and return Full-layer IDs."""
    if not isinstance(value, str):
        raise ValueError(f"pattern must be a string, got {type(value).__name__}")
    pattern = "".join(ch for ch in value.upper() if not ch.isspace() and ch not in ",;")
    if len(pattern) != num_layers:
        raise ValueError(
            f"pattern length {len(pattern)} does not match num_layers={num_layers}"
        )
    invalid = sorted(set(pattern) - {"F", "S"})
    if invalid:
        raise ValueError(f"pattern contains invalid symbols: {invalid}")
    return parse_full_layers([i for i, role in enumerate(pattern) if role == "F"], num_layers)


def build_source_layers(full_layers: Sequence[int], num_layers: int) -> tuple[int, ...]:
    full = set(parse_full_layers(full_layers, num_layers))
    source: list[int] = []
    last_full = -1
    for layer in range(num_layers):
        if layer in full:
            last_full = layer
        if last_full < 0:
            raise ValueError("layer 0 must be Full")
        source.append(last_full)
    return tuple(source)


def _stable_generation(num_layers: int, full_layers: Sequence[int], origin: str) -> int:
    payload = json.dumps(
        {
            "num_layers": int(num_layers),
            "full_layers": [int(x) for x in full_layers],
            "origin": str(origin),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    # Keep the value within signed int64 for tensor/log compatibility.
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") & ((1 << 63) - 1)


@dataclass(frozen=True)
class IndexCachePolicy:
    num_layers: int
    full_layers: tuple[int, ...]
    source_layers: tuple[int, ...]
    generation: int
    origin: str

    @classmethod
    def from_full_layers(
        cls,
        *,
        num_layers: int,
        full_layers: Any,
        generation: Optional[int] = None,
        origin: str = "unknown",
    ) -> "IndexCachePolicy":
        parsed = parse_full_layers(full_layers, num_layers)
        source = build_source_layers(parsed, num_layers)
        gen = (
            _stable_generation(num_layers, parsed, origin)
            if generation is None
            else _parse_int(generation, "policy generation")
        )
        if gen < 0:
            raise ValueError("policy generation must be non-negative")
        return cls(
            num_layers=int(num_layers),
            full_layers=parsed,
            source_layers=source,
            generation=gen,
            origin=str(origin),
        )

    @classmethod
    def from_pattern(
        cls,
        *,
        num_layers: int,
        pattern: str,
        generation: Optional[int] = None,
        origin: str = "unknown",
    ) -> "IndexCachePolicy":
        return cls.from_full_layers(
            num_layers=num_layers,
            full_layers=parse_pattern(pattern, num_layers),
            generation=generation,
            origin=origin,
        )

    @property
    def pattern(self) -> str:
        full = set(self.full_layers)
        return "".join("F" if layer in full else "S" for layer in range(self.num_layers))

    def is_full(self, layer_id: int) -> bool:
        return int(layer_id) in self.full_layers

    def source_layer(self, layer_id: int) -> int:
        layer = int(layer_id)
        if layer < 0 or layer >= self.num_layers:
            raise IndexError(layer)
        return self.source_layers[layer]

    def same_layout(self, other: "IndexCachePolicy") -> bool:
        return self.num_layers == other.num_layers and self.full_layers == other.full_layers

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_layers": self.num_layers,
            "full_layers": list(self.full_layers),
            "source_layers": list(self.source_layers),
            "pattern": self.pattern,
            "generation": self.generation,
            "origin": self.origin,
        }


class IndexCachePolicyLoader:
    """Resolve one immutable policy snapshot for each new request.

    Precedence, highest first:
      1. Request ``SamplingParams.extra_args.kv_transfer_params`` entries.
      2. JSON policy file from ``AISSD_INDEXCACHE_POLICY_FILE``.
      3. ``AISSD_F_LAYERS`` environment variable.
      4. Constructor default.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        default_full_layers: Optional[Sequence[int]] = None,
        policy_file: Optional[str] = None,
    ) -> None:
        self.num_layers = int(num_layers)
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        default = (
            tuple(range(self.num_layers))
            if default_full_layers is None
            else parse_full_layers(default_full_layers, self.num_layers)
        )
        self._default_policy = IndexCachePolicy.from_full_layers(
            num_layers=self.num_layers,
            full_layers=default,
            generation=0,
            origin="constructor-default",
        )
        self.policy_file = policy_file or os.environ.get("AISSD_INDEXCACHE_POLICY_FILE")
        self._cached_file_stat: Optional[tuple[int, int]] = None
        self._cached_file_policy: Optional[IndexCachePolicy] = None

    def _from_mapping(self, mapping: Mapping[str, Any], origin: str) -> Optional[IndexCachePolicy]:
        full_layers = _mapping_get(mapping, _REQUEST_FULL_LAYER_KEYS)
        pattern = _mapping_get(mapping, _REQUEST_PATTERN_KEYS)
        generation = _mapping_get(mapping, _REQUEST_GENERATION_KEYS)
        if full_layers is None and pattern is None:
            # Policy-file schema uses unprefixed canonical names.
            full_layers = mapping.get("full_layers")
            pattern = mapping.get("pattern")
            generation = mapping.get("generation", generation)
        if full_layers is not None and pattern is not None:
            raise ValueError(f"{origin} specifies both full_layers and pattern")
        if full_layers is not None:
            return IndexCachePolicy.from_full_layers(
                num_layers=self.num_layers,
                full_layers=full_layers,
                generation=generation,
                origin=origin,
            )
        if pattern is not None:
            return IndexCachePolicy.from_pattern(
                num_layers=self.num_layers,
                pattern=str(pattern),
                generation=generation,
                origin=origin,
            )
        return None

    def _load_file_policy(self) -> Optional[IndexCachePolicy]:
        if not self.policy_file:
            return None
        path = Path(self.policy_file)
        try:
            stat = path.stat()
        except FileNotFoundError:
            return None
        signature = (int(stat.st_mtime_ns), int(stat.st_size))
        if self._cached_file_stat == signature and self._cached_file_policy is not None:
            return self._cached_file_policy
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, Mapping):
            raise ValueError(f"policy file {path} must contain a JSON object")
        file_num_layers = payload.get("num_layers")
        if file_num_layers is not None and int(file_num_layers) != self.num_layers:
            raise ValueError(
                f"policy file num_layers={file_num_layers} does not match model {self.num_layers}"
            )
        policy = self._from_mapping(payload, f"file:{path}")
        if policy is None:
            raise ValueError(f"policy file {path} has neither full_layers nor pattern")
        self._cached_file_stat = signature
        self._cached_file_policy = policy
        return policy

    def snapshot(
        self,
        request_configs: Optional[Mapping[str, Any]] = None,
        *,
        reuse_enabled: bool = True,
    ) -> IndexCachePolicy:
        if not reuse_enabled:
            return IndexCachePolicy.from_full_layers(
                num_layers=self.num_layers,
                full_layers=range(self.num_layers),
                generation=0,
                origin="layer-reuse-disabled",
            )

        if request_configs:
            request_policy = self._from_mapping(request_configs, "request")
            if request_policy is not None:
                return request_policy

        file_policy = self._load_file_policy()
        if file_policy is not None:
            return file_policy

        env_layers = os.environ.get("AISSD_F_LAYERS")
        if env_layers:
            return IndexCachePolicy.from_full_layers(
                num_layers=self.num_layers,
                full_layers=env_layers,
                generation=None,
                origin="env:AISSD_F_LAYERS",
            )
        return self._default_policy
