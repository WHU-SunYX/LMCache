# SPDX-License-Identifier: Apache-2.0
"""Single-layer BF16 staging for exact dense AISSD prefill.

The caller owns logical positions, archive durability, local page lifetimes,
and the FlashAttention call. This primitive does not modify the scheduler.
"""

import ctypes
from dataclasses import dataclass
from operator import index
from os import fspath

from lmcache.integration.vllm.aissd_page_access import (
    gather_kv_pages_into,
    make_page_index,
    page_major,
)
from lmcache.integration.vllm.resident_cache import layer_read_layout


@dataclass(frozen=True, slots=True)
class _ChunkRead:
    path: str
    token_start: int
    k0: int
    v0: int
    plane_bytes: int


class PrefillLayerStager:
    """Reuse one [2, max_context, heads, dim] workspace across model layers.

    Archived chunks must be immutable and in logical token order, before all
    live local KV. The caller must enqueue attention on the same CUDA stream
    before staging the next layer. Contexts with a partially cached last chunk
    must restore that partial history to the local pages before calling.
    """

    def __init__(self, *, max_tokens, num_layers, chunk_size, block_size,
                 heads, head_dim):
        (self.max_tokens, self.num_layers, self.chunk_size, self.block_size,
         self.heads, self.head_dim) = map(index, (
            max_tokens, num_layers, chunk_size, block_size, heads, head_dim))
        if min(self.max_tokens, self.num_layers, self.chunk_size,
               self.block_size, self.heads, self.head_dim) <= 0:
            raise ValueError("Invalid AISSD prefill staging geometry")
        if self.chunk_size % self.block_size:
            raise ValueError("AISSD archive chunk must contain complete GPU pages")
        self.capacity = self._round_up(self.max_tokens)
        self.token_bytes = self.heads * self.head_dim * 2  # BF16, one plane
        self._buffer = None
        self._stream = None

    @property
    def workspace_bytes(self):
        return 2 * self.capacity * self.token_bytes

    @property
    def allocated_bytes(self):
        return 0 if self._buffer is None else self.workspace_bytes

    def _round_up(self, n):
        return (n + self.block_size - 1) // self.block_size * self.block_size

    def plan_archive(self, manifests, *, prefix_tokens):
        """Validate the committed full-chunk prefix once per scheduler step."""
        prefix_tokens = index(prefix_tokens)
        if not 0 <= prefix_tokens <= self.max_tokens or prefix_tokens % self.chunk_size:
            raise ValueError("AISSD staged prefill prefix length must be chunk-aligned")
        result = []
        for ordinal, manifest in enumerate(manifests):
            start = ordinal * self.chunk_size
            end = start + self.chunk_size
            if (manifest.get("token_start") != start
                    or manifest.get("token_end") != end or end > prefix_tokens):
                raise ValueError("AISSD staged prefill prefix has a gap, overlap or overflow")
            k0, v0, size = layer_read_layout(
                manifest, 0, self.num_layers, self.chunk_size, self.heads * self.head_dim)
            path = fspath(manifest["path"])
            if not isinstance(path, str) or not path:
                raise ValueError("AISSD staged prefill archive path is missing")
            result.append(_ChunkRead(path, start, k0, v0, size))
        if len(result) * self.chunk_size != prefix_tokens:
            raise ValueError("AISSD staged prefill prefix is missing committed chunks")
        return tuple(result)

    def _workspace(self, device):
        import torch

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("AISSD staged prefill requires eager CUDA")
        stream = torch.cuda.current_stream(device)
        device = stream.device
        if self._buffer is None:
            self._buffer = torch.empty((2, self.capacity, self.heads, self.head_dim),
                                       dtype=torch.bfloat16, device=device)
            self._stream = stream
        if self._buffer.device != device or stream != self._stream:
            raise RuntimeError("AISSD staged prefill workspace must stay on its original CUDA stream")
        return self._buffer, stream

    def _read_archive(self, backend, archive, layer, buffer, stream, planes):
        if not archive:
            return
        # The previous layer may still be consuming the staging buffer.
        # _load_gds is a blocking read outside the Torch stream.
        stream.synchronize()
        bases = tuple(buffer[p].data_ptr() for p in planes)
        for item in archive:
            for plane, base in zip(planes, bases):
                offset = (item.k0, item.v0)[plane] + layer * item.plane_bytes
                addr = ctypes.c_void_p(base + item.token_start * self.token_bytes)
                actual = backend._load_gds(item.path, offset, addr, item.plane_bytes, 0)
                if actual != item.plane_bytes:
                    raise IOError(f"Short staged prefill layer read: {actual}/{item.plane_bytes} "
                                  f"bytes from {item.path}")

    def archive_key(self, backend, archive, layer, device):
        """Use the same buffer to build PQ from archived K after the last step."""
        if not 0 <= layer < self.num_layers or not archive:
            raise ValueError("Invalid AISSD PQ archive layer")
        buffer, stream = self._workspace(device)
        self._read_archive(backend, archive, layer, buffer, stream, (0,))
        return buffer[0, :len(archive) * self.chunk_size]

    def paged_view(self, total_tokens, *, pages_first):
        """Expose a dense page table's backing storage to native paged FA."""
        if self._buffer is None or not 0 < total_tokens <= self.max_tokens:
            raise ValueError("Invalid AISSD staged FA cache length")
        padded = self._round_up(total_tokens)
        pages = self._buffer[:, :padded].view(
            2, padded // self.block_size, self.block_size, self.heads, self.head_dim)
        return pages.transpose(0, 1) if pages_first else pages

    def stage(self, backend, archive, layer, cache, local_block_ids, *, local_tokens,
              restore_tokens=0):
        """Read a layer's committed prefix and gather its live local pages.

        Returns borrowed contiguous [tokens, heads, dim] K and V views. A
        short read aborts before a caller can submit attention on stale data.
        """
        layer, local_tokens = index(layer), index(local_tokens)
        restore_tokens = index(restore_tokens)
        prefix = len(archive) * self.chunk_size
        if restore_tokens:
            if (not archive or restore_tokens != self.chunk_size - 1
                    or local_tokens != self.chunk_size):
                raise ValueError("AISSD can restore only a full-hit chunk minus its recomputed token")
            prefix -= self.chunk_size
        total = prefix + local_tokens
        if not 0 <= layer < self.num_layers or local_tokens <= 0 or total > self.max_tokens:
            raise ValueError("Invalid AISSD staged prefill layer or token length")
        if not cache.is_cuda:
            raise RuntimeError("AISSD staged prefill requires eager CUDA")
        paged = page_major(cache, self.block_size, self.heads, self.head_dim)
        local_padded = self._round_up(local_tokens)
        ids = make_page_index(local_block_ids, paged.shape[0], cache.device,
                              expected=local_padded // self.block_size)
        buffer, stream = self._workspace(cache.device)
        self._read_archive(backend, archive, layer, buffer, stream, (0, 1))
        if restore_tokens:
            # The live last token has already been written by vLLM's KV update.
            # Restore only the earlier positions of its archived chunk.
            for first in range(0, restore_tokens, self.block_size):
                end = min(restore_tokens, first + self.block_size)
                page = local_block_ids[first // self.block_size]
                paged[page, :, :end - first].copy_(buffer[:, prefix + first:prefix + end])
        gather_kv_pages_into(paged, ids, buffer[:, prefix:prefix + local_padded])
        return buffer[0, :total], buffer[1, :total]

    def forward(self, attention, backend, archive, layer, cache, local_block_ids,
                *, query, local_tokens, softmax_scale):
        """Run one bottom-right causal attention over the complete logical KV.

        ``attention(q, k, v, *, causal, softmax_scale)`` binds the installed
        FlashAttention wrapper. The query is the last Q tokens of the local
        window. No GPU/SSD selector or PQ approximation is used here.
        """
        if (query.ndim != 3 or not 0 < query.shape[0] <= local_tokens
                or query.shape[1] < self.heads or query.shape[1] % self.heads
                or query.shape[2] != self.head_dim or query.device != cache.device
                or query.dtype != cache.dtype):
            raise ValueError("AISSD staged prefill requires a compatible local query suffix")
        key, value = self.stage(backend, archive, layer, cache, local_block_ids,
                                local_tokens=local_tokens)
        return attention(query, key, value, causal=True, softmax_scale=softmax_scale)
