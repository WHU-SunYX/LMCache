# SPDX-License-Identifier: Apache-2.0
"""Dependency-free cache policy and raw KV layout checks."""

from collections import OrderedDict
import time


def wait_for_archive(chunks, contains, check_error, timeout, *, clock=time.monotonic,
                     pause=time.sleep):
    if timeout <= 0:
        raise ValueError("Archive timeout must be positive")
    deadline = clock() + timeout
    pending = list(chunks)
    while pending:
        check_error()
        pending = [chunk for chunk in pending if not contains(chunk[0])]
        if not pending:
            return
        if clock() >= deadline:
            raise TimeoutError(
                f"AISSD resident archive not committed: {len(pending)} chunks pending; "
                "GPU blocks were not released"
            )
        pause(min(0.01, max(0.0, deadline - clock())))


def layer_read_layout(manifest, layer, layers, tokens, hidden, element_size=2):
    expected = [2, layers, tokens, hidden]
    if list(manifest.get("shape") or []) != expected:
        raise ValueError(f"Resident raw KV layout must be {expected}, got {manifest.get('shape')}")
    fmt = str(manifest.get("fmt"))
    if fmt not in ("MemoryFormat.KV_2LTD", "KV_2LTD", "1"):
        raise ValueError(f"Resident raw KV requires KV_2LTD, got {fmt}")
    if str(manifest.get("dtype")) not in ("torch.bfloat16", "bfloat16") or element_size != 2:
        raise ValueError("Resident layer reads require BF16 KV")
    if not 0 <= layer < layers or manifest.get("num_tokens") != tokens:
        raise ValueError("Invalid resident layer or incomplete chunk")
    size = tokens * hidden * element_size
    base = int(manifest.get("file_offset", 4096))
    if base < 0 or int(manifest.get("nbytes", -1)) < 2 * layers * size:
        raise ValueError("Resident raw KV extent is truncated")
    return base + layer * size, base + (layers + layer) * size, size


class LayerLRU:
    """One layer's slot validity. All selected chunks are pinned during refill.

    A failed refill removes the previous owner *before* the first byte is
    overwritten; it can never leave stale KV marked as a cache hit.
    """

    def __init__(self, capacity, bootstrap=True):
        self.capacity = capacity
        self.entries = OrderedDict((i, i) for i in range(capacity)) if bootstrap else OrderedDict()

    def resize(self, capacity):
        if capacity > self.capacity or capacity < 0:
            raise ValueError("Resident hot pool may only shrink within an epoch")
        self.entries = OrderedDict((k, slot) for k, slot in self.entries.items() if slot < capacity)
        self.capacity = capacity

    def resolve(self, selected, load):
        selected = list(selected)
        if not selected or len(set(selected)) != len(selected) or len(selected) > self.capacity:
            raise ValueError("Invalid selected set or insufficient resident hot capacity")
        pinned = set(selected)
        hits = sum(k in self.entries for k in selected)
        loaded_bytes = 0
        evictions = 0
        for chunk in selected:
            if chunk in self.entries:
                continue
            used = set(self.entries.values())
            free = next((i for i in range(self.capacity) if i not in used), None)
            if free is None:
                victim = next((k for k in self.entries if k not in pinned), None)
                if victim is None:
                    raise RuntimeError("Resident LRU has no unpinned victim")
                free = self.entries.pop(victim)
                evictions += 1
            loaded_bytes += load(chunk, free)
            self.entries[chunk] = free
        for chunk in selected:
            self.entries.move_to_end(chunk)
        # Preserve the existing native selector's order, including after its
        # candidate capping/reordering. Do not change FA reduction order here.
        return [(chunk, self.entries[chunk]) for chunk in selected], {
            "hits": hits, "misses": len(selected) - hits,
            "evictions": evictions, "load_bytes": loaded_bytes,
        }
