# SPDX-License-Identifier: Apache-2.0
"""Bounded next-layer prefetch for the single-request staged prefill path.

One CPU worker runs the existing blocking GDS reader in a CUDA device/stream
context. It never synchronizes the compute stream. A future certifies read
completion; a recorded compute event certifies that a buffer may be reused.
The synchronous PrefillLayerStager remains the default, unchanged path.
"""

from concurrent.futures import CancelledError, ThreadPoolExecutor
from dataclasses import dataclass
import time

from lmcache.integration.vllm.aissd_prefill_stage import PrefillLayerStager


@dataclass
class _Slot:
    buffer: object
    available: object
    future: object = None
    layer: object = None


class PrefetchingPrefillLayerStager(PrefillLayerStager):
    """Two workspaces, one in-flight layer read, one immutable step plan.

    begin_prefetch_step() is called before the model forward. For each layer,
    stage()/paged_view() are followed by finish_layer() AFTER attention has
    been enqueued on the original compute stream. finish_prefetch_step() or
    abort_prefetch_step() must run before changing requests/plans or building
    PQ. close() must run before shutting down the GDS backend.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._slots = []
        self._io_stream = None
        self._executor = None
        self._prefetch_step = None
        self._armed_layer = None
        self._last_prefetch_stats = None
        self._closed = False

    @property
    def allocated_bytes(self):
        return len(self._slots) * self.workspace_bytes

    def _check_compute_stream(self):
        import torch

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("AISSD prefill prefetch requires eager CUDA")
        stream = torch.cuda.current_stream(self._buffer.device)
        if stream != self._stream:
            raise RuntimeError("AISSD prefill prefetch changed compute streams")
        return stream

    def begin_prefetch_step(self, backend, archive, device, *, req_id, computed):
        import torch

        if self._closed or self._prefetch_step is not None:
            raise RuntimeError("AISSD prefill prefetch step was not drained")
        archive = tuple(archive)
        self._last_prefetch_stats = None
        self._armed_layer = None
        step = {
            "backend": backend, "archive": archive, "req_id": str(req_id),
            "computed": int(computed), "next_layer": 0,
            "started": time.perf_counter(), "submitted_layers": 0,
            "staged_layers": 0, "ready_on_arrival": 0,
            "read_ops": 0, "read_bytes": 0, "io_ms": 0.0,
            "reuse_wait_ms": 0.0, "host_wait_ms": 0.0,
            "first_layer_wait_ms": 0.0,
        }
        # The first cold segment has no historical KV. Do not allocate a
        # staging buffer, launch a thread, or manufacture prefetch work.
        if archive:
            if not self._slots:
                buffer, stream = self._workspace(device)
                buffers = [buffer]
                if self.num_layers > 1:
                    buffers.append(torch.empty_like(buffer))
                # Torch may reuse an allocation with pending stream work.
                # External DMA must wait for that work, even on first use.
                allocated = torch.cuda.Event()
                allocated.record(stream)
                self._slots = [_Slot(b, allocated) for b in buffers]
                self._io_stream = torch.cuda.Stream(device=buffer.device)
            else:
                # Configs may say "cuda" while tensors say "cuda:0".
                # Resolve the index through the stream, as the base does.
                stream = torch.cuda.current_stream(device)
                if self._buffer.device != stream.device:
                    raise RuntimeError("AISSD prefill prefetch changed devices")
                self._check_compute_stream()
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="aissd-prefill-read")
        self._prefetch_step = step
        if archive:
            try:
                self._submit_layer(0)
            except BaseException:
                self.abort_prefetch_step()
                raise

    def _submit_layer(self, layer):
        step = self._prefetch_step
        slot = self._slots[layer % len(self._slots)]
        if slot.future is not None:
            raise RuntimeError("AISSD prefill prefetch queue exceeded one layer")
        slot.layer = layer
        # Everything needed by the job is captured now. No worker looks up a
        # mutable request, archive plan, active slot, or current CUDA stream.
        slot.future = self._executor.submit(
            self._read_job, step["backend"], step["archive"], layer,
            slot.buffer, slot.available, self._io_stream)
        step["submitted_layers"] += 1

    def _read_job(self, backend, archive, layer, buffer, available, io_stream):
        import torch

        with torch.inference_mode(), torch.cuda.device(buffer.device), torch.cuda.stream(io_stream):
            started = time.perf_counter()
            available.synchronize()
            reuse_wait_ms = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            # Reuse the validated byte layout and exact blocking reader. Its
            # synchronization applies ONLY to the worker's I/O stream here.
            PrefillLayerStager._read_archive(
                self, backend, archive, layer, buffer, io_stream, (0, 1))
            io_stream.synchronize()
            return {
                "io_ms": (time.perf_counter() - started) * 1000,
                "reuse_wait_ms": reuse_wait_ms,
                "read_ops": len(archive) * 2,
                "read_bytes": sum(item.plane_bytes for item in archive) * 2,
            }

    def stage(self, backend, archive, layer, cache, local_block_ids, *, local_tokens,
              restore_tokens=0):
        step = self._prefetch_step
        if (step is None or not step["archive"] or self._armed_layer is not None
                or layer != step["next_layer"] or not 0 <= layer < self.num_layers
                or backend is not step["backend"] or tuple(archive) != step["archive"]):
            raise RuntimeError("AISSD prefill prefetch request/plan/layer mismatch")
        self._buffer = self._slots[layer % len(self._slots)].buffer
        self._armed_layer = layer
        # All geometry checks, warm-tail restoration, and local-page gathers
        # use the existing implementation. Only _read_archive is substituted.
        return super().stage(
            backend, archive, layer, cache, local_block_ids,
            local_tokens=local_tokens, restore_tokens=restore_tokens)

    def _read_archive(self, backend, archive, layer, buffer, stream, planes):
        step = self._prefetch_step
        if step is None:
            # PQ construction after prefill deliberately keeps its existing
            # synchronous K-only path and the same tail-token correction.
            return super()._read_archive(backend, archive, layer, buffer, stream, planes)
        if (planes != (0, 1) or self._armed_layer != layer
                or backend is not step["backend"] or tuple(archive) != step["archive"]):
            raise RuntimeError("AISSD prefill prefetch consumed an unexpected read")
        slot = self._slots[layer % len(self._slots)]
        if slot.buffer is not buffer or slot.layer != layer or slot.future is None:
            raise RuntimeError("AISSD prefill prefetch buffer is not ready for this layer")
        self._check_compute_stream()
        future = slot.future
        was_ready = future.done()
        started = time.perf_counter()
        try:
            result = future.result()
        finally:
            waited = (time.perf_counter() - started) * 1000
            step["host_wait_ms"] += waited
            if layer == 0:
                step["first_layer_wait_ms"] += waited
        # A submission is not a completion: publish readiness only after the
        # worker has checked every byte count and synchronized its I/O stream.
        slot.future = None
        slot.layer = None
        step["ready_on_arrival"] += int(was_ready)
        step["staged_layers"] += 1
        for name, value in result.items():
            step[name] += value

    def finish_layer(self, layer):
        import torch

        step = self._prefetch_step
        if (step is None or self._armed_layer != layer or layer != step["next_layer"]):
            raise RuntimeError("AISSD prefill prefetch attention was not staged")
        stream = self._check_compute_stream()
        slot = self._slots[layer % len(self._slots)]
        consumed = torch.cuda.Event()
        consumed.record(stream)
        slot.available = consumed
        self._armed_layer = None
        step["next_layer"] += 1
        if step["next_layer"] < self.num_layers:
            self._submit_layer(step["next_layer"])

    def forward(self, attention, backend, archive, layer, cache, local_block_ids,
                *, query, local_tokens, softmax_scale):
        try:
            result = super().forward(
                attention, backend, archive, layer, cache, local_block_ids,
                query=query, local_tokens=local_tokens, softmax_scale=softmax_scale)
            self.finish_layer(layer)
            return result
        except BaseException:
            self.abort_prefetch_step()
            raise

    def _drain(self, *, cancel):
        import torch

        error = None
        # An exception may occur after stage() has enqueued a local gather or
        # some attention work, before finish_layer() could record its fence.
        if self._armed_layer is not None:
            slot = self._slots[self._armed_layer % len(self._slots)]
            consumed = torch.cuda.Event()
            consumed.record(self._check_compute_stream())
            slot.available = consumed
        for slot in self._slots:
            future = slot.future
            if future is not None:
                if cancel:
                    future.cancel()
                while True:
                    try:
                        future.result()
                        break
                    except CancelledError:
                        break
                    except BaseException as exc:
                        if error is None:
                            error = exc
                        if future.done():
                            break
                        # An interrupt on the waiting CPU thread is not an
                        # I/O completion. Keep owning the allocation until
                        # the background DMA/read has actually stopped.
                slot.future = None
                slot.layer = None
        # No CPU task can still write these buffers. Fence their last GPU use
        # before another step/request/PQ builder or the allocator may reuse them.
        for slot in self._slots:
            try:
                slot.available.synchronize()
            except BaseException as exc:
                if error is None:
                    error = exc
        if error is not None:
            raise error

    def finish_prefetch_step(self):
        step = self._prefetch_step
        if step is None:
            return self._last_prefetch_stats
        if step["archive"] and (self._armed_layer is not None
                                or step["next_layer"] != self.num_layers):
            self.abort_prefetch_step()
            raise RuntimeError("AISSD prefill prefetch did not finish all layers")
        try:
            self._drain(cancel=False)
            names = ("submitted_layers", "staged_layers", "ready_on_arrival",
                     "read_ops", "read_bytes", "io_ms", "reuse_wait_ms",
                     "host_wait_ms", "first_layer_wait_ms")
            stats = {name: step[name] for name in names}
            stats.update(
                prefetched_layers=max(0, step["submitted_layers"] - 1),
                workspace_bytes=self.allocated_bytes,
                buffers=len(self._slots),
                wall_ms=(time.perf_counter() - step["started"]) * 1000)
            self._last_prefetch_stats = stats
            return stats
        finally:
            self._prefetch_step = None
            self._armed_layer = None

    def abort_prefetch_step(self):
        if self._prefetch_step is None:
            return
        try:
            self._drain(cancel=True)
        finally:
            self._prefetch_step = None
            self._armed_layer = None
            self._last_prefetch_stats = None

    def close(self):
        if self._closed:
            return
        try:
            if self._prefetch_step is not None:
                self.abort_prefetch_step()
            else:
                self._drain(cancel=True)
        finally:
            if self._executor is not None:
                self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
            self._slots.clear()
            self._buffer = None
            self._io_stream = None
            self._last_prefetch_stats = None
            self._closed = True
