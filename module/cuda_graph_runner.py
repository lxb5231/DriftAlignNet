from contextlib import nullcontext

import torch
import torch.nn as nn


class CUDAGraphInferenceRunner:
    """Replay a fixed-shape model forward with the original CUDA kernels.

    The first CUDA batch warms up and captures the graph. Later batches with
    the same shape copy into static input buffers and replay it. A smaller
    final batch uses eager inference instead of retaining a second graph pool.
    CPU and DataParallel execution also remain eager.

    The returned CUDA tensor belongs to the graph and is overwritten by the
    next replay. Consumers must finish using or copy it before the next call.
    """

    def __init__(self, model, use_amp=False, warmup_steps=3):
        if warmup_steps < 1:
            raise ValueError("warmup_steps must be at least 1.")

        self.model = model
        self.use_amp = bool(use_amp)
        self.warmup_steps = int(warmup_steps)
        self._entry = None
        self._captured_batch_size = None

    @property
    def graph_captured(self):
        return self._entry is not None

    def clear(self):
        self._entry = None
        self._captured_batch_size = None

    def _autocast_context(self, x):
        if self.use_amp and x.is_cuda:
            return torch.autocast(device_type="cuda", enabled=True)
        return nullcontext()

    def _eager(self, x, phase):
        with torch.no_grad(), self._autocast_context(x):
            return self.model(x, phase)

    def _can_capture(self, x, phase):
        return (
            x.is_cuda
            and phase.is_cuda
            and torch.cuda.is_available()
            and not isinstance(self.model, nn.DataParallel)
        )

    @staticmethod
    def _input_signature(x, phase):
        return (
            x.device,
            x.dtype,
            tuple(x.shape),
            tuple(x.stride()),
            phase.dtype,
            tuple(phase.shape),
            tuple(phase.stride()),
        )

    def _capture(self, x, phase):
        static_x = torch.empty_like(x)
        static_phase = torch.empty_like(phase)
        static_x.copy_(x)
        static_phase.copy_(phase)

        current_stream = torch.cuda.current_stream(device=x.device)
        warmup_stream = torch.cuda.Stream(device=x.device)
        warmup_stream.wait_stream(current_stream)
        with (
            torch.cuda.stream(warmup_stream),
            torch.no_grad(),
            self._autocast_context(x),
        ):
            for _ in range(self.warmup_steps):
                eager_reference = self.model(static_x, static_phase)
            if not torch.is_tensor(eager_reference):
                raise TypeError("CUDA Graph inference requires a tensor output.")
            eager_reference = eager_reference.clone()
        current_stream.wait_stream(warmup_stream)

        graph = torch.cuda.CUDAGraph()
        with (
            torch.cuda.graph(graph),
            torch.no_grad(),
            self._autocast_context(x),
        ):
            static_output = self.model(static_x, static_phase)
        if not torch.is_tensor(static_output):
            raise TypeError("CUDA Graph inference requires a tensor output.")

        self._entry = {
            "signature": self._input_signature(x, phase),
            "x": static_x,
            "phase": static_phase,
            "output": static_output,
            "graph": graph,
        }
        self._captured_batch_size = x.size(0)

        # Capture records the operations but its output storage is not a valid
        # result for the current batch until the graph has been replayed.
        # Returning it directly makes the first test batch appear as zeros.
        graph.replay()
        if not torch.equal(static_output, eager_reference):
            raise RuntimeError(
                "CUDA Graph output differs from eager inference on the capture batch."
            )
        return static_output

    def __call__(self, x, phase):
        if self.model.training:
            raise RuntimeError("CUDA Graph inference requires model.eval().")
        if not self._can_capture(x, phase):
            return self._eager(x, phase)

        # Avoid a second graph allocation for a short final DataLoader batch.
        if (
            self._captured_batch_size is not None
            and x.size(0) != self._captured_batch_size
        ):
            return self._eager(x, phase)

        signature = self._input_signature(x, phase)
        if self._entry is None:
            return self._capture(x, phase)
        if signature != self._entry["signature"]:
            return self._eager(x, phase)

        self._entry["x"].copy_(x)
        self._entry["phase"].copy_(phase)
        self._entry["graph"].replay()
        return self._entry["output"]
