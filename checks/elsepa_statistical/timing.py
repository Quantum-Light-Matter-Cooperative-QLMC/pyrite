"""Optional CUDA event timing for both #895 study arms."""

from functools import cache


class KernelEvents:
    def __init__(self):
        from pyrite.montecarlo.transport import _jit_launch

        self.events = []
        original = _jit_launch.transport_kernel_variant

        @cache
        def variant(switches, precision="fp64"):
            if switches.mixed_on:
                raise ValueError("node-mixture study excludes mixed elastic transport")
            return Timed(original(switches, precision), self.events)

        _jit_launch.transport_kernel_variant = variant

    def clear(self):
        self.events.clear()

    def summary(self):
        import cupy

        # The request synchronized before stopping its wall timer. Reading
        # elapsed event times here adds no synchronization to that timer.
        return {
            "kernel_ms": sum(cupy.cuda.get_elapsed_time(a, b) for a, b in self.events),
            "kernel_launches": len(self.events),
        }


class Timed:
    def __init__(self, inner, sink):
        self.inner, self.sink = inner, sink

    def __call__(self, grid, block, args, shared_mem=0):
        import cupy

        start, stop = cupy.cuda.Event(), cupy.cuda.Event()
        start.record()
        self.inner(grid, block, args, shared_mem)
        stop.record()
        self.sink.append((start, stop))

    def __getattr__(self, name):
        return getattr(self.inner, name)
