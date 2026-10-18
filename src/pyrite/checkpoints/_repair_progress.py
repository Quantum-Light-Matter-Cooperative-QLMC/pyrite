"""Save and progress cadence shared by the two checkpoint repair loops."""

import time


class _RepairProgress:
    def __init__(
        self,
        results,
        total,
        skipped,
        save_every,
        save_cb,
        on_progress,
        max_seconds,
        status,
        *,
        progress=True,
        saved_message=False,
    ):
        self.results = results
        self.total = total
        self.skipped = skipped
        self.save_every = save_every
        self.save_cb = save_cb
        self.on_progress = on_progress
        self.max_seconds = max_seconds
        self.status = status
        self.progress = progress
        self.saved_message = saved_message
        self.done = 0
        self.broke = False
        if on_progress is not None:
            on_progress(0, total, skipped)

    def start(self):
        self.t0 = time.perf_counter()
        self.deadline = None if self.max_seconds is None else time.monotonic() + self.max_seconds

    def expired(self):
        self.broke = self.deadline is not None and time.monotonic() >= self.deadline
        return self.broke

    def completed(self):
        self.done += 1
        k = self.done
        if self.on_progress is not None:
            self.on_progress(k, self.total, self.skipped)
        if self.save_cb is not None and self.save_every and k % self.save_every == 0:
            self.save_cb(self.results)
            if self.progress and self.saved_message:
                print(f"    ...saved progress ({k}/{self.total})")
        if self.progress and (k % 50 == 0 or k == self.total):
            print(f"  {k}/{self.total}  ({time.perf_counter() - self.t0:.0f} s)")

    def finish(self):
        # Empty passes historically report completion without saving.
        if self.total and self.save_cb is not None:
            self.save_cb(self.results)
        if self.status is not None:
            self.status["complete"] = not self.broke
        return self.done
