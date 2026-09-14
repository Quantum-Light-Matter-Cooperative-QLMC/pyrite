"""Cross-platform nonblocking keyboard polling for live dashboards."""

import queue
import select
import sys
import threading


class KeyListener:
    """Background nonblocking single-keypress capture for dashboard loops."""

    def __init__(self):
        self.active = False
        self._keys = queue.Queue()
        self._stop_event = threading.Event()
        self._restore = None
        self._thread = None
        if not sys.stdin.isatty():
            return
        try:
            import msvcrt  # noqa: F401 -- Windows only; ImportError selects POSIX
        except ImportError:
            try:
                import termios
                import tty

                fd = sys.stdin.fileno()
                old = termios.tcgetattr(fd)
                tty.setcbreak(fd)
            except (OSError, ValueError, termios.error):
                return
            self._restore = lambda: termios.tcsetattr(fd, termios.TCSADRAIN, old)
            self._thread = threading.Thread(target=self._poll_posix, args=(fd,), daemon=True)
        else:
            self._thread = threading.Thread(target=self._poll_windows, daemon=True)
        self._thread.start()
        self.active = True

    def _poll_windows(self):
        import msvcrt

        while not self._stop_event.is_set():
            if msvcrt.kbhit():  # ty: ignore[unresolved-attribute]
                self._keys.put(msvcrt.getwch())  # ty: ignore[unresolved-attribute]
            else:
                self._stop_event.wait(0.1)

    def _poll_posix(self, fd):
        while not self._stop_event.is_set():
            ready, _write, _exceptional = select.select([fd], [], [], 0.1)
            if not ready:
                continue
            char = sys.stdin.read(1)
            if char == "":
                break
            self._keys.put(char)

    def poll(self):
        """Return buffered keys oldest first."""
        keys = []
        while True:
            try:
                keys.append(self._keys.get_nowait())
            except queue.Empty:
                break
        return keys

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        if self._restore is not None:
            self._restore()
