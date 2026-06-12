import sys
from contextlib import contextmanager


class Tee:
    """
    File-like object that mirrors writes to multiple streams
    while behaving like the primary stream (tty detection, encoding, etc.).
    """

    def __init__(self, *streams):
        if not streams:
            raise ValueError("Tee requires at least one stream")

        self._streams = streams
        self._primary = streams[0]

        # make libraries like tqdm happy
        # (some streams, e.g. io.StringIO, expose encoding=None — fall back too)
        self.encoding = getattr(self._primary, "encoding", None) or "utf-8"

    # ---- core API ----

    def write(self, data):
        primary_n = None
        for s in self._streams:
            n = s.write(data)
            if s is self._primary:
                primary_n = n
        self.flush()
        # File protocol: report chars written (primary stream's count)
        return primary_n if primary_n is not None else len(data)

    def flush(self):
        for s in self._streams:
            s.flush()

    # ---- terminal compatibility ----

    def isatty(self):
        return getattr(self._primary, "isatty", lambda: False)()

    def fileno(self):
        return getattr(self._primary, "fileno", lambda: 1)()

    # ---- transparent passthrough ----

    def __getattr__(self, name):
        return getattr(self._primary, name)


# ------------------------------------------------------------------
# Context managers
# ------------------------------------------------------------------

@contextmanager
def TeeStdout(log_path, mode="w", encoding="utf-8"):
    f = open(log_path, mode, encoding=encoding)
    old = sys.stdout
    sys.stdout = Tee(old, f)
    try:
        yield
    finally:
        sys.stdout.flush()
        sys.stdout = old
        f.close()


@contextmanager
def TeeStderr(log_path, mode="w", encoding="utf-8"):
    f = open(log_path, mode, encoding=encoding)
    old = sys.stderr
    sys.stderr = Tee(old, f)
    try:
        yield
    finally:
        sys.stderr.flush()
        sys.stderr = old
        f.close()
