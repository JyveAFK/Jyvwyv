"""--log FILE support: copy everything printed to the console into a text file.

The file starts with the full command line, working directory and versions,
then every console line (progress, warnings and errors included), then how the
run ended. Console output is unchanged.
"""

import atexit
import os
import platform
import sys
import time
import traceback


class _Tee:
    def __init__(self, stream, f):
        self.stream = stream
        self.f = f

    def write(self, s):
        self.stream.write(s)
        try:
            self.f.write(s)
            self.f.flush()
        except ValueError:          # file already closed at shutdown
            pass
        return len(s)

    def flush(self):
        self.stream.flush()

    def isatty(self):
        return self.stream.isatty()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def start(path):
    """Begin copying console output to path (overwritten). No-op for an empty path."""
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    f = open(path, "w", encoding="utf-8", buffering=1)
    t0 = time.time()
    f.write(f"# started {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"# folder  {os.getcwd()}\n")
    f.write(f"# command python {' '.join(_quote(x) for x in sys.argv)}\n")
    try:
        import torch
        cuda = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
        f.write(f"# python {platform.python_version()}, torch {torch.__version__}, cuda device: {cuda}\n")
    except Exception:
        pass
    try:
        import warp
        f.write(f"# warp {warp.config.version}\n")
    except Exception:
        f.write("# warp not installed\n")
    f.write("\n")
    f.flush()
    sys.stdout = _Tee(sys.stdout, f)
    sys.stderr = _Tee(sys.stderr, f)
    state = {"error": None}
    prev_hook = sys.excepthook

    def hook(etype, value, tb):
        state["error"] = "".join(traceback.format_exception(etype, value, tb))
        prev_hook(etype, value, tb)
    sys.excepthook = hook

    def finish():
        el = time.time() - t0
        try:
            f.write(f"\n# finished {time.strftime('%Y-%m-%d %H:%M:%S')} after {el:.0f}s: "
                    f"{'FAILED (traceback above)' if state['error'] else 'ok'}\n")
            f.close()
        except ValueError:
            pass
    atexit.register(finish)


def _quote(s):
    return f'"{s}"' if (" " in s or not s) else s
