"""Exit if the API process disappears, even during a stuck model call."""
import os
import sys
import threading
import time


def watch_parent():
    # Only the parent holds the pipe's write end. EOF also detects forced death.
    try:
        if os.name == 'nt':
            # A blocking stdin read interferes with NumPy's Windows initialization.
            # The parent never writes: poll for a broken pipe without consuming it.
            import ctypes
            from ctypes import wintypes
            import msvcrt
            peek = ctypes.WinDLL('kernel32', use_last_error=True).PeekNamedPipe
            peek.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
                             ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            peek.restype = wintypes.BOOL
            handle = msvcrt.get_osfhandle(sys.stdin.fileno())
            while peek(handle, None, 0, None, None, None):
                time.sleep(.2)
        else:
            while os.read(sys.stdin.fileno(), 1):
                pass
    finally:
        os._exit(72)


def main():
    mode = sys.argv.pop(1)
    threading.Thread(target=watch_parent, daemon=True).start()
    if mode == 'job':
        from ml.worker import main as worker
    elif mode == 'probe':
        from app.readiness import main as worker
    else:
        raise ValueError('Unknown worker mode')
    worker()


if __name__ == '__main__':
    main()
