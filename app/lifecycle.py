"""Exclusive runtime ownership and narrowly scoped temporary-file cleanup."""
import os
from pathlib import Path
import time


def valid_job_id(value):
    return len(value) == 16 and all(c in '0123456789abcdef' for c in value)


class RuntimeLease:
    def __init__(self, runtime):
        self.path = Path(runtime) / '.server.lock'
        self.stream = None

    def acquire(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stream = self.path.open('a+b')
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            stream.close()
            raise RuntimeError('Runtime already in use; run one API process per runtime directory') from error
        self.stream = stream

    def release(self):
        if self.stream is not None:
            try:
                if os.name == 'nt':
                    import msvcrt
                    self.stream.seek(0)
                    msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
            finally:
                self.stream.close()
                self.stream = None


def remove_temporary(path):
    for attempt in range(5):
        try:
            path.unlink(missing_ok=True)
            return True
        except OSError:
            if attempt < 4:
                time.sleep(.02 * 2 ** attempt)
    return False


def cleanup_job(runtime, job_id):
    if not valid_job_id(job_id):
        return 0
    root = Path(runtime).resolve()
    folder = root / job_id
    if folder.is_symlink() or folder.resolve().parent != root:
        return 0
    pending = 0
    for name in ('input.mov', 'input.mp4', 'result.json', 'result.tmp', 'progress.json', 'progress.tmp'):
        path = folder / name
        if not remove_temporary(path):
            pending += 1
    return pending
