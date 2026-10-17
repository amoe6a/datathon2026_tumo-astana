import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from app.config import ROOT, Settings
from app.lifecycle import RuntimeLease
from app.main import Jobs
from app.processes import start_child, stop_child


@pytest.fixture
def settings(tmp_path):
    return Settings('lifecycle-test-secret', 'demo', tmp_path/'model.npz', tmp_path/'runtime', ROOT/'rules/catalog.json')


def job_files(settings, jid):
    folder = settings.runtime / jid
    folder.mkdir(parents=True)
    for name in ('input.mov', 'progress.json', 'progress.tmp', 'result.tmp'):
        (folder/name).write_bytes(b'synthetic test data')
    result = {'ok': True, 'result': {'verdict': 'no_violation', 'violations': [], 'meta': {}}}
    (folder/'result.json').write_text(json.dumps(result), encoding='utf-8')
    return folder/'input.mov'


def test_recovery_removes_interrupted_and_orphan_uploads_only(settings):
    jobs = Jobs(settings)
    interrupted, orphan = 'a'*16, 'b'*16
    job_files(settings, interrupted)
    job_files(settings, orphan)
    keep = settings.runtime/'manual-notes'
    keep.mkdir()
    (keep/'input.mov').write_bytes(b'must remain')
    jobs.save(interrupted, {'job_id': interrupted, 'status': 'processing'})
    jobs.save('c'*16, {'job_id': 'c'*16, 'status': 'done', 'verdict': 'no_violation'})
    jobs.recover()
    assert jobs.get(interrupted)['status'] == 'failed'
    assert jobs.get('c'*16)['status'] == 'done'
    assert not list((settings.runtime/interrupted).iterdir())
    assert not list((settings.runtime/orphan).iterdir())
    assert (keep/'input.mov').read_bytes() == b'must remain'


def test_locked_cleanup_releases_slot_and_is_retried(settings, monkeypatch):
    jobs = Jobs(settings)
    jid = 'a'*16
    video = job_files(settings, jid)
    class Finished:
        def wait(self, timeout):
            return 0
    monkeypatch.setattr('app.main.start_child', lambda *args: Finished())
    monkeypatch.setattr('app.main.stop_child', lambda process: None)
    monkeypatch.setattr('app.lifecycle.time.sleep', lambda seconds: None)
    unlink = Path.unlink
    def locked(path, *args, **kwargs):
        if path == video:
            raise PermissionError('simulated Windows lock')
        return unlink(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'unlink', locked)
    assert jobs.slot.acquire(blocking=False)
    jobs.run(jid, video)
    assert jobs.get(jid)['status'] == 'done'
    assert jobs.maintenance_info()['cleanup_pending_files'] == 1
    assert jobs.slot.acquire(blocking=False)
    jobs.slot.release()
    monkeypatch.setattr(Path, 'unlink', unlink)
    jobs.retry_cleanup()
    assert not video.exists()
    assert jobs.maintenance_info()['cleanup_pending_files'] == 0


def test_process_start_failure_is_failed_and_releases_capacity(settings, monkeypatch):
    jobs = Jobs(settings)
    jid = 'a'*16
    video = job_files(settings, jid)
    def cannot_start(*args):
        raise OSError('private environment information')
    monkeypatch.setattr('app.main.start_child', cannot_start)
    assert jobs.slot.acquire(blocking=False)
    jobs.run(jid, video)
    assert jobs.get(jid)['status'] == 'failed'
    assert 'private' not in jobs.get(jid)['error']
    assert not video.exists()
    assert jobs.slot.acquire(blocking=False)
    jobs.slot.release()


def test_shutdown_stops_active_child_and_marks_job_failed(settings, monkeypatch):
    jobs = Jobs(settings)
    jid = 'a'*16
    video = job_files(settings, jid)
    owned = []
    def sleeping_child(*args):
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], stdin=subprocess.PIPE)
        owned.append(child)
        return child
    monkeypatch.setattr('app.main.start_child', sleeping_child)
    assert jobs.slot.acquire(blocking=False)
    jobs.launch(jid, video)
    try:
        deadline = time.monotonic() + 5
        while jobs._active is None and time.monotonic() < deadline:
            time.sleep(.01)
        assert jobs._active is not None
        jobs.close()
        assert all(p.poll() is not None for p in owned)
        assert jobs.get(jid)['status'] == 'failed'
        assert not video.exists()
        assert jobs.slot.acquire(blocking=False)
        jobs.slot.release()
    finally:
        for child in owned:
            stop_child(child)


def test_runtime_lease_prevents_second_server_and_releases_on_close(settings):
    first, second = RuntimeLease(settings.runtime), RuntimeLease(settings.runtime)
    first.acquire()
    try:
        with pytest.raises(RuntimeError, match='already in use'):
            second.acquire()
    finally:
        first.release()
    second.acquire()
    second.release()


def test_model_worker_exits_when_parent_pipe_closes(tmp_path):
    entered, release = threading.Event(), threading.Event()
    class StalledModel(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            entered.set()
            release.wait(15)
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), StalledModel)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    config = tmp_path/'bonsai.json'
    config.write_text(json.dumps({'endpoint': f'http://127.0.0.1:{server.server_port}'}))
    child = start_child('probe', ['--engine', 'bonsai', '--model', str(config),
                                 '--rules', str(ROOT/'rules/catalog.json'), '--output', str(tmp_path/'result.json')])
    try:
        assert entered.wait(10), 'Synthetic model probe never reached the stub'
        child.stdin.close()  # The same EOF is delivered when the API process dies.
        assert child.wait(timeout=5) == 72
    finally:
        stop_child(child)
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
