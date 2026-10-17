import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import ROOT, Settings
from app.main import create_app
from ml.worker import atomic_json


def test_status_survives_locked_progress_and_recovers(tmp_path, monkeypatch):
    key = 'synthetic-progress-test-key'
    settings = Settings(key, 'demo', tmp_path/'model.npz', tmp_path/'runtime', ROOT/'rules/catalog.json')
    app = create_app(settings)
    job_id = 'a' * 16
    payload = {'job_id': job_id, 'status': 'processing', 'progress': 1}
    app.state.jobs.save(job_id, payload)
    progress = settings.runtime/job_id/'progress.json'
    progress.parent.mkdir()
    progress.write_text(json.dumps({'progress': 70, 'stage': 'Analyzing visible actions'}))
    read_text = Path.read_text

    def locked_read(path, *args, **kwargs):
        if path == progress:
            raise PermissionError('simulated Windows sharing violation')
        return read_text(path, *args, **kwargs)

    with TestClient(app) as client:
        app.state.jobs.save(job_id, payload)
        progress.write_text(json.dumps({'progress': 70, 'stage': 'Analyzing visible actions'}))
        monkeypatch.setattr(Path, 'read_text', locked_read)
        response = client.get(f'/api/v1/jobs/{job_id}', headers={'X-API-Key': key})
        assert response.status_code == 200
        assert response.json() == payload
        monkeypatch.setattr(Path, 'read_text', read_text)
        recovered = client.get(f'/api/v1/jobs/{job_id}', headers={'X-API-Key': key})
        assert recovered.status_code == 200
        assert recovered.json()['progress'] == 70
        assert recovered.json()['status'] == 'processing'


def test_atomic_json_recovers_from_transient_file_lock(tmp_path, monkeypatch):
    target = tmp_path/'result.json'
    replace = Path.replace
    locked = True

    def sharing_violation(path, destination):
        nonlocal locked
        if locked:
            locked = False
            raise PermissionError('simulated Windows sharing violation')
        return replace(path, destination)

    monkeypatch.setattr(Path, 'replace', sharing_violation)
    monkeypatch.setattr('ml.worker.time.sleep', lambda _: None)
    result = {'ok': True, 'result': {'verdict': 'no_violation'}}
    atomic_json(target, result)
    assert json.loads(target.read_text()) == result
    assert not target.with_suffix('.tmp').exists()


def test_atomic_json_reports_persistent_lock_without_replacing_old_data(tmp_path, monkeypatch):
    target = tmp_path/'result.json'
    target.write_text('{"previous": true}')

    def locked_replace(path, destination):
        raise PermissionError('persistent lock')

    monkeypatch.setattr(Path, 'replace', locked_replace)
    monkeypatch.setattr('ml.worker.time.sleep', lambda _: None)
    with pytest.raises(PermissionError, match='persistent lock'):
        atomic_json(target, {'ok': True})
    assert json.loads(target.read_text()) == {'previous': True}
