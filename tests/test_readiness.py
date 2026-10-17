import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import ROOT, Settings
from app.main import create_app


def reference_weights(path, **overrides):
    values = dict(mean=np.zeros(4608), scale=np.ones(4608), coef=np.zeros((1, 4608)),
                  intercept=np.zeros(1), labels=np.array(['normal', 'cashier_leaves']),
                  threshold=.8, demo_only=False)
    values.update(overrides)
    np.savez(path, **values)


@pytest.fixture
def settings(tmp_path):
    model = tmp_path / 'reference.npz'
    reference_weights(model)
    return Settings('readiness-test-secret', 'reference', model, tmp_path / 'runtime', ROOT / 'rules/catalog.json')


def test_ready_requires_successful_model_load_and_synthetic_inference(settings):
    with TestClient(create_app(settings)) as client:
        assert client.get('/health').json()['status'] == 'ok'
        response = client.get('/ready')
        assert response.status_code == 200
        assert response.json()['ready'] is True
        assert response.json()['readiness']['check_sec'] > 0
        assert not list(settings.runtime.glob('.readiness-*.json'))


@pytest.mark.parametrize('problem', ['missing', 'corrupt', 'demo', 'weak', 'unknown_rule', 'nonfinite'])
def test_unusable_model_is_rejected_before_job_acceptance(settings, problem):
    if problem == 'missing':
        settings.model.unlink()
    elif problem == 'corrupt':
        settings.model.write_bytes(b'not model weights')
    elif problem == 'demo':
        reference_weights(settings.model, demo_only=True)
    elif problem == 'weak':
        reference_weights(settings.model, weak_labels=True)
    elif problem == 'unknown_rule':
        reference_weights(settings.model, labels=np.array(['normal', 'invented']))
    elif problem == 'nonfinite':
        reference_weights(settings.model, coef=np.full((1, 4608), np.nan))
    with TestClient(create_app(settings)) as client:
        assert client.get('/health').status_code == 200
        assert client.get('/ready').status_code == 503
        response = client.post('/api/v1/analyze', headers={'X-API-Key': settings.api_key},
                               files={'file': ('test.mov', b'synthetic placeholder')})
        assert response.status_code == 503
        assert 'verdict' not in response.json()
        assert str(settings.model) not in response.text
        with client.app.state.jobs.db() as db:
            assert db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0] == 0


def test_model_replacement_invalidates_readiness_until_restart(settings):
    with TestClient(create_app(settings)) as client:
        assert client.get('/ready').status_code == 200
        settings.model.write_bytes(b'replaced after startup')
        response = client.get('/ready')
        assert response.status_code == 503
        assert response.json()['model_present'] is True
        assert response.json()['readiness']['code'] == 'restart_required'


def test_probe_timeout_does_not_prevent_server_liveness(settings):
    settings.readiness_timeout = .001
    with TestClient(create_app(settings)) as client:
        assert client.get('/health').status_code == 200
        assert client.get('/ready').json()['readiness']['code'] == 'probe_timeout'
