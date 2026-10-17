from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.limits import ApiLimits


def client(enabled, address='127.0.0.1'):
    app = FastAPI()
    app.add_middleware(ApiLimits, key='synthetic-test-key', max_bytes=100, local_test=enabled)

    @app.get('/api/check')
    def check():
        return {'ok': True}

    return TestClient(app, base_url='http://127.0.0.1', client=(address, 50000))


def test_local_access_is_opt_in_and_local_only():
    assert client(False).get('/api/check').status_code == 401
    assert client(True).get('/api/check').status_code == 200
    assert client(True, '192.168.1.2').get('/api/check').status_code == 401
    for headers in ({'Host': 'public.example'}, {'Origin': 'http://elsewhere.example'},
                    {'X-Forwarded-For': '192.168.1.2'}):
        assert client(True).get('/api/check', headers=headers).status_code == 401
    assert client(False).get('/api/check', headers={'X-API-Key': 'synthetic-test-key'}).status_code == 200
