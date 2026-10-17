import json
import pytest
from fastapi.testclient import TestClient
from app.main import create_app
from app.config import Settings, ROOT
from app.limits import ApiLimits
from ml.qwen import parse_events

def test_admission_and_request_limits(tmp_path):
    model=tmp_path/'present.npz';model.write_bytes(b'placeholder')
    settings=Settings('edge-test-secret-1234','reference',model,tmp_path/'runtime',ROOT/'rules/catalog.json',10,1)
    with TestClient(create_app(settings)) as c:
        h={'X-API-Key':settings.api_key}
        assert c.get('/api/v1/jobs/'+'0'*16,headers={'X-API-Key':'incorrect-key-value'}).status_code==401
        assert c.get('/api/v1/jobs/'+'0'*16,headers=h).status_code==404
        assert c.get('/api/v1/demo/normal',headers=h).status_code==404
        assert c.post('/api/v1/analyze',content=b'x'*3_000_000,headers=h).status_code==413
        assert c.post('/api/v1/analyze',files={'file':('x.mov',b'x'*1_100_000)},headers=h).status_code==413
        assert c.app.state.jobs.slot.acquire(blocking=False)
        try:
            r=c.post('/api/v1/analyze',headers=h,files={'file':('x.mov',b'x')})
            assert r.status_code==429 and r.headers['retry-after']=='5'
        finally:c.app.state.jobs.slot.release()

def test_catalog_has_official_sources():
    catalog=json.loads((ROOT/'rules/catalog.json').read_text(encoding='utf-8'))
    for rule in catalog['rules']:
        if rule['enabled'] and not rule['demo_only']:
            assert rule['url'].startswith('https://adilet.zan.kz/')
            assert rule['section'] and rule['visual_test'] and rule['caution']

@pytest.mark.parametrize('t',['NaN','Infinity','true'])
def test_nonfinite_model_timestamps_rejected(t):
    raw='{"events":[{"rule_id":"x","timecode_sec":'+t+',"observation":"Visible departure"}],"uncertain":false}'
    with pytest.raises(ValueError):parse_events(raw,{'x':{}},0,10)
