import base64
import json
import time
from pathlib import Path
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from app.config import Settings,ROOT
from app.main import create_app,Jobs
from ml.engine import analyze
from ml.qwen import parse_events
from ml.train import train
from scripts.make_demo import clip

KEY='test-key-not-for-public-use'

@pytest.fixture(scope='session')
def model(tmp_path_factory):
    folder=tmp_path_factory.mktemp('model');rows=[]
    for i,absent in enumerate([False,True,False,True]):
        path=folder/f'train_{i}.mov';clip(path,[(3,6)] if absent else [],i)
        segments=([{'start':0,'end':3,'label':'normal'},{'start':3,'end':6,'label':'demo_absent'},{'start':6,'end':10,'label':'normal'}] if absent else [{'start':0,'end':10,'label':'normal'}])
        rows.append({'path':path.name,'group':str(i),'split':'train','segments':segments})
    manifest=folder/'manifest.json';manifest.write_text(json.dumps({'videos':rows}))
    target=folder/'model.npz';train(manifest,target,demo=True)
    return target

@pytest.fixture
def settings(tmp_path,model):
    return Settings(KEY,'demo',model,tmp_path/'runtime',ROOT/'rules/catalog.json',30,64)

def wait(client,jid):
    for _ in range(150):
        r=client.get('/api/v1/jobs/'+jid,headers={'X-API-Key':KEY});assert r.status_code==200
        value=r.json()
        if value['status']!='processing':return value
        time.sleep(.1)
    pytest.fail('job did not finish')

@pytest.mark.parametrize('events,count',[([],0),([(3,6)],1),([(2,4),(6,8)],2)])
def test_actual_video_api(settings,tmp_path,events,count):
    video=tmp_path/'new.mov';clip(video,events,seed=99)
    with TestClient(create_app(settings)) as c:
        with video.open('rb') as f:r=c.post('/api/v1/analyze',headers={'X-API-Key':KEY},files={'file':('new.mov',f)})
        assert r.status_code==202
        result=wait(c,r.json()['job_id']);assert result['status']=='done',result
        assert len(result['violations'])==count
        assert result['verdict']==('violation' if count else 'no_violation')
        assert result['meta']['demo_only'] is True
        for v in result['violations']:
            assert any(s<=v['timecode_sec']<=e for s,e in events)
            assert len(v['evidence_frames'])==3
            assert v['timecode_sec'] in [f['timestamp_sec'] for f in v['evidence_frames']]
            for f in v['evidence_frames']:
                img=cv2.imdecode(np.frombuffer(base64.b64decode(f['image_base64']),dtype=np.uint8),cv2.IMREAD_COLOR)
                assert img is not None and img.shape[0]>0
        assert not list(settings.runtime.glob('*/input.mov'))

def test_auth_and_bad_uploads(settings):
    with TestClient(create_app(settings)) as c:
        assert c.get('/api/v1/jobs/'+'a'*16).status_code==401
        assert c.post('/api/v1/analyze',files={'file':('x.mov',b'x')}).status_code==401
        assert c.post('/api/v1/analyze',headers={'X-API-Key':KEY},files={'file':('x.exe',b'x')}).status_code==415
        assert c.post('/api/v1/analyze',headers={'X-API-Key':KEY},files={'file':('x.mov',b'')}).status_code==400
        r=c.post('/api/v1/analyze',headers={'X-API-Key':KEY},files={'file':('x.mov',b'not a video')})
        result=wait(c,r.json()['job_id']);assert result['status']=='failed';assert 'verdict' not in result

def test_missing_model_is_not_clean_verdict(settings):
    settings.model=Path('missing-reference.npz')
    with TestClient(create_app(settings)) as c:
        assert c.post('/api/v1/analyze',headers={'X-API-Key':KEY},files={'file':('x.mov',b'x')}).status_code==503

def test_timeout_releases_slot(settings):
    settings.timeout=.001
    with TestClient(create_app(settings)) as c:
        r=c.post('/api/v1/analyze',headers={'X-API-Key':KEY},files={'file':('x.mov',b'x')})
        value=wait(c,r.json()['job_id']);assert value['status']=='failed'
        assert c.app.state.jobs.slot.acquire(blocking=False)
        c.app.state.jobs.slot.release()

def test_restart_marks_interrupted_jobs(settings):
    jobs=Jobs(settings);jobs.save('a'*16,{'job_id':'a'*16,'status':'processing'})
    fresh=Jobs(settings);assert fresh.get('a'*16)['status']=='failed'

def test_no_demo_model_in_competition(settings,tmp_path):
    video=tmp_path/'x.mov';clip(video,[])
    with pytest.raises(ValueError,match='Synthetic'):
        analyze(video,settings.model,'reference',settings.rules)

@pytest.mark.parametrize('raw',['not json','{"events":[],"uncertain":"false"}','{"events":[{"rule_id":"invented","timecode_sec":1,"observation":"An invented event"}],"uncertain":false}','{"events":[{"rule_id":"known","timecode_sec":99,"observation":"A visible action"}],"uncertain":false}'])
def test_model_output_validation(raw):
    with pytest.raises(ValueError):parse_events(raw,{'known':{}},0,5)

def test_model_valid_empty_and_positive():
    assert parse_events('{"events":[],"uncertain":false}',{},0,5)==([],False)
    events,u=parse_events('{"events":[{"rule_id":"known","timecode_sec":2,"observation":"Cashier walks away"}],"uncertain":true}',{'known':{}},0,5)
    assert len(events)==1 and u
