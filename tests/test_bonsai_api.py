"""Synthetic HTTP stub proves adapter/API wiring, never banking model accuracy."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from app.config import Settings,ROOT
from app.main import create_app
from scripts.make_demo import clip


@pytest.mark.parametrize('rule_id,status',[('cashier_leaves','done'),('invented','failed')])
def test_bonsai_worker_http_and_evidence(tmp_path,rule_id,status):
    calls = 0
    healthy = True
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200 if healthy else 503)
            self.send_header('Content-Length','0');self.end_headers()
        def do_POST(self):
            nonlocal calls
            calls += 1
            body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            assert body['messages'][0]['content'][-1]['image_url']['url'].startswith('data:image/jpeg;base64,')
            content=json.dumps({'events':[{'rule_id':'cashier_leaves' if calls == 1 else rule_id,'timecode_sec':2.,
                'observation':'Synthetic stub response, not a banking accuracy result.'}],'uncertain':False})
            result=json.dumps({'choices':[{'finish_reason':'stop','message':{'content':content}}]}).encode()
            self.send_response(200);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(result)));self.end_headers();self.wfile.write(result)
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    config=tmp_path/'bonsai.json';config.write_text(json.dumps({'endpoint':f'http://127.0.0.1:{server.server_port}'}))
    key='synthetic-bonsai-api-test'
    settings=Settings(key,'bonsai',config,tmp_path/'runtime',ROOT/'rules/catalog.json',30,64)
    video=tmp_path/'synthetic.mov';clip(video,[(1,2.5)],duration=3)
    try:
        with TestClient(create_app(settings)) as client:
            headers={'X-API-Key':key}
            assert client.get('/ready').status_code == 200
            healthy = False
            assert client.get('/ready').status_code == 503
            blocked = client.post('/api/v1/analyze', headers=headers,
                                  files={'file':('synthetic.mov',b'synthetic placeholder')})
            assert blocked.status_code == 503
            assert client.get('/health').json()['readiness']['code'] == 'model_unavailable'
            healthy = True
            assert client.get('/ready').status_code == 200
            with video.open('rb') as stream:
                response=client.post('/api/v1/analyze',headers=headers,files={'file':('synthetic.mov',stream)})
            assert response.status_code==202
            job=response.json()['job_id']
            for _ in range(100):
                result=client.get('/api/v1/jobs/'+job,headers=headers).json()
                if result['status']!='processing':break
                time.sleep(.1)
            assert result['status']==status,result
            if status=='done':
                assert result['verdict']=='violation'
                assert result['violations'][0]['rule']['id']=='cashier_leaves'
                assert 1<=len(result['violations'][0]['evidence_frames'])<=3
            else:assert 'verdict' not in result
    finally:server.shutdown();server.server_close();thread.join(timeout=2)
