"""Real multipart/API contract check. Writes aggregate diagnostics, never evidence or keys."""
import argparse
import base64
import json
import math
from pathlib import Path
import time
import httpx
import cv2
import numpy as np


def verify(args):
    key=Path(args.key_file).read_text(encoding='utf-8').strip()
    report={'url':args.url,'upload_bytes':Path(args.video).stat().st_size,'checks':{}}
    headers={'X-API-Key':key}
    cap=cv2.VideoCapture(str(args.video))
    try:
        fps=cap.get(cv2.CAP_PROP_FPS);count=cap.get(cv2.CAP_PROP_FRAME_COUNT)
        assert fps>0 and count>0,'Test input has no valid timing metadata'
        duration=count/fps
    finally:cap.release()
    print(json.dumps({'stage':'checking_authentication'}),flush=True)
    with httpx.Client(base_url=args.url,timeout=httpx.Timeout(30,connect=10),trust_env=False,follow_redirects=False) as client:
        for path in ['/api/v1/jobs/'+'a'*32,'/api/v1/rules']:
            for auth in ({},{'X-API-Key':'invalid-jury-key'}):
                response=client.get(path,headers=auth)
                assert response.status_code==401,(path,response.status_code)
        for auth in ({},{'X-API-Key':'invalid-jury-key'}):
            response=client.post('/api/v1/analyze',headers=auth,files={'file':('tiny.mov',b'invalid','video/quicktime')})
            assert response.status_code==401,response.status_code
        report['checks']['missing_and_wrong_keys_rejected']=True
        print(json.dumps({'stage':'checking_readiness'}),flush=True)
        response=client.get('/ready');response.raise_for_status()
        assert response.json()['ready'] is True
        report['checks']['ready']=True
        started=time.monotonic()
        print(json.dumps({'stage':'uploading','bytes':report['upload_bytes']}),flush=True)
        with open(args.video,'rb') as stream:
            response=client.post('/api/v1/analyze',headers=headers,
                files={'file':(Path(args.video).name,stream,'video/quicktime')},
                timeout=httpx.Timeout(240,connect=10,write=120,pool=10))
        assert response.status_code==202,response.status_code
        report['upload_accept_seconds']=round(time.monotonic()-started,3)
        accepted=response.json();assert accepted['status']=='processing'
        jid=accepted['job_id'];report['job_id']=jid
        print(json.dumps({'stage':'polling_job'}),flush=True)
        while time.monotonic()-started<300:
            response=client.get('/api/v1/jobs/'+jid,headers=headers);response.raise_for_status()
            result=response.json();assert result['status'] in ['processing','done','failed']
            assert result['job_id']==jid
            if result['status']=='failed':
                report.update(status='failed',error=result.get('error',''),seconds=round(time.monotonic()-started,3))
                break
            if result['status']=='done':
                assert result['verdict'] in ['violation','no_violation']
                assert (result['verdict']=='violation')==bool(result['violations'])
                frames=0
                for event in result['violations']:
                    assert isinstance(event['type'],str) and event['type']
                    assert isinstance(event['description'],str) and event['description']
                    assert not isinstance(event['timecode_sec'],bool) and isinstance(event['timecode_sec'],(int,float))
                    assert math.isfinite(event['timecode_sec']) and 0<=event['timecode_sec']<=duration+.55
                    if result.get('meta',{}).get('engine')=='qwen_rag':
                        rule=event['rule']
                        assert rule['url'].startswith('https://adilet.zan.kz/') and rule['document'] and rule['section']
                    assert 1<=len(event['evidence_frames'])<=3
                    for frame in event['evidence_frames']:
                        decoded=base64.b64decode(frame['image_base64'],validate=True)
                        assert decoded.startswith(b'\xff\xd8') and decoded.endswith(b'\xff\xd9')
                        pixels=cv2.imdecode(np.frombuffer(decoded,np.uint8),cv2.IMREAD_COLOR)
                        assert pixels is not None and pixels.size>0
                        assert not isinstance(frame['timestamp_sec'],bool) and isinstance(frame['timestamp_sec'],(int,float))
                        assert math.isfinite(frame['timestamp_sec']) and 0<=frame['timestamp_sec']<=duration+.55
                        frames+=1
                report.update(status='done',verdict=result['verdict'],events=len(result['violations']),
                    evidence_frames=frames,seconds=round(time.monotonic()-started,3),
                    engine=result.get('meta',{}).get('engine'),response_schema_valid=True)
                report['jpeg_pixels_decoded']=True
                report['within_five_minutes']=report['seconds']<=300
                if not report['within_five_minutes']:report['status']='deadline_exceeded'
                break
            time.sleep(3)
        else:report.update(status='timeout',seconds=round(time.monotonic()-started,3))
    Path(args.output).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))
    if report['status']!='done':raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('video');p.add_argument('--url',required=True)
    p.add_argument('--key-file',required=True);p.add_argument('--output',required=True)
    verify(p.parse_args())
