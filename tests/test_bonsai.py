import json
import numpy as np
import pytest
import httpx
from ml.bonsai import BonsaiSession,checked_endpoint
from ml.video import Frame


@pytest.mark.parametrize('url',['https://127.0.0.1:8080','http://localhost:8080','http://example.com',
    'http://127.0.0.1.evil.test:8080','http://user:secret@127.0.0.1:8080','http://127.0.0.1:8080?forward=remote'])
def test_bonsai_rejects_nonloopback_and_ambiguous_urls(url):
    with pytest.raises(ValueError,match='loopback'):
        checked_endpoint(url)


def test_bonsai_local_image_and_truncation(tmp_path):
    config=tmp_path/'config.json';config.write_text(json.dumps({'endpoint':'http://127.0.0.1:8080'}))
    responses=[{'finish_reason':'stop','message':{'content':'{"events":[],"uncertain":false}'}},
               {'finish_reason':'length','message':{'content':'{"events":[],"uncertain":false}'}}]
    def respond(request):
        assert str(request.url)=='http://127.0.0.1:8080/v1/chat/completions'
        body=json.loads(request.content)
        assert body['chat_template_kwargs']['enable_thinking'] is False
        content=body['messages'][0]['content']
        assert content[-1]['image_url']['url'].startswith('data:image/jpeg;base64,')
        return httpx.Response(200,json={'choices':[responses.pop(0)]})
    session=BonsaiSession(config,transport=httpx.MockTransport(respond))
    window=[Frame(0,np.zeros((16,16,3),np.uint8))]
    try:
        assert session.infer(window,'prompt')=='{"events":[],"uncertain":false}'
        with pytest.raises(ValueError,match='Truncated'):
            session.infer(window,'prompt')
    finally:session.close()


def test_bonsai_does_not_follow_redirects(tmp_path):
    config=tmp_path/'config.json';config.write_text(json.dumps({'endpoint':'http://127.0.0.1:8080'}))
    session=BonsaiSession(config,transport=httpx.MockTransport(lambda request:
        httpx.Response(307,headers={'Location':'https://external.invalid/inference'})))
    try:
        with pytest.raises(httpx.HTTPStatusError):
            session.infer([],'prompt')
    finally:session.close()


@pytest.mark.parametrize('limit,expected_height', [(448, 448), (896, 848)])
def test_bonsai_resolution_reaches_local_model(tmp_path, limit, expected_height):
    import base64
    import cv2
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'endpoint': 'http://127.0.0.1:8080', 'image_max_side': limit}))
    def respond(request):
        content = json.loads(request.content)['messages'][0]['content']
        encoded = content[-1]['image_url']['url'].split(',', 1)[1]
        image = cv2.imdecode(np.frombuffer(base64.b64decode(encoded), np.uint8), cv2.IMREAD_COLOR)
        assert image.shape[0] == expected_height
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {'content': '{}'}}]})
    session = BonsaiSession(config, transport=httpx.MockTransport(respond))
    try: session.infer([Frame(0, np.zeros((848, 464, 3), np.uint8))], 'synthetic')
    finally: session.close()
