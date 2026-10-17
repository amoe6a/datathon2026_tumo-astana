"""Reject unauthenticated or oversized API bodies before multipart parsing."""
import hmac
from urllib.parse import urlsplit
from starlette.responses import JSONResponse

def local_test_request(scope):
    headers = dict(scope.get('headers', []))
    host = headers.get(b'host', b'').decode('latin-1')
    client = scope.get('client') or ('', 0)
    if client[0] not in {'127.0.0.1', '::1'}:
        return False
    if urlsplit('http://' + host).hostname not in {'localhost', '127.0.0.1', '::1'}:
        return False
    if any(name in headers for name in (b'forwarded', b'x-forwarded-for', b'x-forwarded-host')):
        return False
    origin = headers.get(b'origin')
    return not origin or origin.decode('latin-1') == scope.get('scheme', 'http') + '://' + host

class BodyTooLarge(Exception):
    pass

class ApiLimits:
    def __init__(self, app, key, max_bytes, local_test=False):
        self.app,self.key,self.max_bytes=app,key,max_bytes
        self.local_test = local_test

    async def __call__(self, scope, receive, send):
        if scope['type']!='http' or not scope.get('path','').startswith('/api/'):
            return await self.app(scope,receive,send)
        headers=dict(scope.get('headers',[]))
        if self.local_test and local_test_request(scope):
            scope = dict(scope)
            scope['headers'] = [(k, v) for k, v in scope['headers'] if k != b'x-api-key'] + [(b'x-api-key', self.key.encode())]
            headers = dict(scope['headers'])
        if not hmac.compare_digest(headers.get(b'x-api-key',b''),self.key.encode()):
            return await JSONResponse({'detail':'Invalid or missing X-API-Key'},401)(scope,receive,send)
        try:
            if int(headers.get(b'content-length',b'0'))>self.max_bytes:
                return await JSONResponse({'detail':'Upload exceeds request limit'},413)(scope,receive,send)
        except ValueError:
            return await JSONResponse({'detail':'Invalid Content-Length'},400)(scope,receive,send)
        total=0
        async def limited_receive():
            nonlocal total
            message=await receive()
            if message['type']=='http.request':
                total+=len(message.get('body',b''))
                if total>self.max_bytes:raise BodyTooLarge()
            return message
        try:
            await self.app(scope,limited_receive,send)
        except BodyTooLarge:
            await JSONResponse({'detail':'Upload exceeds request limit'},413)(scope,receive,send)
