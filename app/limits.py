"""Reject unauthenticated or oversized API bodies before multipart parsing."""
import hmac
from starlette.responses import JSONResponse

class BodyTooLarge(Exception):
    pass

class ApiLimits:
    def __init__(self, app, key, max_bytes):
        self.app,self.key,self.max_bytes=app,key,max_bytes

    async def __call__(self, scope, receive, send):
        if scope['type']!='http' or not scope.get('path','').startswith('/api/'):
            return await self.app(scope,receive,send)
        headers=dict(scope.get('headers',[]))
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
