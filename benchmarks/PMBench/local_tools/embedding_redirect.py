"""Record the configured embedding route without redirects or provider fallback.

The class name is retained for existing transport imports. Fresh release runs
must configure the active endpoint in a-mem/.env; historical migrations are not
silently applied to new requests.
"""
import threading
import httpx

class EmbeddingRedirectTransport(httpx.BaseTransport):
    def __init__(self,inner=None):
        self.inner=inner or httpx.HTTPTransport()
        self.local=threading.local()

    @property
    def last_route(self):
        return getattr(self.local,'route',None)

    def handle_request(self,request):
        request.read()
        self.local.route=dict(configured_url=str(request.url),actual_url=str(request.url),
                              method=request.method,redirects=[])
        return self.inner.handle_request(request)

    def close(self):self.inner.close()
