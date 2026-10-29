from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
import json
from urllib.parse import urlparse
from pathlib import Path
ROOT=Path(__file__).parent
class App(SimpleHTTPRequestHandler):
    def __init__(self,*a,**k):super().__init__(*a,directory=str(ROOT),**k)
    def log_message(self,*a):pass
    def do_GET(self):
        if self.path == '/api/status':
            from lumina_general import runtime_status
            payload=json.dumps({'provider':runtime_status(),'local':True},ensure_ascii=False).encode('utf-8')
            self.send_response(200); self.send_header('Content-Type','application/json; charset=utf-8'); self.send_header('Content-Length',str(len(payload))); self.end_headers(); self.wfile.write(payload); return
        return super().do_GET()
    def do_POST(self):
        if self.path == '/api/chat':
            try:
                size=int(self.headers.get('Content-Length','0')); data=json.loads(self.rfile.read(size) or b'{}')
                from lumina_general import generate
                answer,meta=generate(data.get('messages',[]),max_tokens=int(data.get('max_tokens',200)))
                payload=json.dumps({'reply':answer,'meta':meta,'provider':meta.get('provider',meta.get('engine'))},ensure_ascii=False).encode('utf-8')
                self.send_response(200); self.send_header('Content-Type','application/json; charset=utf-8'); self.send_header('Content-Length',str(len(payload))); self.end_headers(); self.wfile.write(payload)
            except Exception as exc:
                payload=json.dumps({'error':str(exc)},ensure_ascii=False).encode('utf-8'); self.send_response(500); self.send_header('Content-Type','application/json; charset=utf-8'); self.send_header('Content-Length',str(len(payload))); self.end_headers(); self.wfile.write(payload)
            return
        if self.path != '/api/voice': self.send_error(404); return
        try:
            size=int(self.headers.get('Content-Length','0')); data=json.loads(self.rfile.read(size) or b'{}')
            from lumina_voice_neural import synthesize_bytes
            audio=synthesize_bytes(str(data.get('text','')))
            self.send_response(200); self.send_header('Content-Type','audio/wav'); self.send_header('Content-Length',str(len(audio))); self.end_headers(); self.wfile.write(audio)
        except Exception as exc:
            payload=json.dumps({'error':str(exc)}).encode(); self.send_response(500); self.send_header('Content-Type','application/json'); self.send_header('Content-Length',str(len(payload))); self.end_headers(); self.wfile.write(payload)
if __name__=='__main__':ThreadingHTTPServer(('127.0.0.1',7020),App).serve_forever()
