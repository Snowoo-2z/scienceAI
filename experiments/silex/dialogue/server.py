"""Single-process local demonstration. No external model; no prompt persistence."""
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import time
from model import load, generate

ROOT=Path(__file__).resolve().parent
PARAMS,META=load()


def current_rss():
    try:
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                return int(line.split()[1])*1024
    except OSError:
        return None


class Handler(BaseHTTPRequestHandler):
    def reply(self, code, body, kind='application/json; charset=utf-8'):
        data=body if isinstance(body,bytes) else json.dumps(body,ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header('Content-Type',kind)
        self.send_header('Content-Length',str(len(data)))
        self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path=='/':
            self.reply(200,(ROOT/'index.html').read_bytes(),'text/html; charset=utf-8')
        elif self.path=='/api/status':
            self.reply(200,dict(name=META['model'],parameters=META['parameter_count'],
                               weight_bytes=META['parameter_bytes'],rss_bytes=current_rss(),
                               training_pairs=META['training_pairs'],training_seconds=META['training_seconds']))
        elif self.path=='/api/results':
            self.reply(200,(ROOT/'responses.json').read_bytes())
        else:
            self.reply(404,dict(error='Ressource inconnue.'))

    def do_POST(self):
        if self.path!='/api/chat':
            self.reply(404,dict(error='Ressource inconnue.')); return
        try:
            size=int(self.headers.get('Content-Length','0'))
        except ValueError:
            self.reply(400,dict(error='Longueur invalide.')); return
        if not 0<size<=8192:
            self.reply(413,dict(error='Message trop long ou vide.')); return
        try:
            data=json.loads(self.rfile.read(size))
            q=data.get('question') if isinstance(data,dict) else None
            if not isinstance(q,str) or not q.strip() or len(q)>500:
                raise ValueError()
        except (ValueError,UnicodeDecodeError):
            self.reply(400,dict(error='Écris une question de 1 à 500 caractères.')); return
        start=time.perf_counter()
        result=generate(PARAMS,META['vocab'],q)
        result['elapsed_ms']=round((time.perf_counter()-start)*1000,2)
        result['rss_bytes']=current_rss()
        self.reply(200,result)

    def log_message(self,*args):
        # No question or response is logged or persisted.
        pass


if __name__=='__main__':
    print('Silex laboratoire : http://0.0.0.0:8080',flush=True)
    HTTPServer(('0.0.0.0',8080),Handler).serve_forever()
