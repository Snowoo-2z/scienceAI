"""Local demonstration lab, not a chatbot. No request logging or external API.
Examples/hypotheses live in a temporary directory until deletion or shutdown.
"""
import argparse
import json
from http.server import BaseHTTPRequestHandler,HTTPServer
from pathlib import Path
import resource
import tempfile
import time
import uuid
from core import Space,bounded
from probes import generate

ROOT=Path(__file__).parent


def state(s):
    return dict(hypotheses=s.count(),search=s.get('search'),examples=s.get('examples'),
                sample_programs=[p for _,p in list_head(s.programs(),5)])


def list_head(iterator,n):
    import itertools
    return list(itertools.islice(iterator,n))


def serve(port):
    with tempfile.TemporaryDirectory(prefix='silex-induction-') as tmp:
        sessions={}
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def send(self,status,payload,kind='application/json; charset=utf-8'):
                body=payload if isinstance(payload,bytes) else json.dumps(payload,ensure_ascii=False).encode()
                self.send_response(status);self.send_header('Content-Type',kind)
                self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store')
                self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(body)
            def do_GET(self):
                if self.path=='/':return self.send(200,(ROOT/'index.html').read_bytes(),'text/html; charset=utf-8')
                if self.path=='/api/health':return self.send(200,dict(ok=True,language_model=False,address_space_limit_bytes=100000000))
                self.send(404,dict(error='Not found'))
            def do_POST(self):
                s=None;new_path=None
                try:
                    length=int(self.headers.get('Content-Length','0'))
                    if not 0<length<=32768:raise ValueError('Requête vide ou trop volumineuse.')
                    data=json.loads(self.rfile.read(length))
                    if type(data) is not dict:raise ValueError('Objet JSON attendu.')
                    if self.path=='/api/learn':
                        if len(sessions)>=8:raise ValueError('Huit sessions actives : effacez une session avant de continuer.')
                        examples=data.get('examples')
                        if not isinstance(examples,list) or not all(isinstance(e,dict) for e in examples):raise ValueError('Exemples invalides.')
                        ident=uuid.uuid4().hex;path=Path(tmp)/f'{ident}.sqlite';new_path=path;s=Space(path)
                        begin=time.perf_counter();s.fit(examples)
                        revision=s.revise_if_empty(max_depth=4)
                        sessions[ident]=path
                        return self.send(200,dict(session=ident,state=state(s),seconds=time.perf_counter()-begin,revision=revision))
                    ident=data.get('session')
                    if not isinstance(ident,str) or ident not in sessions:raise ValueError('Session inconnue. Apprenez une procédure d’abord.')
                    if self.path=='/api/delete':
                        sessions.pop(ident).unlink(missing_ok=True);return self.send(200,dict(deleted=True))
                    s=Space(sessions[ident])
                    if self.path=='/api/predict':return self.send(200,s.predict(data.get('input')))
                    if self.path=='/api/question':
                        x=s.get('examples')[0]['input'];kind='text' if type(x) is str else 'int' if type(x) is int else 'list'
                        return self.send(200,s.clarify(generate(kind,7)))
                    if self.path=='/api/observe':
                        s.observe(data.get('input'),data.get('output'))
                        revision=s.revise_if_empty(max_depth=4)
                        return self.send(200,dict(state=state(s),revision=revision))
                    self.send(404,dict(error='Not found'))
                except (ValueError,TypeError,KeyError,json.JSONDecodeError) as e:self.send(400,dict(error=str(e)))
                finally:
                    if s is not None:s.close()
                    if new_path is not None and new_path not in sessions.values():new_path.unlink(missing_ok=True)
        httpd=HTTPServer(('0.0.0.0',port),Handler)
        print(f'Induction lab listening on 0.0.0.0:{port}',flush=True)
        try:httpd.serve_forever()
        finally:httpd.server_close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8082);args=p.parse_args()
    resource.setrlimit(resource.RLIMIT_AS,(100000000,100000000))
    serve(args.port)
