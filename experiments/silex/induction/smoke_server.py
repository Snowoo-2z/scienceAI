"""Exercise an already-running local lab; delete only sessions created here."""
import argparse
import json
from pathlib import Path
import urllib.error
import urllib.request


def run(base,pid=None):
    def post(path,data,expected=200):
        req=urllib.request.Request(base+'/api/'+path,data=json.dumps(data).encode(),headers={'Content-Type':'application/json'})
        try:
            with urllib.request.urlopen(req,timeout=10) as r:status=r.status;value=json.load(r)
        except urllib.error.HTTPError as e:status=e.code;value=json.load(e)
        assert status==expected,(status,value)
        return value
    checks=[];sessions=[]
    try:
        health=json.load(urllib.request.urlopen(base+'/api/health'));assert health['language_model'] is False;checks.append('health')
        post('predict',{'session':'unknown','input':1},400);checks.append('unknown_session_rejected')
        post('learn',{'examples':[{'input':{},'output':1}]},400);checks.append('invalid_value_rejected')
        learned=post('learn',{'examples':[{'input':'  BONJOUR MONDE  ','output':'bonjour monde'},
                                         {'input':'  LUNDI  ','output':'lundi'}]})
        sid=learned['session'];sessions.append(sid);checks.append('learned_from_examples')
        output=post('predict',{'session':sid,'input':'  SALUT PARIS  '})
        assert output['status']=='consensus' and output['output']=='salut paris';checks.append('new_input_correct')
        q=post('question',{'session':sid});assert q['status']=='question';checks.append('asks_discriminating_question')
        after=post('observe',{'session':sid,'input':' HELLO_WORLD ','output':'hello_world'})
        assert after['state']['hypotheses']<=learned['state']['hypotheses'];checks.append('incremental_filter')
        r=post('predict',{'session':sid,'input':' HELLO_WORLD '})
        assert r['output']=='hello_world';checks.append('observation_persisted_across_requests')
        post('predict',{'session':sid,'input':'x'*257},400);checks.append('resource_limits_rejected')
        post('delete',{'session':sid});sessions.remove(sid)
        post('predict',{'session':sid,'input':1},400);checks.append('delete_effective')
        result=dict(checks=checks,example_result=output,question=q,health=health)
        if pid:
            status=Path(f'/proc/{pid}/status').read_text()
            limits=Path(f'/proc/{pid}/limits').read_text()
            result['server_process_memory']={line.split(':')[0]:line.split(':')[1].strip()
                                              for line in status.splitlines() if line.startswith(('VmRSS:','VmHWM:','VmSize:','VmPeak:'))}
            result['server_address_space_limit']=[line for line in limits.splitlines() if line.startswith('Max address space')][0]
            result['scope']='actual_server_process_not_shell_parent; OS/page_cache excluded'
        return result
    finally:
        for sid in sessions:post('delete',{'session':sid})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:8082');p.add_argument('--pid',type=int)
    a=p.parse_args();print(json.dumps(run(a.url,a.pid),ensure_ascii=False,indent=2))
