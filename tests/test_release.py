"""Release boundaries: no model calls, fixed protocol, all ten entry mappings."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('release_runner',ROOT/'scripts/run.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)

@pytest.mark.parametrize('method',runner.METHODS)
def test_cli_dry_run_does_not_create_outputs(tmp_path,method):
    out=tmp_path/method
    result=subprocess.run([sys.executable,str(ROOT/'scripts/run.py'),'--method',method,'--out',str(out),'--dry-run'],capture_output=True,text=True,check=True)
    plan=json.loads(result.stdout)
    assert plan['step_query_cap']==3 and plan['episode_query_cap'] is None
    assert Path(plan['run'][1]).exists() and Path(plan['audit'][1]).exists()
    assert not out.exists()
    if method=='ren':
        command=plan['run']
        assert command[command.index('--candidate-mode')+1]=='remaining'
        assert command[command.index('--query-budget')+1]=='3'
        assert command[command.index('--episode-cap')+1]=='0'

def test_existing_output_rejected_even_during_dry_run(tmp_path):
    before=tmp_path/'keep.txt';before.write_text('old attempt')
    result=subprocess.run([sys.executable,str(ROOT/'scripts/run.py'),'--method','ren','--out',str(tmp_path),'--dry-run'],capture_output=True,text=True)
    assert result.returncode and before.read_text()=='old attempt'

def test_relay_wire_and_jev_route_without_credentials():
    script='''import sys
sys.path.insert(0,sys.argv[1])
from qwen_wire import wire,MODEL
from qwen_transport import check_route,payload_for,BASE_URL
request=wire(dict(model=MODEL,messages=[dict(role='system',content='keep'),dict(role='user',content='public')]))
assert 'extra_body' not in request
assert request['messages'][0]['content']=='keep'
p=payload_for('compile',request['messages'],{'type':'object','properties':{},'required':[],'additionalProperties':False},100)
assert 'extra_body' not in p
check_route(BASE_URL+'/chat/completions',p)
try: check_route('https://openrouter.ai/api/v1/chat/completions',p)
except AssertionError: pass
else: raise AssertionError('non-Jev routing escaped')
check_route('https://openrouter.ai/api/alpha/decisions',{'model':'typesafe/jev-1.13'})
'''
    import os
    env=dict(os.environ,PMBENCH_PROFILE='relay',PMBENCH_MODEL='gpt-4o-mini-2024-07-18')
    subprocess.run([sys.executable,'-c',script,str(ROOT/'benchmarks/PMBench/local_tools')],env=env,check=True)

def test_embedding_redirect_does_not_forward_credentials():
    import httpx
    sys.path.insert(0,str(ROOT/'benchmarks/PMBench/local_tools'))
    from embedding_redirect import EmbeddingRedirectTransport
    seen=[]
    def respond(request):
        seen.append(str(request.url))
        return httpx.Response(301,headers={'location':'https://different.example/v1/embeddings'})
    transport=EmbeddingRedirectTransport(httpx.MockTransport(respond))
    with httpx.Client(transport=transport,follow_redirects=False) as client:
        response=client.post('https://configured.example/v1/embeddings',headers={'authorization':'Bearer dummy'},json={'input':['public']})
    assert response.status_code==301
    assert seen==['https://configured.example/v1/embeddings']
    assert transport.last_route['redirects']==[]
