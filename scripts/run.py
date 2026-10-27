#!/usr/bin/env python3
"""One fresh trajectory, followed by the method's offline audit.

All output goes to an ignored run directory. --dry-run prints commands without
calling a model. Prior attempts/caches are never overwritten or automatically
restarted. Use a fresh directory for each method and seed.
"""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
BENCH=ROOT/'benchmarks/PMBench'
TOOLS=BENCH/'local_tools'
METHODS={
 'single':'single-baseline', 'heartbeat':'heartbeat-proactive',
 'heartbeat30':'heartbeat-auto-30m', 'heartbeat60':'heartbeat-auto-60m',
 'hier':'hier-union-query', 'pis':'pis-official', 'pma':'pma', 'rag':'rag', 'mem0':'mem0', 'ren':'ren',
}

def commands(method,scenario,out):
    py=sys.executable;run=out/'attempt-1';cache=out/'cache'
    common=['--out',str(run),'--cache',str(cache)]
    if method=='ren':
        cmd=[py,str(TOOLS/'run_qwen_ren.py'),'--adapter','verified','--candidate-mode','remaining',
             '--episode-cap','0','--query-policy','round_robin','--query-budget','3','--days','7',
             '--scenario',str(scenario),'--out',str(run),'--compiler-cache',str(out/'compiler-cache'),
             '--judge-cache',str(out/'judge-cache'),'--max-api-calls','1200','--max-input-tokens','4000000','--max-jev-usd','2']
        audit=[py,str(TOOLS/'audit_qwen_ren.py'),'--run',str(run),'--dest',str(out/'audit')]
    elif method in ('pma','rag','mem0'):
        cmd=[py,str(TOOLS/'run_memory_baseline.py'),'run','--method',method,'--scenario',str(scenario),*common]
        audit=[py,str(TOOLS/'numeric_audit.py'),*common]
    elif method=='pis-official':
        cmd=[py,str(TOOLS/'run_author_pis.py'),'run','--scenario',str(scenario),*common]
        audit=[py,str(TOOLS/'run_author_pis.py'),'audit',*common]
    else:
        cmd=[py,str(TOOLS/'run_step3_baseline.py'),'--condition',method,'--scenario',str(scenario),*common]
        audit=[py,str(TOOLS/'audit_step3_baselines.py'),'--run',str(run),'--scenario',str(scenario),'--dest',str(out/'audit')]
    return cmd,audit

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=METHODS,required=True)
    p.add_argument('--profile',choices=['relay','qwen9b','qwen4b','gemma26b'],default='relay')
    p.add_argument('--model',default='gpt-4o-mini-2024-07-18',help='Relay model; local profiles have fixed names')
    p.add_argument('--scenario',type=Path,default=BENCH/'data/synthetic_week_v9.json')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();scenario=a.scenario.resolve();out=a.out.resolve()
    if out.exists():p.error('Output already exists. Retain it; choose a fresh directory.')
    method=METHODS[a.method]
    cmd,audit=commands(method,scenario,out)
    env=dict(os.environ,PMBENCH_PROFILE=a.profile,PMBENCH_MODEL=a.model,
             MEMORY_BASELINES_STOP=str(out/'STOP.json'),PAPER_BASELINES_STOP=str(out/'STOP.json'),
             RELAY_STOP_FILE=str(out/'STOP.json'),QWEN_STOP_FILE=str(out/'STOP.json'))
    plan=dict(method=method,profile=a.profile,model=a.model if a.profile=='relay' else {'qwen9b':'Qwen3.5-9B','qwen4b':'Qwen3.5-4B','gemma26b':'gemma-4-26B-A4B-it'}[a.profile],run=cmd,audit=audit,step_query_cap=3,episode_query_cap=None)
    if a.dry_run:
        print(json.dumps(plan,indent=2));return
    if not (BENCH/'sim/pm_bench.py').exists() or not scenario.exists():
        p.error('Run scripts/bootstrap.py and provide a valid scenario first')
    from dotenv import dotenv_values
    values=dotenv_values(ROOT/'a-mem/.env')
    if a.profile=='relay' or method in ('pma','rag','mem0'):
        if not values.get('OPENAI_BASE_URL') or not values.get('OPENAI_API_KEY'):
            p.error('Configure OPENAI_BASE_URL and OPENAI_API_KEY in a-mem/.env')
        from urllib.parse import urlsplit
        parsed=urlsplit(values['OPENAI_BASE_URL'])
        if parsed.scheme!='https' or not parsed.hostname or parsed.username or parsed.password or 'openrouter' in parsed.hostname.lower():
            p.error('Non-Jev remote calls require the configured HTTPS relay')
    if method=='ren' and not values.get('OPENROUTER_API_KEY'):
        p.error('REN requires OPENROUTER_API_KEY exclusively for its Jev judge')
    out.mkdir(parents=True)
    # Freeze shipped code + benchmark runtime so audits record the whole release.
    tracked=subprocess.check_output(['git','-C',str(ROOT),'ls-files','--cached','--others','--exclude-standard','-z'],text=True).split('\0')
    sources=[ROOT/n for n in tracked if n and (ROOT/n).is_file() and not (ROOT/n).is_relative_to(out)]
    sources+=list((BENCH/'sim').glob('*.py'))
    plan.update(commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
                source_hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources},
                scenario_sha256=hashlib.sha256(scenario.read_bytes()).hexdigest())
    for source in sources:
        target=out/'source'/source.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
    (out/'working-tree.patch').write_text(subprocess.check_output(['git','-C',str(ROOT),'diff','HEAD'],text=True))
    (out/'working-tree-status.txt').write_text(subprocess.check_output(['git','-C',str(ROOT),'status','--short'],text=True))
    (out/'packages.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    (out/'command.json').write_text(json.dumps(plan,indent=2)+'\n')
    with (out/'console.log').open('w') as log:
        result=subprocess.run(cmd,cwd=BENCH,env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode==0:
            result=subprocess.run(audit,cwd=BENCH,env=env,stdout=log,stderr=subprocess.STDOUT)
    status='complete' if result.returncode==0 else 'failed-retained'
    (out/'status.json').write_text(json.dumps(dict(status=status,returncode=result.returncode))+'\n')
    print(status, out)
    raise SystemExit(result.returncode)

if __name__=='__main__':main()
