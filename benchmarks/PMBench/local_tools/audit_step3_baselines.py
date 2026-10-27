"""Score, budget, provenance and cache-only private-label perturbation audit."""
import argparse
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from dotenv import dotenv_values
from run_official_baselines import ROOT, MODEL, PROVIDER, rows, write, digest
from run_amem_small import pm

from release_common import sha

def action_entries(path):
    # Upstream hierarchy includes real host logging time in each action row.
    # This is not the simulation clock; preserve every other field exactly.
    return [{k:v for k,v in row.items() if k!='timestamp_pst'} for row in pm.read_log(str(path))]



def audit_run(run, scenario_path, dest):
    m=json.loads((run/'manifest.json').read_text());assert m['status']=='complete'
    scenario=json.loads(scenario_path.read_text());entries=pm.read_log(m['action_log'])
    expected=[(d['name'],s['id']) for d in scenario['days'] for s in d['steps']]
    assert [(e['day'],e['step_id']) for e in entries]==expected
    score,per_day,steps=pm.score_log(scenario,entries)
    saved=json.loads((run/'score.json').read_text())
    assert dict(summary=score,per_day=per_day,steps=steps)==saved
    overlay=Path(m['query_overlay'])
    events=rows(overlay/'budget_events.jsonl');query_events=[r for r in events if r['event']=='query']
    counts=Counter((r['day'],r['step_id'],r['channel']) for r in query_events)
    assert m['query_episode_cap'] is None and m['query_step_cap']==3
    for entry in entries:
        qs=pm.get_action_state_query_counts(entry)
        assert sum(qs.values())<=m['query_step_cap']
        observed={ch:n for (day,step,ch),n in counts.items() if (day,step)==(entry['day'],entry['step_id'])}
        assert dict(qs)==observed
    assert len(query_events)==m['query_total']==score['state_query_calls']
    bases=[run.parent]
    if m.get('legacy_base'):bases.append(Path(m['legacy_base']))
    physical={digest(r):r for base in bases for p in base.glob('attempt-*/**/model_calls.jsonl') for r in rows(p)}
    cache=run.parent/'cache';accepted=0
    for path in cache.rglob('*.json'):
        record=json.loads(path.read_text())
        if not isinstance(record,dict) or 'wire_request' not in record:continue
        assert digest([record.get('canonical_request',record['wire_request']),record['occurrence']])==path.stem
        assert digest(record) in physical,('unlogged cache',path)
        wire=record['wire_request'];response=record['response']
        if record.get('route')=='local-qwen':
            from relay_transport import validate_base, EXPECTED, relay_request
            validate_base(record['api_base_url'])
            assert wire==relay_request(record['official_request'])
            assert wire['model']==EXPECTED and 'provider' not in wire.get('extra_body',{})
        else:
            raise AssertionError('New B3 baseline cache must originate on user relay')
        assert record['identity_accepted']
        assert response['model'].split('/')[-1]==MODEL.split('/')[-1]
        if record.get('route')!='local-qwen':assert response.get('provider','').lower()=='openai'
        accepted+=1
    if m['condition']=='amem':
        runtime=json.loads((run/'memory_runtime.json').read_text())
        for name,path in runtime['loaded_modules'].items():
            assert '/.worktrees/amem-native-ceffb86/agentic_memory/' in path
            assert sha(path)==runtime['source_hashes'][name]
        final=json.loads((run/'memory_final.json').read_text())
        assert len(final['notes'])==len(rows(run/'memory_writes.jsonl'))>0
        assert final['evolution_count']>0
    dest.mkdir(parents=True,exist_ok=True)
    replay=dest/'replay'
    altered=deepcopy(scenario)
    for day in altered['days']:
        for step in day['steps']:
            step['groundtruth']={'status':'AUDIT_PRIVATE_CANARY','actions':['DO_NOT_LEAK']}
    canary=dest/'private_canary_scenario.json';write(canary,altered)
    if not (replay/'manifest.json').exists():
        cmd=[sys.executable,str(ROOT/'local_tools/run_step3_baseline.py'),
            '--condition',m['condition'],'--scenario',str(canary),'--out',str(replay),'--cache',str(cache),
            '--step-cap',str(m['query_step_cap'])]
        with (dest/'replay.console.log').open('w') as log:
            result=subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,MATCHED_OFFLINE='1',PYTHONHASHSEED='0'),stdout=log,stderr=subprocess.STDOUT)
        assert result.returncode==0,'Offline replay failed: '+str(dest)
    rm=json.loads((replay/'manifest.json').read_text());assert rm['status']=='complete'
    assert action_entries(rm['action_log'])==action_entries(m['action_log'])
    assert rows(Path(rm['query_overlay'])/'budget_events.jsonl')==events
    assert not any(rows(p) for p in replay.rglob('model_calls.jsonl'))
    def requests(root):
        return Counter(digest(r.get('canonical_request',r['wire_request'])) for p in root.rglob('requests.jsonl') for r in rows(p))
    assert requests(run)==requests(replay)
    result=dict(status='passed',run=str(run),steps=steps,queries=len(query_events),
        accepted_cache_files=accepted,private_label_canary_and_actions_equal=True,new_model_calls=0)
    write(dest/'audit.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--scenario',type=Path,required=True)
    p.add_argument('--dest',type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(audit_run(a.run.resolve(),a.scenario.resolve(),a.dest.resolve())))
