#!/usr/bin/env python3
"""Reconstruct accepted compiler context from proposal/decision logs, without APIs."""
from copy import deepcopy
import argparse,json
from pathlib import Path
from run_recommendation import digest,write_json
from run_intentions import IntentionStore,prospective
from release_common import rows

def check(run,contexts_override=None):
    manifest=json.loads((run/'manifest.json').read_text())
    complete=manifest['status']=='complete'
    contexts=contexts_override if contexts_override is not None else rows(run/'compiler_contexts.jsonl')
    contexts={c['evidence_id']:c for c in contexts}
    compilations={c['evidence_id']:c for c in rows(run/'compilation.jsonl')}
    decisions={r['step_ordinal']:r for r in rows(run/'decisions.jsonl')}
    trial=IntentionStore();checks=0;steps=0;rejections=0
    for r in rows(run/'revisions.jsonl'):
        step=r['step']
        if step not in decisions:break
        evidence_id=f'obs_{step:03}';day=r['store']['evidence'][evidence_id]['day']
        trial.observe(evidence_id,r['evidence'],day,step)
        for n,batch in enumerate(r['verification_batches']):
            key=f'{evidence_id}_{n}';ctx=contexts[key]
            assert ctx['accepted_store_sha256']==digest(trial.dump()),('context hash',key)
            expected=[dict(id=i.id,revision=i.revision,status=i.status,**i.spec()) for i in trial.intents.values()]
            assert ctx['intentions']==expected,('rejected or stale intention in context',key)
            cache=Path(manifest['compiler_cache'])/(compilations[key]['cache_key']+'.json')
            request=json.loads(cache.read_text())['request']
            state=json.loads(request[-1]['content'])
            assert state['observation']==batch['source']
            assert state['existing']==[dict(id=i.id,status=i.status,**i.spec()) for i in trial.intents.values()],key
            checks+=1
            for index in batch['indices']:
                accepted=r['answers'][f'patch_{index}']['noul']>=.5
                trial.apply(r['proposals'][index],evidence_id,accepted=accepted)
                rejections+=int(not accepted)
        assert trial.dump()==r['store'],('proposal reconstruction',step)
        decision=decisions[step]
        for n,item in enumerate(decision['state']['intentions']):
            answers=decision['answers'];handle=answers[f'action_{n}']['choice']
            due=item['kind']!='event' or answers[f'cue_{n}']['noul']>=.5
            if due and handle!='NONE':
                assert handle in decision['action']['task_ids']
                trial.attempted(item['id'],day,step,handle)
        assert trial.dump()==decision['store'],('attempt reconstruction',step)
        steps+=1
    if complete:assert steps==manifest['expected_steps']
    return dict(status='passed',complete_trajectory=complete,steps=steps,compiler_batches=checks,
        rejected_proposals_excluded_from_future_intentions=rejections,new_model_calls=0)

def main(run,out):
    result=check(run)
    altered=deepcopy(rows(run/'compiler_contexts.jsonl'));altered[0]['intentions'].append({'id':'injected_unverified'})
    try:check(run,altered)
    except AssertionError:result['injected_unverified_context_rejected']=True
    else:raise AssertionError('Negative context control was not rejected')
    write_json(out,result);print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.run.resolve(),a.out.resolve())
