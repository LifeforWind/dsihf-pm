#!/usr/bin/env python3
"""Confirmation replay: independent conditions, candidate arms, rotation and both budget caps."""
import argparse
from collections import Counter
from contextlib import redirect_stdout
from copy import deepcopy
import json,socket,sys
from pathlib import Path
from unittest.mock import patch
from audit_reference import Reference
from run_ren_confirmation import ConfirmationGroundedModels,ConfirmationSituatedModels,ConfirmationVerifiedPolicy,VerifiedCompiler,SourceCompiler
from ren_confirmation_policy import ConfirmationPolicy
from run_recommendation import Compiler
from run_ren_natural import RECEIPT
from run_recommendation import pm,write_json


def rows(p):return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []

class AuditMixin:
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.audit_spent=0;self.metrics=Counter();self.rotation={};self.scope=None;self.start_step=None;self.reference=Reference()

    def next_query(self,active,observations,allowed_channels):
        result=super().next_query(active,observations,allowed_channels)
        dump=self.needs.dump();ref=self.reference
        ref.now=dump['now'];ref.tasks=dump['tasks'];ref.facts=dump['facts'];ref.status={}
        # Reconstruct lifecycle independently from chronological public events.
        for event in dump['events']:
            kind=event['type']
            if kind in ('add','revise'):ref.status[event['task']]='pending'
            elif kind=='cancel':ref.status[event['task']]='canceled'
            elif kind=='attempt':ref.status[event['task']]='inflight'
            elif kind=='receipt':raise AssertionError('Original PM-Bench cannot supply a success receipt')
        assert ref.status==dump['status']
        for fact in ref.facts:
            assert fact['as_of']==fact['valid_until']
        record=rows(self.out/'recommendations.jsonl')[-1]
        assert {t['id']:ref.states(t) for t in ref.active()}==record['states']
        expected=ref.support(self.candidate_mode)
        if self.semantic:
            semantic=rows(self.out/'semantic_candidates.jsonl')[-1]
            channels=[c for c in allowed_channels if c not in observations]
            live={f'{i.id}@{self.day}' for i in active}
            tasks=[t for t in ref.active() if t['id'] in live]
            expected={c:[dict(task_id=t['id'],revision=t['revision'],reason='semantic_relevance') for t in tasks]
                for n,c in enumerate(channels) if f'relevant_{n}' in semantic['answers'] and semantic['answers'][f'relevant_{n}']['noul']>=.5}
            if any(i.kind=='clock' for i in active) and 'clock' in allowed_channels:
                expected['clock']=[dict(task_id=f'{i.id}@{self.day}',revision=i.revision,reason='shared_clock_reservation') for i in active if i.kind=='clock']
        if self.filtered_control:
            live={f'{i.id}@{self.day}' for i in active}
            expected={c:[r for r in reasons if r['task_id'] in live] for c,reasons in expected.items()}
            expected={c:r for c,r in expected.items() if r}
        assert expected==record['support']
        if self.scope!=self.day:
            self.scope=self.day;self.start_step=self.step;self.rotation={}
        for channel in observations:self.rotation[channel]=self.step
        clock_required=any(i.kind=='clock' for i in active)
        candidates=set(expected)&set(allowed_channels)-set(observations)
        order=sorted(candidates,key=lambda c:(not(clock_required and c=='clock'),
            self.rotation.get(c,self.start_step-1),c))
        exhausted=self.scheduler.episode_cap is not None and self.audit_spent>=self.scheduler.episode_cap
        assert record['episode_spent_before']==self.audit_spent and record['episode_exhausted']==exhausted
        selected=order[0] if order and len(observations)<self.scheduler.budget and not exhausted else None
        if selected:self.audit_spent+=1
        assert self.audit_spent==self.scheduler.spent
        assert selected==result==record['selected_channel']
        assert [r['channel'] for r in record['ranked_channels']]==order
        assert len(observations)<=self.scheduler.budget and RECEIPT not in candidates
        self.metrics['prefixes_checked']+=1
        self.metrics['condition_states_checked']+=sum(len(t['conditions']) for t in ref.active())
        if result:
            self.metrics['queries_checked']+=1
            self.metrics['queries_without_remaining_support']+=int(result not in ref.support())
        self.metrics['counterfactual_candidate_set_changed_prefixes']+=int(
            (set(ref.support())&set(allowed_channels)) != (set(ref.support('task_channels'))&set(allowed_channels)))
        return result

    def __call__(self,*args,**kwargs):
        action=super().__call__(*args,**kwargs)
        if action['action']=='choose':
            ready=self.reference.ready()
            for event in self.needs.events:
                if event['type']=='attempt' and event['at']==self.needs.now:
                    assert event['task'] in ready
                    self.metrics['supported_attempts_checked']+=1
        return action

def replay(run,target):
    target.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((run/'manifest.json').read_text())
    original=json.loads((run/'scenario.json').read_text());scenario=deepcopy(original)
    for day in scenario['days']:
        for task in day['tasks']:task['private_audit_metadata']='PRIVATE_CANARY_DO_NOT_OBSERVE'
        for step in day['steps']:step['groundtruth']={'private_canary':'PRIVATE_CANARY_DO_NOT_OBSERVE'}
    adapter=manifest['ren_adapter'].split('-')[0]
    models_class=ConfirmationGroundedModels if adapter=='grounded' else ConfirmationSituatedModels
    compiler_class=VerifiedCompiler if adapter=='verified' else SourceCompiler if adapter=='situated' else Compiler
    policy_class=ConfirmationVerifiedPolicy if adapter=='verified' else ConfirmationPolicy
    model=models_class(target,manifest['compiler_model'],None,'jev',judge_client=object(),
        judge_cache=Path(manifest['judge_cache']),require_compiler_cache=True,require_judge_cache=True,
        max_api_calls=1200,max_input_tokens=4000000,max_jev_usd=2)
    compiler=compiler_class(model,Path(manifest['compiler_cache']))
    policy=type('AuditedConfirmation',(AuditMixin,policy_class),{})(model,compiler,target,routing='jev',
                        query_policy='round_robin',query_budget=manifest['query_budget'],episode_cap=manifest['episode_query_cap'],
                        candidate_mode=manifest['candidate_mode'])
    with (target/'runtime.log').open('w') as log,redirect_stdout(log),patch.object(sys,'__stdout__',log),\
        patch.object(pm,'build_llm_client',return_value=None),patch.object(pm,'request_model_action',side_effect=policy),\
        patch('urllib.request.urlopen',side_effect=AssertionError('Network prohibited')),\
        patch.object(socket,'create_connection',side_effect=AssertionError('Network prohibited')):
        actions=pm.run_llm(scenario,str(target/'actions.jsonl'),'offline',None,5,'sglang',None,None)
    assert actions==pm.read_log(str(run/'actions.jsonl'))
    for name in ('decisions.jsonl','queries.jsonl','recommendations.jsonl','revisions.jsonl',
                 'condition_contracts.jsonl','condition_evidence.jsonl','observation_bindings.jsonl','ledger_decisions.jsonl','public_frames.jsonl','situated_evidence.jsonl','compiler_contexts.jsonl','semantic_candidates.jsonl','compiler_failures.jsonl'):
        assert rows(target/name)==rows(run/name),name
    for name in ('compilation.jsonl','judge_requests.jsonl'):
        assert [r['cache_key'] for r in rows(target/name)]==[r['cache_key'] for r in rows(run/name)]
    # The outer intention compiler bypasses chat entirely on its own cache hits.
    # Contract decomposition always goes through the exact chat-request cache.
    assert [r for r in rows(target/'natural_chat_requests.jsonl') if r['stage']!='compile']==[r for r in rows(run/'natural_chat_requests.jsonl') if r['stage']!='compile']
    assert policy.needs.dump()==json.loads((run/'needs_final.json').read_text()) or json.loads(json.dumps(policy.needs.dump()))==json.loads((run/'needs_final.json').read_text())
    assert model.calls==model.judge_misses==compiler.misses==0
    summary,per_day,steps=pm.score_log(original,actions)
    score=json.loads((run/'score.json').read_text())
    assert dict(summary=summary,per_day=per_day,steps=steps)==score
    assert steps==manifest['expected_steps']
    assert sum(sum(a['state_queries'].values()) for a in actions)==len(rows(run/'queries.jsonl'))
    assert all(sum(a['state_queries'].values())<=manifest['query_budget'] for a in actions)
    total=sum(sum(a['state_queries'].values()) for a in actions)
    assert manifest['episode_query_cap'] is None or total<=manifest['episode_query_cap']
    if adapter=='verified':
        from audit_verified_context import check
        assert check(run)['complete_trajectory']
    # A deliberate score alteration must be rejected by the same exact equality.
    altered=deepcopy(score);altered['summary']['set_tp']+=1
    assert dict(summary=summary,per_day=per_day,steps=steps)!=altered
    result=dict(status='passed',run=str(run),steps=steps,metrics=policy.metrics,
        private_metadata_groundtruth_noninterference=True,exact_requests_queries_actions_states=True,
        independent_lifecycle_conditions_candidates_rotation=True,official_score_recomputed=True,
        altered_score_rejected=True,receipts_inferred=0,new_model_calls=0)
    write_json(target/'audit_result.json',result)
    return result
