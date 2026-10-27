"""Unmodified author PIS backend/session over the fixed local PM-Bench.

Author commit 43e9b426a478bdd6b4e6fb12cb609ca26bd71ff9. Only the public
integration, audited relay, B3 limits and replay identity tape live here.
"""
import argparse
from contextlib import ExitStack,redirect_stdout
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
from types import SimpleNamespace
import uuid
from unittest.mock import patch

from release_common import ROOT,WORKSPACE,write,rows,sha,pm,MODEL,append
from relay_transport import RelayTransport,configure,make_client
from run_official_baselines import StudyStop
from step3_budget import QueryBudget,instrument,guarded_resolver

AUTHOR=WORKSPACE/'benchmarks/PIS'
AUTHOR_COMMIT='43e9b426a478bdd6b4e6fb12cb609ca26bd71ff9'
AUTHOR_PM=WORKSPACE/'.deps/author_pm_bench.py'
sys.path.insert(0,str(AUTHOR/'code'))
from pm_memory import intention_store as author_store
from pm_memory.session import MemorySession
from llm_env import _merge_chat_extra_body
from llm_cost import current_role


def load_author_pm():
    spec=importlib.util.spec_from_file_location('author_pm_decision',AUTHOR_PM)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class IdentityTape:
    """Retain actual random author IDs across exact prefix recovery/replay."""
    def __init__(self,path,offline=False):
        self.path=Path(path);self.offline=offline;self.position=0
        self.values=json.loads(self.path.read_text()) if self.path.exists() else []

    def uuid4(self):
        if self.position==len(self.values):
            if self.offline:raise StudyStop('Offline author identity tape exhausted')
            self.values.append(uuid.uuid4().hex);write(self.path,self.values)
        value=uuid.UUID(hex=self.values[self.position]);self.position+=1
        return value


class NativeClient:
    def __init__(self,transport,budget):
        self.transport,self.budget=transport,budget
        self.chat=SimpleNamespace(completions=SimpleNamespace(create=self.create))
        self.ids=[];self.channels=[]

    def with_options(self,**kwargs):return self

    def create(self,**kwargs):
        request=deepcopy(kwargs)
        # Exactly the author's GPT-specific removal of sglang-only extras.
        _merge_chat_extra_body(request)
        if current_role()=='choose':
            request=self.budget.request(request)
            if not self.budget.remaining:
                schema=pm.build_action_schema(self.ids,self.channels)
                schema['properties']['action']['enum']=['choose']
                schema['properties']['channel']['enum']=['NONE']
                request['response_format']=dict(type='json_schema',json_schema=dict(name='pm_action',strict=True,schema=schema))
        return self.transport.create(**request)


def prepare_environment(offline=False):
    # configure() installs the socket prohibition only when this flag is
    # already set. The author transport and collector share one circuit breaker.
    if offline:os.environ['MATCHED_OFFLINE']='1'
    stop=os.environ.get('PAPER_BASELINES_STOP')
    if stop:os.environ['RELAY_STOP_FILE']=stop
    configure()


def normalize_action(action,ids,channels):
    if not isinstance(action,dict):raise RuntimeError('Author decision is not an object')
    act=action.get('action')
    tasks=pm.normalize_task_ids(action)
    if act in ('query_state','check_time'):
        if 'choice' not in action:raise RuntimeError('Author query is missing choice')
        channel='clock' if act=='check_time' else action.get('channel')
        if action['choice']!='NONE' or tasks or channel not in channels:
            # These are explicit native retry branches: Invalid choice,
            # Invalid task_ids, and Invalid channel. Preserve the raw action
            # so the unchanged simulator supplies that feedback. No provider
            # call, guessed channel, or task execution is added here.
            return deepcopy(action)
        return dict(action=act,choice='NONE',task_ids=[],channel=channel)
    # MAX_ACTION_TASK_IDS constrains the prompt/schema, not native execution.
    # Keep an overlong cached action intact; native run_llm maps/deduplicates
    # its handles without truncation or an extra model retry.
    if act=='choose':
        if action.get('choice') not in ('A','B','C') or action.get('channel') not in (None,'NONE'):
            # Native run_llm returns Invalid choice/channel feedback and
            # requests the next action without advancing or querying state.
            # Preserve the original response for that native feedback path.
            return deepcopy(action)
        # Native run_llm maps the public handles and skips unknown tokens.
        # Preserve every model token here; only that unchanged simulator
        # decides what executes. Do not invent a handle or retry the decision.
        return dict(action='choose',choice=action['choice'],task_ids=tasks,channel='NONE')
    raise RuntimeError('Unsupported author action')


def execute(scenario_path,out,cache,offline=False):
    out,cache=Path(out).resolve(),Path(cache).resolve()
    out.mkdir(parents=True,exist_ok=False);cache.mkdir(parents=True,exist_ok=True)
    scenario_path=Path(scenario_path).resolve();scenario=json.loads(scenario_path.read_text())
    shutil.copyfile(scenario_path,out/'scenario.json')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=AUTHOR,text=True).strip()
    if commit!=AUTHOR_COMMIT:raise RuntimeError('Author source revision changed')
    dirty=subprocess.check_output(['git','status','--porcelain'],cwd=AUTHOR,text=True).strip()
    if dirty:raise RuntimeError('Author source checkout is not clean')
    prepare_environment(offline)
    transport=RelayTransport(make_client(),out/'transport',cache/'requests')
    budget=QueryBudget(out,3,None);client=NativeClient(transport,budget)
    tape=IdentityTape(cache/'identities.json',offline)
    native_pm=load_author_pm()
    backend=author_store.IntentionStoreBackend(store_path=str(out/'intentions.json'),
        max_inject_chars=8000,llm_client=client,llm_model=MODEL,
        use_llm_update=True,use_llm_extract=True,use_llm_match=True,use_llm_cadence=False,
        gate_time_due=False,force_query_channels=False,llm_response_format='json_object',
        retrieval_mode='due',ignore_status=False)
    session=MemorySession(backend,top_k=5,log_path=str(out/'memory.jsonl'),keep_full_history=False,
        force_check_time=True,force_query_channels=False,gate_time_due=False)
    def decision(c,m,messages,ids,channels,b,allow_heartbeat=False,max_tokens=1024,response_format_mode='json_object'):
        client.ids=list(ids);client.channels=list(channels)
        try:
            # Author's adapter only changed request_model_action/run_llm to add
            # response-format support. Keep its request function and the local
            # unchanged simulator, providers, menus and scorer.
            return native_pm.request_model_action(client,m,messages,ids,channels,'sglang',
                allow_heartbeat=False,max_tokens=1024,response_format_mode='json_object')
        except native_pm.InvalidModelResponseError as e:
            raise RuntimeError('Invalid author model response; evaluator fallback prohibited') from e
    wrapped=session.wrap(decision)
    def callback(c,m,messages,ids,channels,b,**kwargs):
        action=normalize_action(wrapped(c,m,messages,ids,channels,b,**kwargs),ids,channels)
        append(out/'decisions.jsonl',dict(day=budget.step[0],step_id=budget.step[1],action=action))
        return action
    files=[Path(__file__),ROOT/'sim/pm_bench.py',AUTHOR_PM,
        AUTHOR/'code/pm_memory/intention_store.py',AUTHOR/'code/pm_memory/session.py',
        AUTHOR/'code/pm_memory/common.py',AUTHOR/'code/llm_env.py',AUTHOR/'code/llm_cost.py']
    manifest=dict(status='running',method='pis-official',scope='author-release-on-fixed-PMBench',
        author_url='https://github.com/ChengcanWu/prospective-memory-reaearch',author_commit=commit,
        model=MODEL,api_base_url=str(transport.client.base_url),credential_source='selected release profile; relay key from a-mem/.env or local sentinel',
        query_step_cap=3,query_episode_cap=None,source_hashes={str(p):sha(p) for p in files},
        scenario_sha256=sha(scenario_path),runtime_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        configuration=dict(retrieval='due',top_k=5,max_inject_chars=8000,response_format='json_object',
            force_check_time=True,force_query_channels=False,gate_time_due=False,ignore_status=False),
        adaptations=['unchanged fixed benchmark simulator/providers/scorer',
            'B3 common budget notice and choose-only schema at exhausted cap',
            'audited user relay and bounded faults, no oracle fallback',
            'record/replay actual random identity tape; no semantic ID changes'],
        started_utc=pm.now_utc_iso())
    write(out/'manifest.json',manifest)
    try:
        session.open()
        with ExitStack() as stack:
            stack.enter_context(patch.object(author_store,'uuid',SimpleNamespace(uuid4=tape.uuid4)))
            stack.enter_context(patch.object(pm,'build_llm_client',return_value=client))
            stack.enter_context(patch.object(pm,'request_model_action',side_effect=callback))
            stack.enter_context(patch.object(pm,'_matched_budget',budget,create=True))
            stack.enter_context(patch.object(pm,'resolve_state_query_items',guarded_resolver(pm.resolve_state_query_items,budget)))
            run=instrument(pm,'run_llm',budget,out)
            with (out/'runtime.log').open('w') as f,redirect_stdout(f):
                entries=run(scenario,str(out/'actions.jsonl'),MODEL,None,5,'sglang',None,None,max_tokens=1024)
        assert len(entries)==sum(len(d['steps']) for d in scenario['days'])
        counts=[sum(pm.get_action_state_query_counts(e).values()) for e in entries]
        assert all(v<=3 for v in counts) and sum(counts)==budget.total
        score,per_day,steps=pm.score_log(scenario,entries)
        write(out/'score.json',dict(summary=score,per_day=per_day,steps=steps))
        manifest.update(status='complete',steps=len(entries),query_total=budget.total,
            identities_consumed=tape.position,identity_tape_sha256=sha(tape.path),finished_utc=pm.now_utc_iso())
    except BaseException as e:
        manifest.update(status='failed',error_type=type(e).__name__,error=str(e),finished_utc=pm.now_utc_iso())
        write(out/'manifest.json',manifest)
        raise
    finally:session.close()
    write(out/'manifest.json',manifest)
    return out


def canonical_memory_rows(records):
    """Canonicalize only the author's set-valued diagnostic due-ID field.

    IntentionStoreBackend builds this field with list(due_ids), where due_ids
    is a set. Its order is not used by the policy. All other values, including
    other list orders and row order, remain exact. Never rewrite source logs.
    """
    result=deepcopy(records)
    for record in result:
        if 'recall_match_due_ids' in record:
            values=record['recall_match_due_ids']
            assert isinstance(values,list) and all(isinstance(v,str) for v in values)
            assert len(values)==len(set(values)), 'Duplicate diagnostic due ID'
            record['recall_match_due_ids']=sorted(values)
    return result


def audit(run,cache):
    run,cache=Path(run),Path(cache)
    m=json.loads((run/'manifest.json').read_text());assert m['status']=='complete'
    scenario=json.loads((run/'scenario.json').read_text());entries=pm.read_log(str(run/'actions.jsonl'))
    score,per_day,steps=pm.score_log(scenario,entries)
    assert json.loads((run/'score.json').read_text())==dict(summary=score,per_day=per_day,steps=steps)
    assert all(sum(pm.get_action_state_query_counts(e).values())<=3 for e in entries)
    dest=run.parent/(run.name+'-audit')
    if (dest/'validation.json').exists():
        v=json.loads((dest/'validation.json').read_text());assert v['action_sha256']==sha(run/'actions.jsonl');return v
    dest.mkdir(exist_ok=False)
    for d in scenario['days']:
        for s in d['steps']:s['groundtruth']={'private_label':'must not affect the policy'}
    write(dest/'perturbed.json',scenario)
    replay=execute(dest/'perturbed.json',dest/'replay',cache,True)
    assert pm.read_log(str(replay/'actions.jsonl'))==entries
    for name in ('decisions.jsonl','memory.jsonl','budget_events.jsonl'):
        original,replayed=rows(run/name),rows(replay/name)
        if name=='memory.jsonl':
            original,replayed=canonical_memory_rows(original),canonical_memory_rows(replayed)
        assert original==replayed,name
    assert json.loads((run/'intentions.json').read_text())==json.loads((replay/'intentions.json').read_text())
    result=dict(status='passed',author_commit=m['author_commit'],official_score_recomputed=True,
        exact_no_network_private_label_replay=True,steps=m['steps'],query_total=m['query_total'],
        memory_comparison='exact except set order of recall_match_due_ids; membership and uniqueness checked',
        action_sha256=sha(run/'actions.jsonl'))
    write(dest/'validation.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['run','audit'])
    p.add_argument('--scenario',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--offline',action='store_true')
    a=p.parse_args()
    if a.command=='audit':audit(a.out,a.cache)
    else:execute(a.scenario,a.out,a.cache,a.offline)
