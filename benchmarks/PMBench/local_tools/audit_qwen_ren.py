"""Independent offline replay and physical cache provenance for Qwen REN."""
import argparse, hashlib, json
from collections import Counter
from pathlib import Path
from audit_ren_confirmation import replay, rows
from audit_confirmation_delivery import check_judge
from run_recommendation import digest, write_json
from run_intentions import PROPOSALS_SCHEMA
from qwen_transport import MODEL, TRANSPORT, BASE_URL, payload_for, validate_response
from openmodel_config import TEMPLATE, PROFILE
from compiler_failure import validate_failure_cache
from run_ren_natural import SCHEMA as CONTRACTS_SCHEMA
from ren_public_frames import SCHEMA as FRAMES_SCHEMA

FAILURE_SCHEMAS=dict(compile=PROPOSALS_SCHEMA,compile_condition_contracts=CONTRACTS_SCHEMA,
                     compile_public_observation_frames=FRAMES_SCHEMA)

TRACE_FILES = ('decisions.jsonl','queries.jsonl','recommendations.jsonl','revisions.jsonl',
    'condition_contracts.jsonl','condition_evidence.jsonl','observation_bindings.jsonl',
    'ledger_decisions.jsonl','public_frames.jsonl','situated_evidence.jsonl','compiler_contexts.jsonl',
    'compiler_failures.jsonl')

def provenance(week):
    physical = {}
    totals = {'llm': Counter(), 'jev': Counter()}
    for path in sorted(week.glob('attempt-*/model_calls.jsonl')):
        for row in rows(path):
            physical[digest(row)] = row
            u = row['response'].get('usage', {})
            total = totals[row['kind']]
            total['responses'] += 1
            total['input_tokens'] += u.get('prompt_tokens', u.get('input_tokens',0))
            total['output_tokens'] += u.get('completion_tokens', u.get('output_tokens',0))
            total['cached_input_tokens'] += u.get('prompt_cache_hit_tokens', 0)
            if u.get('cost') is None:
                total['responses_without_returned_cost'] += 1
            else:
                total['returned_cost_usd'] += u['cost']
    chats = {};failed_chats=0;failed_compilers=0
    for path in (week/'natural-chat-cache').glob('*.json'):
        c = json.loads(path.read_text()); req = c['request']
        assert digest(req) == path.stem and c['records']
        assert req['model'] == MODEL and req['thinking'] == 'disabled' and req['transport'] == TRANSPORT
        for row in c['records']:
            assert digest(row) in physical
        last = c['records'][-1]
        assert last['kind'] == 'llm' and last['stage'] == req['stage']
        assert last['request'] == payload_for(req['stage'],req['messages'],req['schema'],req['max_tokens'])
        validate_response(last['response'],req['schema'])
        if 'terminal_failure' in c:
            validate_failure_cache(c)
            assert req['stage'] in FAILURE_SCHEMAS and req['schema']==FAILURE_SCHEMAS[req['stage']]
            failed_chats+=1
        else:
            assert last['response']['choices'][0]['finish_reason'] == 'stop'
            assert json.loads(last['response']['choices'][0]['message']['content']) == c['result']
        chats[path.stem] = c
    judges = 0
    for path in (week/'judge-cache').glob('*.json'):
        c = json.loads(path.read_text())
        assert digest(c['request']) == path.stem
        check_judge(c,physical); judges += 1
    compilers = 0
    for path in (week/'compiler-cache').glob('*.json'):
        c = json.loads(path.read_text())
        assert hashlib.sha256(json.dumps([MODEL,c['request'],PROPOSALS_SCHEMA],sort_keys=True).encode()).hexdigest() == path.stem
        req = dict(stage='compile',model=MODEL,messages=c['request'],schema=PROPOSALS_SCHEMA,
            max_tokens=5500,temperature=0,route='configured-direct',thinking='disabled',transport=TRANSPORT)
        if 'terminal_failure' in c:
            assert 'result' not in c and c['failure_chat_request']==req
            assert c['failure_chat_key']==digest(req)
            failed=chats[digest(req)];validate_failure_cache(failed)
            assert c['terminal_failure']==failed['terminal_failure'] and c['model_records']==failed['records']
            failed_compilers+=1
        else:
            assert c['result'] == chats[digest(req)]['result']
        assert all(digest(row) in physical for row in c['model_records'])
        compilers += 1
    return dict(status='passed',chat_caches=len(chats),judge_caches=judges,compiler_caches=compilers,
        all_cache_responses_physically_logged=True,usage=totals,
        terminal_compile_chat_caches=failed_chats,terminal_compile_outer_caches=failed_compilers)

def check_routes(run,manifest):
    fresh=rows(run/'model_calls.jsonl');routes=rows(run/'route_events.jsonl')
    assert manifest['api_calls']==len(fresh)
    assert routes or not fresh, 'Physical responses require route records'
    for event in routes:
        if event['model']==MODEL:
            assert event['url']==BASE_URL+'/chat/completions'
            assert event['thinking']==(TEMPLATE if PROFILE != 'relay' else None)
        else:
            assert event['model']=='typesafe/jev-1.13'
            assert event['url']=='https://openrouter.ai/api/alpha/decisions'
    return dict(route_events_checked=len(routes),cache_only_no_new_model_calls=not fresh)

def audit(run,dest):
    m = json.loads((run/'manifest.json').read_text())
    assert m['status']=='complete' and m['compiler_model']==MODEL and m['thinking']=='disabled'
    assert m['query_budget']==3 and m['episode_query_cap'] is None
    assert m['candidate_mode']=='remaining' and m['ren_adapter']=='verified-confirmation-v1'
    result = replay(run,dest)
    for prior in sorted(run.parent.glob('attempt-*')):
        if not prior.is_dir() or prior==run: continue
        for name in TRACE_FILES:
            old = rows(prior/name)
            assert rows(run/name)[:len(old)]==old,(str(prior),name)
    result.update(check_routes(run,m))
    result['provenance'] = provenance(run.parent)
    result['all_prior_prefixes_exact'] = True
    write_json(dest/'audit.json',result)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--dest',type=Path,required=True)
    args=parser.parse_args()
    result=audit(args.run.resolve(),args.dest.resolve())
    print(json.dumps({'status':result['status'],'steps':result['steps'],'new_model_calls':0}))
