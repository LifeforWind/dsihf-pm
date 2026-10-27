"""A truncated extraction never supplies edits or causes answer resampling."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import tempfile,unittest,json,io,http.client,urllib.request,os
from compiler_failure import failure_cache_entry,validate_failure_cache,TruncatedCompilation
from ren_source_compiler import SourceCompiler
from run_ren_natural import NaturalModels,NaturalRENPolicy,SCHEMA as CONTRACTS_SCHEMA
from ren_public_frames import normalize,SCHEMA as FRAMES_SCHEMA
from run_intentions import Models,PROPOSALS_SCHEMA
from openmodel_config import MODEL
from qwen_transport import payload_for,map_disconnect
from audit_qwen_ren import provenance,check_routes

def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []

def intent(op='add'):
    return dict(op=op,id='invoice',source_quote='Pay invoice when approval arrives.',spec=dict(
        action='Pay invoice',trigger='Approval arrives.',day='Monday',kind='event',at=None,
        channel='email',comparator=None,threshold=None,parent_id=None,dependency=None,offset_minutes=None))

class TruncatedCompilerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.week=Path(self.temp.name);self.out=self.week/'attempt-1';self.out.mkdir()
        self.model=NaturalModels(self.out,MODEL,None,'jev',judge_cache=self.week/'judge-cache')
        self.compiler=SourceCompiler(self.model,self.week/'compiler-cache')
        self.compiler.draft.observe('initial','Pay invoice when approval arrives.','Monday',1)
        self.compiler.draft.apply(intent(),'initial')
        self.before=deepcopy(self.compiler.draft.dump())
        self.invocations=0

    def truncated(self,model,stage,messages,schema,max_tokens):
        self.invocations+=1
        # Even syntactically complete JSON is unusable when the provider says length.
        raw=dict(model=MODEL,choices=[dict(finish_reason='length',message=dict(
            content=json.dumps(dict(proposals=[intent('cancel')]))))],
            usage=dict(prompt_tokens=20,completion_tokens=5500))
        model.record(stage,payload_for(stage,messages,schema,max_tokens),raw,0.,'llm')
        raise RuntimeError('Incomplete model response: length')

    def fail_once(self):
        with patch.object(Models,'chat',lambda model,*args,**kw:self.truncated(model,*args,**kw)):
            result=self.compiler.compile('cancellation','Do not pay this invoice.','Monday',2,['email'])
        self.assertEqual(result,[])

    def test_entire_batch_rejected_and_previous_memory_preserved(self):
        self.fail_once()
        self.assertEqual(self.compiler.draft.dump(),self.before)
        self.assertEqual(self.invocations,1);self.assertEqual(self.model.calls,1)
        event=rows(self.out/'compiler_failures.jsonl')[0]
        self.assertEqual(event['parsed_proposals'],0)
        self.assertEqual(event['max_tokens'],5500)
        self.assertEqual(event['operation'],'reject_entire_truncated_compile_preserve_store')
        compilation=rows(self.out/'compilation.jsonl')[0]
        self.assertEqual(compilation['proposals'],[]);self.assertEqual(compilation['rejected'],[])
        self.assertIn('failure',compilation)

    def test_failure_caches_remain_distinct_and_physically_auditable(self):
        self.fail_once()
        chat=json.loads(next((self.week/'natural-chat-cache').glob('*.json')).read_text())
        outer=json.loads(next((self.week/'compiler-cache').glob('*.json')).read_text())
        self.assertNotIn('result',chat);self.assertNotIn('result',outer)
        self.assertEqual(chat['records'][-1],rows(self.out/'model_calls.jsonl')[-1])
        result=provenance(self.week)
        self.assertEqual(result['terminal_compile_chat_caches'],1)
        self.assertEqual(result['terminal_compile_outer_caches'],1)

    def test_replay_uses_original_failure_without_a_new_answer(self):
        self.fail_once()
        saved={p:p.read_bytes() for name in ['compiler-cache','natural-chat-cache'] for p in (self.week/name).glob('*.json')}
        original_calls=(self.out/'model_calls.jsonl').read_bytes()
        out=self.week/'attempt-2';out.mkdir()
        model=NaturalModels(out,MODEL,None,'jev',judge_cache=self.week/'judge-cache',require_compiler_cache=True,require_judge_cache=True)
        compiler=SourceCompiler(model,self.week/'compiler-cache');compiler.draft=deepcopy(self.compiler.draft)
        with patch.object(Models,'chat',side_effect=AssertionError('No model request permitted')):
            self.assertEqual(compiler.compile('cancellation','Do not pay this invoice.','Monday',2,['email']),[])
        self.assertEqual(compiler.draft.dump(),self.before)
        self.assertEqual(model.calls,0);self.assertEqual(compiler.misses,0)
        self.assertEqual(rows(out/'compiler_failures.jsonl'),rows(self.out/'compiler_failures.jsonl'))
        self.assertTrue(all(p.read_bytes()==value for p,value in saved.items()))
        self.assertEqual((self.out/'model_calls.jsonl').read_bytes(),original_calls)

    def test_natural_failure_cache_cannot_call_the_model_again(self):
        self.fail_once()
        cached=json.loads(next((self.week/'natural-chat-cache').glob('*.json')).read_text());r=cached['request']
        out=self.week/'attempt-2';out.mkdir()
        model=NaturalModels(out,MODEL,None,'jev',judge_cache=self.week/'judge-cache',require_compiler_cache=True)
        with patch.object(Models,'chat',side_effect=AssertionError('No retry permitted')):
            with self.assertRaises(TruncatedCompilation):model.chat(r['stage'],r['messages'],r['schema'],r['max_tokens'])
        self.assertEqual(model.calls,0)

    def test_empty_successful_output_is_not_relabelled_failure(self):
        def empty(model,stage,messages,schema,max_tokens):
            value=dict(proposals=[])
            model.record(stage,payload_for(stage,messages,schema,max_tokens),dict(model=MODEL,
                choices=[dict(finish_reason='stop',message=dict(content=json.dumps(value)))],usage={}),0.,'llm')
            return value
        with patch.object(Models,'chat',empty):
            self.assertEqual(self.compiler.compile('none','Nothing changed.','Monday',2,['email']),[])
        self.assertFalse((self.out/'compiler_failures.jsonl').exists())
        cached=json.loads(next((self.week/'compiler-cache').glob('*.json')).read_text())
        self.assertEqual(cached['result'],dict(proposals=[]))

    def test_other_runtime_error_is_not_treated_as_a_truncated_batch(self):
        with patch.object(Models,'chat',side_effect=RuntimeError('Different semantic failure')):
            with self.assertRaisesRegex(RuntimeError,'Different semantic failure'):
                self.compiler.compile('bad','Cancel it.','Monday',2,['email'])
        self.assertFalse((self.out/'compiler_failures.jsonl').exists())

    def test_fake_successful_result_in_failure_cache_is_rejected(self):
        self.fail_once()
        entry=json.loads(next((self.week/'natural-chat-cache').glob('*.json')).read_text())
        entry['result']=dict(proposals=[])
        with self.assertRaises(AssertionError):validate_failure_cache(entry)

    def test_failure_identity_stage_and_finish_must_match(self):
        self.fail_once()
        entry=json.loads(next((self.week/'natural-chat-cache').glob('*.json')).read_text())
        for field,value in [('stage','compile_condition_contracts'),('model','different-model')]:
            with self.subTest(field=field):
                changed=deepcopy(entry);changed['request'][field]=value
                with self.assertRaises(AssertionError):validate_failure_cache(changed)
        changed=deepcopy(entry);changed['records'][-1]['response']['choices'][0]['finish_reason']='stop'
        with self.assertRaises(AssertionError):validate_failure_cache(changed)

    def test_same_failure_rule_has_no_backbone_name_case(self):
        self.fail_once()
        entry=json.loads(next((self.week/'natural-chat-cache').glob('*.json')).read_text())
        for model in ['Qwen3.5-4B','Qwen3.5-9B','gemma-4-26B-A4B-it','another-backbone']:
            with self.subTest(model=model):
                changed=deepcopy(entry);changed['request']['model']=model
                changed['records'][-1]['request']['model']=model
                changed['records'][-1]['response']['model']=model
                rebuilt=failure_cache_entry(changed['request'],changed['records'])
                self.assertEqual(validate_failure_cache(rebuilt)['terminal_failure']['finish_reason'],'length')

class DisconnectHandlingTests(unittest.TestCase):
    def exercise(self,failures):
        with tempfile.TemporaryDirectory() as temporary:
            out=Path(temporary);model=Models(out,MODEL,None,'jev',judge_client=object());requests=[]
            response=dict(provider='TypeSafe',model='typesafe/jev-1.13-20260917',answers=dict(check=dict(noul=.8)),usage={})
            def opener(request,**kwargs):
                requests.append(request.data)
                if failures=='semantic':raise RuntimeError('semantic marker')
                if len(requests)<=failures:raise http.client.RemoteDisconnected('fixture')
                return io.BytesIO(json.dumps(response).encode())
            answer=None;error=None
            with patch.dict(os.environ,{'OPENROUTER_API_KEY':'fixture-not-a-real-key'}),patch.object(urllib.request,'urlopen',map_disconnect(opener)),patch('time.sleep',return_value=None):
                try:answer=model.evaluate('fixture',dict(observation='Public fixture.'),dict(check=dict(type='noul',instructions='Check.')))
                except RuntimeError as exc:error=str(exc)
            self.assertTrue(all(value==requests[0] for value in requests))
            return len(requests),model.calls,answer,error

    def test_one_disconnect_reuses_identical_wire_and_records_one_response(self):
        count,calls,answer,error=self.exercise(1)
        self.assertEqual((count,calls,error),(2,1,None));self.assertEqual(answer,dict(check=dict(noul=.8)))

    def test_disconnects_respect_existing_three_attempt_bound(self):
        count,calls,answer,error=self.exercise(99)
        self.assertEqual((count,calls,answer),(3,0,None));self.assertIn('three identical attempts',error)

    def test_semantic_failure_is_never_mapped_or_retried(self):
        self.assertEqual(self.exercise('semantic'),(1,0,None,'semantic marker'))

class TruncatedStagesTests(unittest.TestCase):
    def fixture(self,out):
        from test_ren_natural import FixtureModels
        model=FixtureModels(out)
        def chat(stage,messages,schema,max_tokens):
            request=dict(stage=stage,model=MODEL,messages=messages,schema=schema,max_tokens=max_tokens,
                         temperature=0,route='configured-direct')
            record=dict(stage=stage,kind='llm',request=payload_for(stage,messages,schema,max_tokens),
                        response=dict(model=MODEL,choices=[dict(finish_reason='length',message=dict(content='unusable partial batch'))]))
            entry=failure_cache_entry(request,[record])
            from compiler_failure import digest
            raise TruncatedCompilation(digest(request),entry)
        model.chat=chat
        return model

    def test_truncated_decomposition_keeps_exact_accepted_trigger_and_no_truth(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);model=self.fixture(out)
            policy=NaturalRENPolicy(model,None,out,routing='compiled',query_policy='round_robin',query_budget=3)
            policy.day='Monday';policy.step=1;policy.vignette='Waiting.'
            policy.store.observe('initial',intent()['source_quote'],'Monday',1)
            policy.store.apply(intent(),'initial');before=deepcopy(policy.store.dump())
            policy.prepare_active(policy.store.active('Monday'),['email','clock'])
            task=policy.needs.tasks['invoice@Monday']
            self.assertEqual(policy.store.dump(),before)
            self.assertEqual([c.description for c in task.conditions],['Approval arrives.'])
            self.assertFalse(policy.needs.ready(task))
            self.assertFalse(rows(out/'condition_contracts.jsonl')[0]['accepted'])
            self.assertEqual(rows(out/'compiler_failures.jsonl')[0]['parsed_conditions'],0)

    def test_truncated_public_frames_add_no_facts_and_keep_observed_text(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);model=self.fixture(out);text='The account balance is low.'
            frames=normalize(model,'bank',text)
            self.assertEqual(frames['facts'],[]);self.assertEqual(frames['rejected'],[])
            self.assertEqual(frames['failure']['max_tokens'],1800)
            self.assertEqual(frames['failure']['parsed_facts'],0)
            self.assertEqual(frames['failure']['operation'],'reject_truncated_public_frames_preserve_original_observation')

class CacheOnlyRouteAuditTests(unittest.TestCase):
    def test_cache_only_replay_has_no_fabricated_route_or_response(self):
        with tempfile.TemporaryDirectory() as temp:
            self.assertTrue(check_routes(Path(temp),dict(api_calls=0))['cache_only_no_new_model_calls'])

    def test_physical_response_cannot_bypass_route_audit(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);(out/'model_calls.jsonl').write_text('{}\n')
            with self.assertRaises(AssertionError):check_routes(out,dict(api_calls=1))

    def test_manifest_cannot_hide_physical_responses_as_cache_only(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);(out/'model_calls.jsonl').write_text('{}\n')
            with self.assertRaises(AssertionError):check_routes(out,dict(api_calls=0))

if __name__=='__main__':unittest.main()
