from openmodel_config import MODEL as ACTIVE_MODEL, BASE as ACTIVE_BASE
import json
from contextlib import redirect_stdout
from copy import deepcopy
import io
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from run_author_pis import IdentityTape,NativeClient,normalize_action,author_store,StudyStop,prepare_environment,RelayTransport,canonical_memory_rows,pm,ROOT


class AuthorPISBridgeTests(unittest.TestCase):
    def test_invalid_choose_matches_native_feedback_and_execution(self):
        scene=deepcopy(json.loads((ROOT/'data/synthetic_week_v9.json').read_text()))
        scene['days']=scene['days'][:1]
        scene['days'][0]['steps']=scene['days'][0]['steps'][:1]
        for invalid,feedback in [
            (dict(action='choose',choice='D',task_ids=['task_4'],channel='NONE'),'Invalid choice.'),
            (dict(action='choose',choice='A',task_ids=[],channel='clock'),'Invalid channel.'),
        ]:
            with self.subTest(action=invalid):
                results=[]
                for bridge in (False,True):
                    attempts=[]
                    def request(c,m,messages,ids,channels,b,**kwargs):
                        attempts.append(deepcopy(messages))
                        action=deepcopy(invalid) if len(attempts)==1 else dict(action='choose',choice='A',task_ids=[],channel='NONE')
                        original=deepcopy(action)
                        result=normalize_action(action,ids,channels) if bridge else action
                        self.assertEqual(action,original)
                        return result
                    with tempfile.TemporaryDirectory() as folder,redirect_stdout(io.StringIO()), \
                         patch.object(pm,'build_llm_client',return_value=None), \
                         patch.object(pm,'request_model_action',side_effect=request), \
                         patch.object(pm,'resolve_state_query_items') as provider:
                        entries=pm.run_llm(scene,str(Path(folder)/'actions.jsonl'),'fixture',None,5,'sglang',None,None)
                    self.assertEqual(len(attempts),2)
                    self.assertEqual(len(entries),1)
                    self.assertEqual(attempts[1][-2],dict(role='assistant',content=json.dumps(invalid)))
                    self.assertIn(feedback,attempts[1][-1]['content'])
                    self.assertEqual(entries[0]['state_queries'],{})
                    provider.assert_not_called()
                    results.append((entries,attempts))
                self.assertEqual(results[0],results[1])

    def test_unknown_channel_gets_native_feedback_without_provider_query(self):
        scene=json.loads((ROOT/'data/synthetic_week_v9.json').read_text())
        scene=deepcopy(scene);scene['days']=scene['days'][:1]
        scene['days'][0]['steps']=scene['days'][0]['steps'][:1]
        unknown=dict(action='query_state',choice='NONE',task_ids=[],channel='transit_times')
        self.assertEqual(normalize_action(unknown,[],['clock']),unknown)
        attempts=[]
        def request(c,m,messages,ids,channels,b,**kwargs):
            attempts.append(deepcopy(messages))
            action=unknown if len(attempts)==1 else dict(action='choose',choice='A',task_ids=[],channel='NONE')
            return normalize_action(action,ids,channels)
        with tempfile.TemporaryDirectory() as folder,redirect_stdout(io.StringIO()), \
             patch.object(pm,'build_llm_client',return_value=None), \
             patch.object(pm,'request_model_action',side_effect=request), \
             patch.object(pm,'resolve_state_query_items') as provider:
            entries=pm.run_llm(scene,str(Path(folder)/'actions.jsonl'),'fixture',None,5,'sglang',None,None)
        self.assertEqual(len(entries),1)
        self.assertEqual(len(attempts),2)
        self.assertEqual(attempts[1][-2],dict(role='assistant',content=json.dumps(unknown)))
        self.assertIn('Invalid channel. Use one of the available state channels',attempts[1][-1]['content'])
        self.assertEqual(entries[0]['state_queries'],{})
        provider.assert_not_called()

    def test_native_query_field_rejections_are_not_adapter_failures(self):
        scene=json.loads((ROOT/'data/synthetic_week_v9.json').read_text())
        scene=deepcopy(scene);scene['days']=scene['days'][:1]
        scene['days'][0]['steps']=scene['days'][0]['steps'][:1]
        cases=[
            (dict(action='query_state',choice='NONE',task_ids=['private-id'],channel='calendar'),'Invalid task_ids.'),
            (dict(action='query_state',choice='A',task_ids=[],channel='calendar'),'Invalid choice.'),
            (dict(action='check_time',choice='NONE',task_ids=['private-id'],channel='clock'),'Invalid task_ids.'),
        ]
        for invalid,feedback in cases:
            with self.subTest(action=invalid):
                attempts=[]
                def request(c,m,messages,ids,channels,b,**kwargs):
                    attempts.append(deepcopy(messages))
                    action=invalid if len(attempts)==1 else dict(action='choose',choice='A',task_ids=[],channel='NONE')
                    return normalize_action(action,ids,channels)
                with tempfile.TemporaryDirectory() as folder,redirect_stdout(io.StringIO()), \
                     patch.object(pm,'build_llm_client',return_value=None), \
                     patch.object(pm,'request_model_action',side_effect=request), \
                     patch.object(pm,'resolve_state_query_items') as provider:
                    entries=pm.run_llm(scene,str(Path(folder)/'actions.jsonl'),'fixture',None,5,'sglang',None,None)
                self.assertEqual(len(entries),1)
                self.assertEqual(attempts[1][-2],dict(role='assistant',content=json.dumps(invalid)))
                self.assertIn(feedback,attempts[1][-1]['content'])
                self.assertEqual(entries[0]['task_ids'],[])
                self.assertEqual(entries[0]['state_queries'],{})
                provider.assert_not_called()

    def test_only_diagnostic_due_ids_are_unordered_without_mutating_logs(self):
        original=[dict(event='recall',recall_match_due_ids=['b','a'],other=['b','a'])]
        reordered=[dict(event='recall',recall_match_due_ids=['a','b'],other=['b','a'])]
        self.assertEqual(canonical_memory_rows(original),canonical_memory_rows(reordered))
        self.assertEqual(original[0]['recall_match_due_ids'],['b','a'])
        reordered[0]['other']=['a','b']
        self.assertNotEqual(canonical_memory_rows(original),canonical_memory_rows(reordered))

    def test_due_id_membership_and_row_order_still_must_match(self):
        original=[dict(recall_match_due_ids=['a','b']),dict(recall_match_due_ids=['c'])]
        changed=[dict(recall_match_due_ids=['a','d']),dict(recall_match_due_ids=['c'])]
        self.assertNotEqual(canonical_memory_rows(original),canonical_memory_rows(changed))
        self.assertNotEqual(canonical_memory_rows(original),canonical_memory_rows(original[::-1]))
        for invalid in [['a','a'],None,['a',1]]:
            with self.assertRaises(AssertionError):
                canonical_memory_rows([dict(recall_match_due_ids=invalid)])

    def test_offline_flag_precedes_transport_configuration(self):
        observed=[]
        with patch.dict(os.environ,{},clear=True):
            with patch('run_author_pis.configure',side_effect=lambda:observed.append(os.environ.get('MATCHED_OFFLINE'))):
                prepare_environment(offline=True)
        self.assertEqual(observed,['1'])

    def test_shared_stop_blocks_author_physical_call(self):
        with tempfile.TemporaryDirectory() as f,patch.dict(os.environ,{},clear=True):
            root=Path(f);stop=root/'STOP.json';stop.write_text('{}')
            os.environ['PAPER_BASELINES_STOP']=str(stop)
            with patch('run_author_pis.configure'):prepare_environment()
            self.assertEqual(os.environ['RELAY_STOP_FILE'],str(stop))
            calls=[]
            sdk=SimpleNamespace(base_url=ACTIVE_BASE,chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kw:calls.append(kw))))
            relay=RelayTransport(sdk,root/'out',root/'cache')
            client=NativeClient(relay,SimpleNamespace(remaining=3,request=lambda x:x))
            with self.assertRaises(StudyStop):
                client.create(model=ACTIVE_MODEL,messages=[])
            self.assertEqual(calls,[])

    def test_random_identity_tape_exact_recovery_and_offline(self):
        with tempfile.TemporaryDirectory() as f:
            p=Path(f)/'ids.json';first=IdentityTape(p)
            a,b=first.uuid4(),first.uuid4()
            replay=IdentityTape(p,True)
            self.assertEqual([replay.uuid4(),replay.uuid4()],[a,b])
            with self.assertRaises(StudyStop):replay.uuid4()
            resumed=IdentityTape(p);self.assertEqual(resumed.uuid4(),a)

    def test_public_budget_notice_only_on_decisions(self):
        captured=[]
        transport=SimpleNamespace(create=lambda **kw:captured.append(kw))
        def request(kw):
            return dict(kw,messages=kw['messages']+[dict(role='user',content='budget')])
        budget=SimpleNamespace(remaining=1,request=request)
        client=NativeClient(transport,budget)
        base=dict(model=ACTIVE_MODEL,messages=[dict(role='user',content='observed')],extra_body=dict(chat_template_kwargs={}))
        with patch('run_author_pis.current_role',return_value='update_judge'):client.create(**base)
        self.assertEqual(len(captured[-1]['messages']),1)
        from qwen_wire import wire
        # The unchanged author removes these extras for non-Qwen models.
        # Both backbones must reach the actual local endpoint with thinking off.
        from openmodel_config import PROFILE
        if PROFILE == 'relay':
            self.assertNotIn('extra_body', wire(captured[-1]))
        else:
            self.assertEqual(wire(captured[-1])['extra_body']['chat_template_kwargs'], {'enable_thinking': False})
        with patch('run_author_pis.current_role',return_value='choose'):client.create(**base)
        self.assertEqual(len(captured[-1]['messages']),2)

    def test_exhausted_cap_constrains_action_without_inventing_tasks(self):
        captured=[]
        client=NativeClient(SimpleNamespace(create=lambda **kw:captured.append(kw)),
            SimpleNamespace(remaining=0,request=lambda x:x))
        client.ids=['task_1'];client.channels=['clock']
        with patch('run_author_pis.current_role',return_value='choose'):
            client.create(model=ACTIVE_MODEL,messages=[])
        props=captured[0]['response_format']['json_schema']['schema']['properties']
        self.assertEqual(props['action']['enum'],['choose'])
        self.assertEqual(props['channel']['enum'],['NONE'])
        self.assertEqual(props['task_ids']['items']['enum'],['task_1'])

    def test_native_default_flags_preserved(self):
        backend=author_store.IntentionStoreBackend(llm_client=object(),llm_model=ACTIVE_MODEL,
            use_llm_update=True,use_llm_extract=True,use_llm_match=True,max_inject_chars=8000,
            llm_response_format='json_object',retrieval_mode='due')
        self.assertTrue(backend.use_llm_update);self.assertTrue(backend.use_llm_rewrite)
        self.assertTrue(backend.use_llm_match);self.assertFalse(backend.use_llm_done)
        self.assertFalse(backend.gate_time_due);self.assertFalse(backend.force_query_channels)
        self.assertEqual(backend._event_sim_threshold,.62)

    def test_original_public_action_alias_normalization(self):
        a=normalize_action(dict(action='choose',choice='A',task_id='task_1'),['task_1'],['clock'])
        self.assertEqual(a['task_ids'],['task_1']);self.assertEqual(a['channel'],'NONE')
        unknown=dict(action='choose',choice='A',task_ids=['unknown-handle'],channel='NONE')
        self.assertEqual(normalize_action(unknown,['task_1'],['clock']),unknown)

    def test_overlong_choose_matches_native_execution_without_truncation_or_retry(self):
        scene=json.loads((ROOT/'data/synthetic_week_v9.json').read_text())
        scene=deepcopy(scene);scene['days']=scene['days'][:1]
        scene['days'][0]['steps']=scene['days'][0]['steps'][:1]
        results=[]
        for bridge in (False,True):
            attempts=[]
            def request(c,m,messages,ids,channels,b,**kwargs):
                self.assertGreaterEqual(len(ids),9)
                action=dict(action='choose',choice='C',task_ids=ids[:9],channel='NONE')
                original=deepcopy(action);attempts.append(original)
                result=normalize_action(action,ids,channels) if bridge else action
                self.assertEqual(action,original)
                self.assertEqual(result,original)
                return result
            with tempfile.TemporaryDirectory() as folder,redirect_stdout(io.StringIO()), \
                 patch.object(pm,'build_llm_client',return_value=None), \
                 patch.object(pm,'request_model_action',side_effect=request), \
                 patch.object(pm,'resolve_state_query_items') as provider:
                entries=pm.run_llm(scene,str(Path(folder)/'actions.jsonl'),'fixture',None,5,'sglang',None,None)
            self.assertEqual(len(attempts),1)
            self.assertEqual(len(entries[0]['task_ids']),9)
            self.assertEqual(entries[0]['state_queries'],{})
            provider.assert_not_called()
            results.append(entries)
        self.assertEqual(results[0],results[1])

    def test_unknown_choose_handle_matches_native_execution_without_retry(self):
        scene=json.loads((ROOT/'data/synthetic_week_v9.json').read_text())
        scene=deepcopy(scene);scene['days']=scene['days'][:1]
        scene['days'][0]['steps']=scene['days'][0]['steps'][:1]
        results=[]
        for bridge in (False,True):
            attempts=[]
            def request(c,m,messages,ids,channels,b,**kwargs):
                action=dict(action='choose',choice='C',task_ids=[ids[0],'unknown-handle'],channel='NONE')
                attempts.append(deepcopy(action))
                return normalize_action(action,ids,channels) if bridge else action
            with tempfile.TemporaryDirectory() as folder,redirect_stdout(io.StringIO()), \
                 patch.object(pm,'build_llm_client',return_value=None), \
                 patch.object(pm,'request_model_action',side_effect=request), \
                 patch.object(pm,'resolve_state_query_items') as provider:
                entries=pm.run_llm(scene,str(Path(folder)/'actions.jsonl'),'fixture',None,5,'sglang',None,None)
            self.assertEqual(len(attempts),1)
            self.assertEqual(len(entries[0]['task_ids']),1)
            self.assertNotIn('unknown-handle',entries[0]['task_ids'])
            self.assertEqual(entries[0]['state_queries'],{})
            provider.assert_not_called()
            results.append(entries)
        self.assertEqual(results[0],results[1])


if __name__=='__main__':unittest.main()
