import json
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from run_ren_natural import NaturalRENPolicy, SCENE, RECEIPT


class FixtureModels:
    def __init__(self,out):
        self.out=out; self.records=[]; self.inputs=[]; self.accept=True
    def chat(self,stage,messages,schema,max_tokens):
        state=json.loads(messages[-1]['content'])
        return {'contracts':[dict(id=i['id'],conditions=[
            dict(description='Funds arrived',channels=['bank']),
            dict(description='Approval arrived',channels=['email'])]) for i in state['intentions']]}
    def evaluate(self,stage,state,questions):
        self.inputs.append((stage,state))
        if stage=='verify_condition_contracts':return {k:dict(noul=1. if self.accept else 0.) for k in questions}
        assert stage=='interpret_condition_evidence',stage
        result={}
        for k in questions:
            row=state['evidence_items'][int(k.split('_')[-1])]
            value={'approved':'TRUE','funded':'TRUE','explicitly absent':'FALSE'}.get(row['observed_text'],'UNKNOWN')
            result[k]=dict(choice=value)
        return result


class NaturalAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.out=Path(self.temp.name); self.models=FixtureModels(self.out)
        self.p=NaturalRENPolicy(self.models,None,self.out,routing='compiled',query_policy='round_robin',query_budget=3)
        self.p.step=1;self.p.day='Monday';self.p.vignette='Waiting in the office.'
        self.add('pay')
    def add(self,key,op='add',trigger='Funds and approval arrived',day='Monday'):
        quote='Pay when funds and approval arrive.' if op=='add' else 'Change the payment instruction.'
        eid=f'{op}:{key}:{len(self.p.store.evidence)}'
        self.p.store.observe(eid,quote,self.p.day,self.p.step)
        spec=dict(action='Pay invoice',trigger=trigger,day=day,kind='event',at=None,
            channel='bank',comparator=None,threshold=None,parent_id=None,dependency=None,offset_minutes=None)
        self.p.store.apply(dict(op=op,id=key,source_quote=quote,spec=spec),eid)
    def active(self):return self.p.prepare_active(self.p.store.active(self.p.day),['bank','email','clock'])
    def test_actual_condition_ledger_removes_only_satisfied_source(self):
        active=self.active();self.p.interpret(active,{'email':'approved'})
        self.assertEqual(set(self.p.needs.supports()),{'bank'})
        self.assertEqual(set(self.p.needs.supports('task_channels')),{'bank','email'})
        self.assertFalse(self.p.eligible_for_decision(active[0],{}))
        self.p.interpret(active,{'email':'approved','bank':'funded'})
        self.assertTrue(self.p.eligible_for_decision(active[0],{}))
    def test_unqueried_sources_never_enter_judge(self):
        self.p.interpret(self.active(),{})
        self.assertFalse(any(state['evidence_items'] for stage,state in self.models.inputs if stage=='interpret_condition_evidence'))
        self.assertEqual(set(self.p.needs.supports()),{'bank','email'})
    def test_empty_response_is_unknown_not_negative(self):
        self.p.interpret(self.active(),{'bank':'no new items'})
        self.assertEqual(set(self.p.needs.supports()),{'bank','email'})
    def test_false_waits_and_next_step_reopens_without_certificates(self):
        self.p.interpret(self.active(),{'bank':'explicitly absent'})
        self.assertEqual(self.p.needs.supports(),{})
        self.p.step=2;self.active()
        self.assertEqual(set(self.p.needs.supports()),{'bank','email'})
    def test_positive_evidence_expires_next_step(self):
        self.p.interpret(self.active(),{'bank':'funded','email':'approved'})
        self.p.step=2;a=self.active()
        self.assertFalse(self.p.eligible_for_decision(a[0],{}))
    def test_attempt_is_inflight_and_unavailable_receipt_not_queried(self):
        a=self.active();self.p.interpret(a,{'bank':'funded','email':'approved'})
        self.p.needs.attempted('pay@Monday','attempt')
        self.assertEqual(self.p.needs.status['pay@Monday'],'inflight')
        self.assertEqual(set(self.p.needs.supports()),{RECEIPT})
        self.assertIsNone(self.p.next_query(a,{},['bank','email','clock']))
    def test_cancel_removes_only_own_shared_support(self):
        self.add('other');self.active();self.add('pay','cancel');self.active()
        self.assertEqual({r['task_id'] for r in self.p.needs.supports()['bank']},{'other@Monday'})
    def test_revision_invalidates_condition_evidence(self):
        self.p.interpret(self.active(),{'bank':'funded','email':'approved'})
        self.add('pay','revise');a=self.active()
        self.assertEqual(self.p.needs.tasks['pay@Monday'].revision,2)
        self.assertFalse(self.p.eligible_for_decision(a[0],{}))
    def test_cross_day_intention_persists_and_daily_instances_separate(self):
        self.add('daily',day='EVERYDAY');self.add('future',day='Tuesday');self.active()
        self.assertNotIn('future@Monday',self.p.needs.tasks)
        self.p.day='Tuesday';self.p.step=13;self.active()
        self.assertIn('future@Tuesday',self.p.needs.tasks)
        self.assertIn('daily@Tuesday',self.p.needs.tasks)
        self.assertNotIn('daily@Monday',[t.id for t in self.p.needs.active()])
    def test_rejected_decomposition_falls_back_to_whole_trigger(self):
        self.models.accept=False;self.active()
        t=self.p.needs.tasks['pay@Monday']
        self.assertEqual(len(t.conditions),1)
        self.assertEqual(t.conditions[0].description,'Funds and approval arrived')
    def test_empty_source_decomposition_is_rejected_not_episode_failure(self):
        self.models.chat=lambda *args,**kwargs:{'contracts':[dict(id='pay',conditions=[dict(description='Funds arrived',channels=[])])]}
        self.active()
        t=self.p.needs.tasks['pay@Monday']
        self.assertEqual(len(t.conditions),1)
        self.assertEqual(t.conditions[0].description,'Funds and approval arrived')
        self.assertEqual(t.conditions[0].channels,('bank',))
    def fixed_contracts(self, contracts):
        self.chat_calls = 0
        result = {'contracts': contracts}
        before = deepcopy(result)
        def chat(*args, **kwargs):
            self.chat_calls += 1
            return result
        self.models.chat = chat
        return result, before
    def test_exact_duplicate_is_normalized_once_without_changing_response(self):
        row = dict(id='pay', conditions=[dict(description='Funds and approval arrived', channels=['bank','email'])])
        result, before = self.fixed_contracts([row, deepcopy(row)])
        self.active()
        self.assertEqual(result, before)
        self.assertEqual(self.chat_calls, 1)
        judge = [state for stage,state in self.models.inputs if stage=='verify_condition_contracts']
        self.assertEqual(judge[0]['proposed_conditions'], {'pay': row['conditions']})
        log = json.loads((self.out/'contract_normalizations.jsonl').read_text())
        self.assertEqual(log['duplicates'], [dict(id='pay',kept_index=0,duplicate_index=1)])
        self.assertEqual(len(self.p.needs.tasks['pay@Monday'].conditions), 1)
    def test_conflicting_duplicate_fails_before_verification(self):
        self.fixed_contracts([dict(id='pay',conditions=[dict(description='Funds',channels=['bank'])]),
                              dict(id='pay',conditions=[dict(description='Approval',channels=['email'])])])
        with self.assertRaisesRegex(RuntimeError, 'Conflicting duplicate'):
            self.active()
        self.assertFalse(self.models.inputs)
        self.assertEqual(self.chat_calls, 1)
    def test_reordered_channels_are_not_silently_treated_as_identical(self):
        self.fixed_contracts([dict(id='pay',conditions=[dict(description='Funds',channels=['bank','email'])]),
                              dict(id='pay',conditions=[dict(description='Funds',channels=['email','bank'])])])
        with self.assertRaisesRegex(RuntimeError, 'Conflicting duplicate'):
            self.active()
    def test_missing_and_extra_ids_still_fail(self):
        for records in ([], [dict(id='other',conditions=[])]):
            with self.subTest(records=records):
                self.fixed_contracts(records)
                with self.assertRaisesRegex(RuntimeError, 'Missing or extra'):
                    self.active()
    def test_unique_response_does_not_emit_normalization(self):
        self.active()
        self.assertFalse((self.out/'contract_normalizations.jsonl').exists())

    def test_rotation_and_query_cap(self):
        a=self.active();self.assertEqual(self.p.next_query(a,{},['bank','email','clock']),'bank')
        self.assertEqual(self.p.next_query(a,{'bank':'no new'},['bank','email','clock']),'email')
        self.assertIsNone(self.p.next_query(a,{'bank':'no new','email':'no new'},['bank','email','clock']))


if __name__=='__main__':unittest.main()
