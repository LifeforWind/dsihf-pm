import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from test_ren_natural import NaturalAdapterTests,FixtureModels
from run_ren_grounded import GroundedRENPolicy,GroundedModels,evidence_value,evidence_questions
from run_ren_natural import NaturalModels

class GroundedFixture(FixtureModels):
    def evaluate(self,stage,state,questions):
        if stage!='interpret_grounded_evidence':return super().evaluate(stage,state,questions)
        self.inputs.append((stage,state))
        answers={}
        for key in questions:
            row=state['evidence_items'][int(key.split('_')[-1])]
            text=row['observed_text']
            truth=text in ('approved','funded') if key.startswith('positive') else text=='explicitly absent'
            answers[key]=dict(noul=float(truth))
        return answers

class GroundedTests(NaturalAdapterTests):
    def setUp(self):
        super().setUp()
        store=self.p.store
        self.models=GroundedFixture(self.out)
        self.p=GroundedRENPolicy(self.models,None,self.out,routing='compiled',query_policy='round_robin',query_budget=3)
        self.p.store=store;self.p.step=1;self.p.day='Monday';self.p.vignette='Waiting in the office.'

    def test_two_independent_claims_preserve_unknown_and_conflict(self):
        for positive,negative,expected in [(0,0,None),(1,1,None),(1,0,True),(0,1,False)]:
            self.assertIs(evidence_value({'positive_0':dict(noul=positive),'negative_0':dict(noul=negative)},0),expected)

    def test_filtered_control_excludes_attempt_without_completing_it(self):
        self.p.filtered_control=True
        active=self.active();self.p.interpret(active,{'email':'approved','bank':'funded'})
        self.p.needs.attempted('pay@Monday','attempt')
        self.p.store.attempted('pay','Monday',1,'task_1')
        self.assertIsNone(self.p.next_query(self.active(),{},['bank','email','clock']))
        self.assertEqual(self.p.needs.status['pay@Monday'],'inflight')

    def test_shared_decision_must_use_ledger(self):
        active=self.active()
        self.assertFalse(self.models.ledger_ready('pay')[0])
        self.p.interpret(active,{'email':'approved','bank':'funded'})
        self.assertTrue(self.models.ledger_ready('pay')[0])

    def test_no_unobserved_text_sent_to_grounder(self):
        self.p.interpret(self.active(),{})
        self.assertEqual(self.models.inputs[-1][1]['evidence_items'],[])

class ModelTests(unittest.TestCase):
    def test_model_mismatch_retries_identical_request_only(self):
        with tempfile.TemporaryDirectory() as folder:
            m=GroundedModels(Path(folder),'gpt-4o-mini-2024-07-18',None,'jev',judge_client=object(),judge_cache=Path(folder)/'cache')
            with patch.object(NaturalModels,'chat',side_effect=[RuntimeError('Unexpected compiler/judge model: gpt-4.1-mini'),{'ok':True}]) as call:
                self.assertEqual(m.chat('compile',[],{},3),{'ok':True})
                self.assertEqual(call.call_args_list[0],call.call_args_list[1])
            with patch.object(NaturalModels,'chat',side_effect=RuntimeError('Unexpected compiler/judge model: x')) as call:
                with self.assertRaises(RuntimeError):m.chat('compile',[],{},3)
                self.assertEqual(call.call_count,3)

if __name__=='__main__':unittest.main()
