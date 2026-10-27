import unittest
from ren_confirmation_policy import CappedRoundRobin,ConfirmationPolicy
from run_recommendation import ObservationCandidate
from test_ren_grounded import GroundedTests

class BudgetTests(unittest.TestCase):
    def test_episode_cap_survives_day_change_without_modifying_rotation(self):
        scheduler=CappedRoundRobin(2,3)
        choices=[ObservationCandidate('clock',('a',)),ObservationCandidate('email',('b',))]
        self.assertEqual(scheduler.rank(choices,{},1,'Monday',True)['selected_channel'],'clock')
        self.assertEqual(scheduler.rank(choices,{'clock':'now'},1,'Monday',True)['selected_channel'],'email')
        self.assertIsNone(scheduler.rank(choices,{'clock':'now','email':'x'},1,'Monday',True)['selected_channel'])
        self.assertEqual(scheduler.spent,2)
        self.assertEqual(scheduler.rank(choices,{},2,'Tuesday',True)['selected_channel'],'clock')
        r=scheduler.rank(choices,{'clock':'now'},2,'Tuesday',True)
        self.assertIsNone(r['selected_channel']);self.assertTrue(r['episode_exhausted']);self.assertEqual(scheduler.spent,3)
    def test_uncapped_preserves_original_choices(self):
        scheduler=CappedRoundRobin(3)
        choices=[ObservationCandidate('a',('a',)),ObservationCandidate('b',('b',))]
        self.assertEqual(scheduler.rank(choices,{},1,'Monday')['selected_channel'],'a')
        self.assertEqual(scheduler.rank(choices,{'a':'x'},1,'Monday')['selected_channel'],'b')
    def test_invalid_caps_rejected(self):
        for cap in (0,-1,True,1.5):
            with self.assertRaises(ValueError):CappedRoundRobin(2,cap)

class SemanticTests(GroundedTests):
    def test_semantic_candidates_still_require_verified_execution_conditions(self):
        original=self.p;models=self.models
        p=ConfirmationPolicy(models,None,self.out,candidate_mode='semantic',routing='compiled',query_policy='round_robin',query_budget=2)
        p.store=original.store;p.day='Monday';p.step=1;p.vignette='Waiting in the office.'
        original_eval=models.evaluate
        def evaluate(stage,state,questions):
            if stage=='native_semantic_relevance_v1':return {k:dict(noul=1.) for k in questions}
            return original_eval(stage,state,questions)
        models.evaluate=evaluate
        active=p.prepare_active(p.store.active(p.day),['bank','email','clock'])
        self.assertIsNotNone(p.next_query(active,{},['bank','email','clock']))
        self.assertFalse(p.eligible_for_decision(active[0],{}))

if __name__=='__main__':unittest.main()
