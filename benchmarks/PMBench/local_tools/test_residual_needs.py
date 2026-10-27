import unittest
from copy import deepcopy
from residual_policy import Condition, TaskSpec, EvidenceNeeds, ResidualPolicy


def task(key='pay', revision=1, start=0, end=20):
    return TaskSpec(key, revision, 'Pay invoice', 'Pay when funded and approved.', (
        Condition('funds', 'balance', 'Funding available', 'ge', 100, ('bank',), 2),
        Condition('approval', 'approved', 'Invoice approved', 'eq', True, ('email',), 5),
    ), start, end, ('receipt',))


def fact(key, value, at=0, until=20):
    return dict(key=key, value=value, as_of=at, valid_until=until)


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.s = EvidenceNeeds(); self.s.advance(0)
        self.s.install(task(), 'instruction', task().source_quote)

    def funded(self):
        self.s.observe('bank0', 'bank', [fact('balance', 200)])

    def ready(self):
        self.funded(); self.s.observe('email0', 'email', [fact('approved', True)])

    def test_partial_satisfaction_switches_source(self):
        self.funded()
        self.assertEqual(set(self.s.supports()), {'email'})
        self.assertEqual(set(self.s.supports('task_channels')), {'bank', 'email'})
        self.assertFalse(self.s.ready(task()))

    def test_false_waits_then_reopens(self):
        self.s.observe('b', 'bank', [fact('balance', 0)])
        self.assertEqual(self.s.supports(), {})
        self.assertEqual(self.s.status['pay'], 'pending')
        self.s.advance(3)
        self.assertEqual(set(self.s.supports()), {'bank', 'email'})

    def test_unknown_does_not_refute(self):
        self.s.observe('b', 'bank', [fact('balance', None)])
        self.assertEqual(set(self.s.supports()), {'bank', 'email'})

    def test_empty_reply_does_not_refute(self):
        self.s.observe('b', 'bank')
        self.assertEqual(set(self.s.supports()), {'bank', 'email'})

    def test_attempt_only_requests_receipt(self):
        self.ready(); self.s.attempted('pay', 'a')
        self.assertEqual(self.s.status['pay'], 'inflight')
        self.assertEqual(set(self.s.supports()), {'receipt'})

    def receipt(self, outcome='success', **kw):
        return dict(task_id='pay', revision=1, attempt_id='a', outcome=outcome) | kw

    def test_success_closes_instance(self):
        self.ready(); self.s.attempted('pay', 'a')
        self.s.observe('r', 'receipt', receipts=[self.receipt()])
        self.assertEqual(self.s.status['pay'], 'completed')
        self.assertEqual(self.s.supports(), {})

    def test_failure_allows_retry_with_fresh_evidence(self):
        self.ready(); self.s.attempted('pay', 'a')
        self.s.observe('r', 'receipt', receipts=[self.receipt('failed')])
        self.assertTrue(self.s.ready(task()))

    def test_failure_after_expiry_requires_recheck(self):
        self.ready(); self.s.attempted('pay', 'a'); self.s.advance(8)
        self.s.observe('r', 'receipt', receipts=[self.receipt('failed')])
        self.assertFalse(self.s.ready(task()))
        self.assertEqual(set(self.s.supports()), {'bank', 'email'})

    def test_wrong_attempt_revision_channel_cannot_complete(self):
        self.ready(); self.s.attempted('pay', 'a')
        for n, (channel, receipt) in enumerate([('bank', self.receipt()),
                ('receipt', self.receipt(attempt_id='wrong')), ('receipt', self.receipt(revision=2))]):
            self.s.observe(f'r{n}', channel, receipts=[receipt])
        self.assertEqual(self.s.status['pay'], 'inflight')

    def test_pending_receipt_keeps_waiting(self):
        self.ready(); self.s.attempted('pay', 'a')
        self.s.observe('r', 'receipt', receipts=[self.receipt('pending')])
        self.assertEqual(set(self.s.supports()), {'receipt'})

    def test_cancel_not_reactivated_by_receipt(self):
        self.ready(); self.s.attempted('pay', 'a')
        self.s.cancel('pay', 1, 'cancel', 'Cancel pay.', 'Cancel pay.')
        self.s.observe('r', 'receipt', receipts=[self.receipt()])
        self.assertEqual(self.s.status['pay'], 'canceled')
        self.assertEqual(self.s.supports(), {})

    def test_shared_channel_survives_one_cancellation(self):
        self.s.install(task('other'), 'other', task().source_quote)
        self.s.cancel('pay', 1, 'cancel', 'Cancel pay.', 'Cancel pay.')
        self.assertEqual({r['task_id'] for r in self.s.supports()['bank']}, {'other'})

    def test_revision_invalidates_attempt_and_evidence(self):
        self.ready(); self.s.attempted('pay', 'a')
        self.s.install(task(revision=2), 'update', task().source_quote)
        self.s.observe('r', 'receipt', receipts=[self.receipt()])
        self.assertEqual(self.s.status['pay'], 'pending')
        self.assertEqual(set(self.s.supports()), {'bank', 'email'})

    def test_future_task_persists_and_activates(self):
        self.s.install(task('tomorrow', start=6), 'future', task().source_quote)
        self.assertNotIn('tomorrow', [t.id for t in self.s.active()])
        self.s.advance(6)
        self.assertIn('tomorrow', [t.id for t in self.s.active()])

    def test_expiration_is_not_completion(self):
        self.s.advance(21)
        self.assertEqual(self.s.supports(), {})
        self.assertEqual(self.s.status['pay'], 'pending')

    def test_conflict_is_unknown(self):
        self.funded(); self.s.observe('conflict', 'bank', [fact('balance', 0)])
        self.assertEqual(self.s.states(task())['funds']['value'], 'unknown')

    def test_wrong_source_cannot_satisfy(self):
        self.s.observe('wrong', 'email', [fact('balance', 200)])
        self.assertEqual(self.s.states(task())['funds']['value'], 'unknown')

    def test_delivery_does_not_refresh_old_fact(self):
        self.s.advance(5); self.funded()
        self.assertEqual(self.s.states(task())['funds']['value'], 'unknown')

    def test_newer_expired_fact_does_not_resurrect_older_value(self):
        self.funded(); self.s.advance(1)
        self.s.observe('newer', 'bank', [fact('balance', 0, at=1, until=1)])
        self.s.advance(2)
        self.assertEqual(self.s.states(task())['funds']['value'], 'unknown')

    def test_future_timestamp_rejected_atomically(self):
        old = self.s.dump()
        with self.assertRaises(ValueError): self.s.observe('future', 'bank', [fact('balance', 200, at=1)])
        self.assertEqual(old, self.s.dump())

    def test_monotonic_and_immutable_provenance(self):
        self.funded(); self.funded()
        with self.assertRaises(ValueError): self.s.observe('bank0', 'bank', [fact('balance', 0)])
        with self.assertRaises(ValueError): self.s.advance(-1)

    def test_attempt_without_evidence_rejected(self):
        with self.assertRaises(ValueError): self.s.attempted('pay', 'a')


class AdapterTests(unittest.TestCase):
    def test_round_robin_shared_support_and_budget(self):
        from dataclasses import asdict
        p = ResidualPolicy('remaining', budget=1)
        p.begin(dict(time=0, day='Monday', context='Invoice payment', catalog={'bank': 'Balance', 'email': 'Approval', 'receipt': 'Result'},
            updates=[dict(op='add', task=asdict(task()), text=task().source_quote)]))
        self.assertEqual(p.next_query(), 'bank')
        p.receive('bank', dict(facts=[fact('balance', 200)], receipts=[]))
        self.assertIsNone(p.next_query())
        self.assertEqual(p.act(), [])
        p.begin(dict(time=1, day='Monday', context='', catalog=p.public['catalog'], updates=[]))
        self.assertEqual(p.next_query(), 'email')

    def test_hidden_fields_rejected(self):
        p = ResidualPolicy('remaining')
        with self.assertRaises(ValueError): p.begin(dict(time=0, groundtruth='private'))

    def test_development_episodes_all_arms_and_visibility(self):
        from residual_scenarios import make_scenario, Environment, FAMILIES
        class Models:
            def evaluate(self, stage, state, questions):
                self.state = deepcopy(state)
                assert 'signals' not in state and 'execution' not in state
                return {k: {'noul': 1.} for k in questions}
        for family in FAMILIES:
            for arm in ('semantic','task_channels','remaining'):
                env=Environment(make_scenario(family,3900));p=ResidualPolicy(arm,Models())
                for tick in range(12):
                    p.begin(env.begin(tick))
                    while (channel:=p.next_query()) is not None: p.receive(channel,env.query(channel))
                    self.assertLessEqual(len(p.observed),3)
                    env.execute(p.act())
                score=env.score(p.store.status)
                self.assertTrue(score['correct_closures'])
                self.assertEqual(score['invalid_attempts'],0)

    def test_private_metadata_cannot_affect_public_policy(self):
        from residual_scenarios import make_scenario, Environment
        scene=make_scenario('retry_receipt',3900)
        altered=deepcopy(scene);altered['groundtruth']={'leak':'WRONG'};altered['metadata']={'future':'WRONG'}
        envs=[Environment(s) for s in (scene,altered)]
        policies=[ResidualPolicy('remaining') for _ in range(2)]
        for tick in range(12):
            public=[e.begin(tick) for e in envs];self.assertEqual(*public)
            for p,v in zip(policies,public):p.begin(v)
            while True:
                choices=[p.next_query() for p in policies];self.assertEqual(*choices)
                if choices[0] is None:break
                replies=[e.query(c) for e,c in zip(envs,choices)];self.assertEqual(*replies)
                for p,c,r in zip(policies,choices,replies):p.receive(c,r)
            actions=[p.act() for p in policies];self.assertEqual(*actions)
            for e,a in zip(envs,actions):e.execute(a)



if __name__ == '__main__': unittest.main()
