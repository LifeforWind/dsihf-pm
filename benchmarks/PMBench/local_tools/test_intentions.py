"""Functional checks for evidence isolation, lifecycle and causal updates."""
from copy import deepcopy
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import urllib.error
import unittest
from unittest.mock import patch

from run_intentions import Compiler, Models, IntentionStore, IntentionPolicy, deterministic_trigger, factual_vignette, preserve_explicit_scope, ROOT, pm


def proposal(key, op='add', **kwargs):
    spec = dict(action='Do ' + key, trigger='At the agreed time', day='Wednesday',
                kind='clock', at='12:00', channel='clock', comparator=None, threshold=None,
                parent_id=None, dependency=None, offset_minutes=None)
    spec.update(kwargs)
    return dict(id=key, op=op, source_quote='explicit instruction', spec=spec)


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.store = IntentionStore()
        self.store.observe('e', 'An explicit instruction in observed data.', 'Monday', 1)

    def test_update_propagates_only_to_dependents(self):
        for p in [proposal('meeting'), proposal('prep', parent_id='meeting', dependency='time_offset', offset_minutes=-1440),
                  proposal('pack', parent_id='prep', dependency='time_offset', offset_minutes=-60), proposal('unrelated')]:
            self.store.apply(p, 'e')
        self.store.apply(proposal('meeting', 'revise', day='Friday', at='15:00'), 'e')
        self.assertEqual((self.store.intents['prep'].day, self.store.intents['prep'].at), ('Thursday', '15:00'))
        self.assertEqual((self.store.intents['pack'].day, self.store.intents['pack'].at), ('Thursday', '14:00'))
        self.assertEqual(self.store.intents['unrelated'].revision, 1)
        self.store.apply(proposal('meeting', 'cancel'), 'e')
        self.assertTrue(self.store.intents['pack'].blocked_by)
        self.assertEqual(self.store.active('Thursday'), [])

    def test_cycle_rejected_atomically(self):
        self.store.apply(proposal('a'), 'e')
        self.store.apply(proposal('b', parent_id='a', dependency='time_offset', offset_minutes=-10), 'e')
        before = deepcopy(self.store.dump())
        with self.assertRaises(ValueError):
            self.store.apply(proposal('a', 'revise', parent_id='b', dependency='time_offset', offset_minutes=-10), 'e')
        self.assertEqual(before, self.store.dump())

    def test_attempt_is_not_confirmation_and_failure_allows_retry(self):
        self.store.apply(proposal('parent'), 'e')
        self.store.apply(proposal('child', parent_id='parent', dependency='requires_confirmed'), 'e')
        self.store.attempted('parent', 'Wednesday', 4, 'task_1')
        self.assertEqual(self.store.active('Wednesday'), [])
        self.store.acknowledge('parent', 'Wednesday', False, 'e')
        self.assertEqual([i.id for i in self.store.active('Wednesday')], ['parent'])
        self.store.acknowledge('parent', 'Wednesday', True, 'e')
        self.assertEqual([i.id for i in self.store.active('Wednesday')], ['child'])

    def test_daily_and_crossday(self):
        self.store.apply(proposal('daily', day='EVERYDAY'), 'e')
        self.store.apply(proposal('later'), 'e')
        self.store.attempted('daily', 'Monday', 1, 'task_1')
        self.assertEqual(self.store.active('Monday'), [])
        self.assertEqual([i.id for i in self.store.active('Tuesday')], ['daily'])
        self.assertEqual([i.id for i in self.store.active('Wednesday')], ['daily', 'later'])

    def test_grounding_and_rejected_revision(self):
        self.store.apply(proposal('a'), 'e')
        self.store.apply(proposal('a', 'cancel'), 'e', accepted=False)
        self.assertEqual(self.store.intents['a'].status, 'pending')
        p = proposal('b'); p['source_quote'] = 'unseen instruction'
        with self.assertRaises(ValueError): self.store.apply(p, 'e')
        self.assertNotIn('b', self.store.intents)

    def test_deterministic_time_and_numeric_conditions(self):
        self.store.apply(proposal('a'), 'e')
        item = self.store.intents['a']
        self.assertFalse(deterministic_trigger(item, {}))
        self.assertFalse(deterministic_trigger(item, {'clock': 'Time 11:59 | Stopwatch: 100 min'}))
        self.assertTrue(deterministic_trigger(item, {'clock': 'Time 12:00 | Stopwatch: 101 min'}))
        self.store.apply(proposal('b', kind='numeric', at=None, channel='bank_balance', comparator='lt', threshold=95), 'e')
        self.assertTrue(deterministic_trigger(self.store.intents['b'], {'bank_balance': 'Balance: $94.00'}))
        self.assertFalse(deterministic_trigger(self.store.intents['b'], {'bank_balance': 'Balance: $95.00'}))


class FakeModels:
    judge = 'offline'; calls = 0
    def __init__(self): self.requests = []
    def evaluate(self, stage, state, questions):
        self.requests.append(dict(stage=stage, state=state, questions=questions))
        return {key: ({'noul': 1} if q['type'] == 'noul' else {'choice': 'NONE'}) for key, q in questions.items()}


class FakeCompiler:
    hits = 0
    def __init__(self): self.requests = []
    def compile(self, evidence_id, text, day, step, channels):
        self.requests.append(text)
        if len(self.requests) != 1: return []
        p = proposal('daily', day='EVERYDAY', at='11:00')
        p['source_quote'] = 'Take asthma medication at 11:00 and 21:00.'
        return [p]


class IntegrationTests(unittest.TestCase):
    def test_transport_retry_keeps_the_same_model_and_records_failure(self):
        response = dict(model='typesafe/jev-1.13-20260917', provider='TypeSafe',
                        answers={'q': {'type': 'noul', 'noul': .9}}, usage={'input_tokens': 10, 'output_tokens': 5, 'cost': .0001})
        with tempfile.TemporaryDirectory() as directory, patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test-placeholder'}), \
                patch('run_intentions.urllib.request.urlopen', side_effect=[urllib.error.URLError('EOF'), StringIO(json.dumps(response))]) as call, \
                patch('run_intentions.time.sleep'):
            model = Models(Path(directory), 'gpt-4o-mini', None, 'jev')
            answer = model.evaluate('test', 'observed state', {'q': {'type': 'noul', 'instructions': 'A factual question.'}})
            self.assertEqual(answer['q']['noul'], .9)
            self.assertEqual(call.call_count, 2)
            self.assertEqual(call.call_args_list[0].args[0].data, call.call_args_list[1].args[0].data)
            self.assertEqual(model.calls, 1)
            self.assertNotIn('test-placeholder', (Path(directory)/'model_calls.jsonl').read_text())
            self.assertIn('URLError', (Path(directory)/'api_errors.jsonl').read_text())

    def test_scope_comes_from_observed_heading_and_preserves_explicit_repetition(self):
        batch = [proposal('a', day='EVERYDAY'), proposal('b', day='EVERYDAY'), proposal('c', day='EVERYDAY')]
        batch[1]['source_quote'] = 'Do this every day.'
        batch[2]['source_quote'] = 'Do this on Thursday.'
        changes = preserve_explicit_scope(batch, '=== Monday ===\n- explicit instruction', 'Monday', IntentionStore())
        self.assertEqual(len(changes), 1)
        self.assertEqual([p['spec']['day'] for p in batch], ['Monday', 'EVERYDAY', 'EVERYDAY'])

    def test_wrong_compiler_target_cannot_overwrite_unrelated_intention(self):
        class TargetModels(FakeModels):
            def evaluate(self, stage, state, questions):
                if stage == 'resolve_update_targets':
                    return {key: {'choice': 'book'} for key in questions}
                return super().evaluate(stage, state, questions)
        class TargetCompiler(FakeCompiler):
            def compile(self, evidence_id, text, day, step, channels):
                self.requests.append(text)
                if len(self.requests) == 1:
                    return [proposal('book', action='Return the book', kind='event', at=None, channel=None, day='Monday'),
                            proposal('email', action='Send an email', kind='event', at=None, channel=None, day='Monday')]
                if step == 2:
                    return [proposal('email', 'revise', action='Return the book', trigger='SMS arrives',
                                     kind='event', at=None, channel=None, day='Monday')]
                return []
        messages = [dict(role='system', content='Observed messages only.'),
                    dict(role='user', content='Regular tasks for every day:\n- explicit instruction'),
                    dict(role='user', content='=== Monday ==='),
                    dict(role='user', content='First scene.\nA) Wait\nStep action menu:\n- task_1: Return the book.\n- task_2: Send an email.')]
        with tempfile.TemporaryDirectory() as directory:
            policy = IntentionPolicy(TargetModels(), TargetCompiler(), Path(directory))
            action = policy(None, 'offline', messages, ['task_1', 'task_2'], ['clock'], 'offline')
            messages.extend([dict(role='assistant', content=json.dumps(action)),
                dict(role='user', content='An explicit instruction updates the book return to SMS.\nA) Wait\nStep action menu:\n- task_1: Return the book.\n- task_2: Send an email.')])
            policy(None, 'offline', messages, ['task_1', 'task_2'], ['clock'], 'offline')
        self.assertEqual(policy.store.intents['book'].trigger, 'SMS arrives')
        self.assertEqual(policy.store.intents['book'].revision, 2)
        self.assertEqual(policy.store.intents['email'].action, 'Send an email')
        self.assertEqual(policy.store.intents['email'].revision, 1)

    def test_compiler_allocates_ids_and_reuses_identical_cache(self):
        class CompilerModels:
            model = 'fake'
            calls = 0
            def chat(self, *args):
                self.calls += 1
                p = proposal('same', action='First')
                p2 = proposal('same', action='Second')
                p['spec']['parent_id'] = 'null'
                return {'proposals': [p, p2]}
        with tempfile.TemporaryDirectory() as directory:
            model = CompilerModels(); model.out = Path(directory)
            cache = Path(directory)/'cache'
            first = Compiler(model, cache).compile('e', 'explicit instruction', 'Monday', 1, ['clock'])
            second = Compiler(model, cache).compile('e', 'explicit instruction', 'Monday', 1, ['clock'])
            self.assertEqual(first, second)
            self.assertEqual(model.calls, 1)
            self.assertEqual(len({p['id'] for p in first}), 2)

    def test_schema_failure_never_uses_official_label_fallback(self):
        scenario = json.loads((ROOT/'data/synthetic_week_v9.json').read_text())
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()), \
                patch.object(pm, 'build_llm_client', return_value=None), \
                patch.object(pm, 'request_model_action', side_effect=RuntimeError('invalid typed decision')):
            with self.assertRaisesRegex(RuntimeError, 'invalid typed decision'):
                pm.run_llm(scenario, str(Path(directory)/'actions.jsonl'), 'offline', None, 5, 'sglang', None, None)

    def test_official_visible_only_replay(self):
        scenario = json.loads((ROOT / 'data/synthetic_week_v9.json').read_text())
        scenario['days'] = scenario['days'][:3]
        for day in scenario['days']:
            for step in day['steps']:
                step['groundtruth']['private_canary'] = 'SECRET_LABEL_CANARY'
        models, compiler = FakeModels(), FakeCompiler()
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            policy = IntentionPolicy(models, compiler, out)
            with redirect_stdout(StringIO()), patch.object(pm, 'build_llm_client', return_value=None), \
                    patch.object(pm, 'request_model_action', side_effect=policy):
                entries = pm.run_llm(scenario, str(out / 'actions.jsonl'), 'offline', None, 5, 'sglang', None, None)
            summary, _, steps = pm.score_log(scenario, entries)
        self.assertEqual(steps, 36)
        self.assertEqual(summary['check_time_calls'], 36)
        self.assertEqual((summary['set_tp'], summary['set_fp'], summary['set_fn']), (0, 0, 34))
        self.assertNotIn('SECRET_LABEL_CANARY', json.dumps(models.requests) + json.dumps(compiler.requests))
        self.assertNotIn('Step action menu', json.dumps(compiler.requests))
        self.assertNotIn('On Wednesday', compiler.requests[0])
        self.assertTrue(any('On Wednesday' in text for text in compiler.requests))

    def test_options_not_treated_as_observations(self):
        self.assertEqual(factual_vignette('A room.\nA) Read an email\nB) Go\nStep action menu:\n- task_1: Go'), 'A room.')


if __name__ == '__main__':
    unittest.main()
