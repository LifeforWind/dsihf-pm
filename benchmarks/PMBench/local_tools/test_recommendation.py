"""Observation recommendation invariants and official visible-only integration."""
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from run_recommendation import (CachedModels, ObservationCandidate, ObservationScheduler,
                                RecommendationPolicy, digest)
from run_intentions import Compiler, IntentionPolicy, ROOT, pm
from test_intentions import FakeCompiler, FakeModels, proposal


class SchedulerTests(unittest.TestCase):
    def test_budget_clock_and_group_coverage(self):
        scheduler = ObservationScheduler('ranked', 2)
        candidates = [ObservationCandidate('portal', ('p1', 'p2'), 1),
                      ObservationCandidate('mail', ('m',), 1), ObservationCandidate('clock', ('c',))]
        self.assertEqual(scheduler.rank(candidates, {}, 1, 'Monday', True)['selected_channel'], 'clock')
        self.assertEqual(scheduler.rank(candidates, {'clock'}, 1, 'Monday', True)['selected_channel'], 'portal')
        self.assertIsNone(scheduler.rank(candidates, {'clock', 'portal'}, 1, 'Monday', True)['selected_channel'])

    def test_unobserved_channel_eventually_gets_priority(self):
        scheduler = ObservationScheduler('ranked', 1)
        candidates = [ObservationCandidate('frequent', ('a',), .7), ObservationCandidate('quiet', ('b',), 0)]
        selected = []
        for step in range(1, 9):
            channel = scheduler.rank(candidates, {}, step, 'Monday')['selected_channel']
            selected.append(channel)
            scheduler.rank(candidates, {channel}, step, 'Monday')
        self.assertIn('quiet', selected)

    def test_round_robin_and_day_reset(self):
        scheduler = ObservationScheduler('round_robin', 1)
        candidates = [ObservationCandidate('a', ('a',), 0), ObservationCandidate('b', ('b',), 1)]
        self.assertEqual(scheduler.rank(candidates, {}, 1, 'Monday')['selected_channel'], 'a')
        scheduler.rank(candidates, {'a'}, 1, 'Monday')
        self.assertEqual(scheduler.rank(candidates, {}, 2, 'Monday')['selected_channel'], 'b')
        self.assertEqual(scheduler.rank(candidates, {}, 3, 'Tuesday')['selected_channel'], 'a')

    def test_invalid_configuration_and_inflated_coverage(self):
        for value in (True, 0, -1):
            with self.assertRaises(ValueError): ObservationScheduler(budget=value)
        with self.assertRaises(ValueError): ObservationScheduler('all', 1)
        with self.assertRaises(ValueError): ObservationCandidate('x', ('a', 'a'))
        for score in (float('nan'), float('inf'), -1, True):
            with self.assertRaises(ValueError): ObservationCandidate('x', ('a',), score)


class RoutingModels(FakeModels):
    def __init__(self):
        super().__init__()
        self.route = 'portal'
        self.support = .9

    def evaluate(self, stage, state, questions):
        self.requests.append(dict(stage=stage, state=deepcopy(state), questions=deepcopy(questions)))
        if stage == 'route_observation_channel':
            return {key: {'choice': self.route} for key in questions}
        if stage == 'verify_observation_channel':
            return {key: {'noul': self.support} for key in questions}
        return {key: ({'noul': .9} if q['type'] == 'noul' else {'choice': 'NONE'})
                for key, q in questions.items()}


class PolicyTests(unittest.TestCase):
    def test_routing_is_versioned_and_does_not_mutate_intentions(self):
        with tempfile.TemporaryDirectory() as directory:
            models = RoutingModels()
            policy = RecommendationPolicy(models, FakeCompiler(), Path(directory))
            policy.store.observe('e', 'explicit instruction', 'Monday', 1)
            policy.store.apply(proposal('a', day='Monday', kind='event', at=None, channel='wrong'), 'e')
            policy.step, policy.day = 1, 'Monday'
            active = policy.prepare_active(policy.store.active('Monday'), ['wrong', 'portal'])
            self.assertEqual(active[0].channel, 'portal')
            self.assertEqual(policy.store.intents['a'].channel, 'wrong')
            policy.prepare_active(policy.store.active('Monday'), ['wrong', 'portal'])
            self.assertEqual(len(models.requests), 2)
            models.route = 'NONE'
            policy.store.apply(proposal('a', 'revise', day='Monday', kind='event', at=None,
                                        channel=None, trigger='Visible text arrives'), 'e')
            active = policy.prepare_active(policy.store.active('Monday'), ['wrong', 'portal'])
            self.assertIsNone(active[0].channel)
            self.assertEqual(len(models.requests), 4)

    def test_unsupported_binding_preserves_source_suggestion(self):
        with tempfile.TemporaryDirectory() as directory:
            models = RoutingModels(); models.support = .1
            policy = RecommendationPolicy(models, FakeCompiler(), Path(directory))
            policy.store.observe('e', 'explicit instruction', 'Monday', 1)
            policy.store.apply(proposal('a', day='Monday', kind='event', at=None, channel='mail'), 'e')
            policy.step = 1
            item = policy.prepare_active(policy.store.active('Monday'), ['mail', 'portal'])[0]
            self.assertEqual(item.channel, 'mail')
            self.assertFalse(policy.eligible_for_decision(item, {'portal': 'ready'}))
            self.assertTrue(policy.eligible_for_decision(item, {'mail': 'no updates'}))

    def test_numeric_cannot_be_routed_to_visible_none(self):
        with tempfile.TemporaryDirectory() as directory:
            models = RoutingModels(); models.route = 'NONE'
            policy = RecommendationPolicy(models, FakeCompiler(), Path(directory))
            policy.store.observe('e', 'explicit instruction', 'Monday', 1)
            policy.store.apply(proposal('a', day='Monday', kind='numeric', at=None,
                channel='bank', comparator='lt', threshold=10), 'e')
            self.assertEqual(policy.prepare_active(policy.store.active('Monday'), ['bank'])[0].channel, 'bank')

    def test_all_due_actions_survive_query_budget(self):
        class CompilerTwo:
            hits = 0
            def compile(self, evidence_id, text, day, step, channels):
                if text == 'explicit instruction':
                    return [proposal(key, day='Monday', kind='event', at=None, channel=None)
                            for key in ('one', 'two')]
                return []
        class ChooseTwo(FakeModels):
            def evaluate(self, stage, state, questions):
                return {key: ({'noul': .9} if q['type'] == 'noul' else
                              {'choice': 'task_1' if key.endswith('_0') else 'task_2'})
                        for key, q in questions.items()}
        messages = [dict(role='system', content='Observed only'),
                    dict(role='user', content='explicit instruction'),
                    dict(role='user', content='=== Monday ==='),
                    dict(role='user', content='Both cues occur.\nA) Wait\nStep action menu:\n- task_1: Do one\n- task_2: Do two')]
        # An instruction without task bullets is not compiled from the global
        # header; provide it as the current visible vignette instead.
        messages[-1]['content'] = 'explicit instruction\nA) Wait\nStep action menu:\n- task_1: Do one\n- task_2: Do two'
        with tempfile.TemporaryDirectory() as directory:
            policy = RecommendationPolicy(ChooseTwo(), CompilerTwo(), Path(directory), routing='compiled', query_budget=1)
            action = policy(None, 'offline', messages, ['task_1', 'task_2'], ['clock'], 'offline')
            self.assertEqual(action['task_ids'], ['task_1', 'task_2'])

    def test_full_week_budget_and_hidden_state_isolation(self):
        class ConstantCompiler:
            hits = 0
            def __init__(self): self.requests = []
            def compile(self, evidence_id, text, day, step, channels):
                self.requests.append(text)
                if len(self.requests) != 1: return []
                result = []
                for index, channel in enumerate(channels[:5]):
                    item = proposal(str(index), day='EVERYDAY', kind='event', at=None, channel=channel)
                    item['source_quote'] = text.splitlines()[-1]
                    result.append(item)
                return result
        scenario = json.loads((ROOT/'data/synthetic_week_v9.json').read_text())
        for day in scenario['days']:
            for step in day['steps']:
                step['groundtruth']['private_canary'] = 'PRIVATE_GROUNDTRUTH'
        with tempfile.TemporaryDirectory() as directory:
            models, compiler = FakeModels(), ConstantCompiler()
            policy = RecommendationPolicy(models, compiler, Path(directory), routing='compiled', query_budget=2)
            with redirect_stdout(StringIO()), patch.object(pm, 'build_llm_client', return_value=None), \
                    patch.object(pm, 'request_model_action', side_effect=policy):
                entries = pm.run_llm(scenario, str(Path(directory)/'actions.jsonl'), 'offline', None, 5, 'sglang', None, None)
            summary, _, steps = pm.score_log(scenario, entries)
            self.assertEqual(steps, 80)
            self.assertLessEqual(summary['state_query_calls'], 160)
            requests = json.dumps(models.requests) + json.dumps(compiler.requests)
            self.assertNotIn('PRIVATE_GROUNDTRUTH', requests)
            self.assertNotIn('Step action menu', json.dumps(compiler.requests))
            checked_steps = 0
            for entry in entries:
                if 'step_id' in entry:
                    checked_steps += 1
                    self.assertLessEqual(sum(pm.get_action_state_query_counts(entry).values()), 2)
            self.assertEqual(checked_steps, 80)

    def test_exact_judge_cache_records_hit_and_rejects_tampering(self):
        response = dict(model='typesafe/jev-1.13-20260917', provider='TypeSafe',
                        answers={'q': {'noul': .8}}, usage={'input_tokens': 5, 'output_tokens': 1, 'cost': .0001})
        with tempfile.TemporaryDirectory() as directory, patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test-placeholder'}), \
                patch('run_intentions.urllib.request.urlopen', return_value=StringIO(json.dumps(response))) as call:
            out = Path(directory)
            model = CachedModels(out, 'gpt-4o-mini', None, 'jev', judge_cache=out/'cache')
            question = {'q': {'type': 'noul', 'instructions': 'Observed?'}}
            first = model.evaluate('test', {'text': 'observed'}, question)
            self.assertEqual(first, model.evaluate('test', {'text': 'observed'}, question))
            self.assertEqual(call.call_count, 1)
            self.assertEqual(model.judge_hits, 1)
            path = next((out/'cache').glob('*.json'))
            cached = json.loads(path.read_text()); cached['answers'] = {'q': {'noul': .2}}
            path.write_text(json.dumps(cached))
            with self.assertRaisesRegex(RuntimeError, 'verified Jev'):
                model.evaluate('test', {'text': 'observed'}, question)


class HistoricalReplayTests(unittest.TestCase):
    def test_legacy_successful_trajectory_unchanged(self):
        root = ROOT/'runs/local/jev-intentions-20260928'
        if not (root/'jev/model_calls.jsonl').exists():
            self.skipTest('Historical local experiment artifacts not installed')
        calls = [json.loads(line) for line in (root/'jev/model_calls.jsonl').read_text().splitlines()]
        calls = [record for record in calls if record['kind'] == 'jev']
        class Replay(FakeModels):
            model = 'gpt-4o-mini-2024-07-18'
            records = []
            def evaluate(self, stage, state, questions):
                if not questions: return {}
                expected = calls[len(self.requests)]
                actual = dict(model='typesafe/jev-1.13', state=state, questions=questions)
                if stage != expected['stage'] or digest(actual) != digest(expected['request']):
                    raise AssertionError(f'Legacy request changed at {len(self.requests)}: {stage}')
                self.requests.append(actual)
                return deepcopy(expected['response']['answers'])
            def chat(self, *args):
                raise AssertionError('Legacy compiler cache miss')
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            shutil.copytree(root/'compiler-cache', out/'cache')
            model = Replay(); model.out = out
            compiler = Compiler(model, out/'cache')
            policy = IntentionPolicy(model, compiler, out)
            scenario = json.loads((root/'jev/scenario.json').read_text())
            with redirect_stdout(StringIO()), patch.object(pm, 'build_llm_client', return_value=None), \
                    patch.object(pm, 'request_model_action', side_effect=policy):
                entries = pm.run_llm(scenario, str(out/'actions.jsonl'), 'offline', None, 5, 'sglang', None, None)
            summary, _, steps = pm.score_log(scenario, entries)
            self.assertEqual((summary['set_tp'], summary['set_fp'], summary['set_fn']), (26, 0, 8))
            self.assertEqual(steps, 36)
            self.assertEqual(len(model.requests), len(calls))


if __name__ == '__main__':
    unittest.main()
