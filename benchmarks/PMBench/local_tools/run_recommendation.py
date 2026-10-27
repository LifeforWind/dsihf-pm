#!/usr/bin/env python3
"""Jev-based evidence routing and observation recommendation on PM-Bench.

Reuses the successful typed-intention compiler, source checks, lifecycle,
event judge and action binding. Only the outer harness can read a scenario.
"""
import argparse
from collections import defaultdict
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from unittest.mock import patch

from run_intentions import (Compiler, Models, IntentionPolicy, SOURCE, ROOT, WORKSPACE,
                            append, configure, git, pm, write_json)

RECOMMENDATION_SOURCE = WORKSPACE / 'a-mem/agentic_memory/recommendation.py'
spec = importlib.util.spec_from_file_location('recommendation', RECOMMENDATION_SOURCE)
recommendation = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = recommendation
spec.loader.exec_module(recommendation)
ObservationCandidate = recommendation.ObservationCandidate
ObservationScheduler = recommendation.ObservationScheduler

ABLATIONS = ('none', 'opportunity', 'age', 'coverage')


def ranking_weights(ablation):
    if ablation not in ABLATIONS:
        raise ValueError('Unknown ranking ablation')
    return dict(opportunity=0. if ablation == 'opportunity' else 1.,
                age=0. if ablation == 'age' else .15, age_cap=6,
                additional_coverage=0. if ablation == 'coverage' else .05,
                additional_coverage_cap=4)


class AblationScheduler(ObservationScheduler):
    """Remove one score term while retaining the V1 scheduler's state handling.

    The unablated path returns the original scheduler output verbatim. This
    adapter is experimental; the A-mem implementation remains unchanged.
    """
    def __init__(self, mode='ranked', budget=3, ablation='none'):
        super().__init__(mode, budget)
        self.weights = ranking_weights(ablation)
        if ablation != 'none' and mode != 'ranked':
            raise ValueError('Score ablations require ranked scheduling')
        self.ablation = ablation

    def rank(self, candidates, observed_channels, step, scope, clock_required=False):
        result = super().rank(candidates, observed_channels, step, scope, clock_required)
        if self.ablation == 'none':
            return result
        weights = self.weights
        for row in result['ranked_channels']:
            row['priority'] = (weights['opportunity'] * row['opportunity']
                + weights['age'] * min(row['age_steps'], weights['age_cap'])
                + weights['additional_coverage'] * min(len(row['intention_ids']) - 1,
                                                       weights['additional_coverage_cap']))
        result['ranked_channels'].sort(key=lambda r: (not r['clock_reserved'], -r['priority'], r['channel']))
        result['selected_channel'] = (result['ranked_channels'][0]['channel']
            if result['ranked_channels'] and result['remaining'] != 0 else None)
        return result


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class CachedModels(Models):
    """Share exact observed requests across comparison arms, retaining provenance.

    Physical calls and logical (cache-inclusive) costs are logged separately.
    Nothing is cached by step ID or benchmark labels.
    """
    def __init__(self, *args, judge_cache, require_compiler_cache=False, require_judge_cache=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.require_compiler_cache = require_compiler_cache
        self.require_judge_cache = require_judge_cache
        self.judge_cache = Path(judge_cache)
        self.judge_cache.mkdir(parents=True, exist_ok=True)
        self.judge_hits = self.judge_misses = 0
        self.evaluate_hash = hashlib.sha256(inspect.getsource(Models.evaluate).encode()).hexdigest()

    def chat(self, stage, *args, **kwargs):
        if self.require_compiler_cache and stage.startswith('compile'):
            raise RuntimeError('Frozen compiler cache miss; refusing to change compiled intentions')
        return super().chat(stage, *args, **kwargs)

    def evaluate(self, stage, state, questions):
        if not questions:
            return {}
        request = dict(stage=stage, state=state, questions=questions, judge=self.judge,
                       judge_model=self.judge_model, evaluator_sha256=self.evaluate_hash,
                       jev_version='typesafe/jev-1.13-20260917')
        key = digest(request)
        path = self.judge_cache / f'{key}.json'
        hit = path.exists()
        if hit:
            cached = json.loads(path.read_text())
            if digest(cached['request']) != key or not cached['model_records']:
                raise RuntimeError('Invalid judge cache provenance')
            answers = cached['answers']
            self.judge_hits += 1
        else:
            if self.require_judge_cache:
                raise RuntimeError('Required historical judge cache miss: ' + stage)
            begin = len(self.records)
            answers = super().evaluate(stage, state, questions)
            cached = dict(request=request, answers=answers, model_records=self.records[begin:])
            write_json(path, cached)
            self.judge_misses += 1
        # Refuse malformed or altered cache answers just as for live responses.
        if set(answers) != set(questions):
            raise RuntimeError('Invalid cached answer keys')
        raw_answers = cached['model_records'][-1]['response']
        if self.judge == 'jev':
            if (raw_answers.get('model') != 'typesafe/jev-1.13-20260917'
                    or raw_answers.get('provider') != 'TypeSafe'
                    or raw_answers.get('answers') != answers):
                raise RuntimeError('Judge cache does not match verified Jev response')
        for key_name, question in questions.items():
            answer = answers[key_name]
            if question['type'] == 'noul':
                number = answer.get('noul')
                if (isinstance(number, bool) or not isinstance(number, (float, int))
                        or not math.isfinite(number) or not 0 <= number <= 1):
                    raise RuntimeError('Invalid cached probability')
            elif answer.get('choice') not in question['criteria']:
                raise RuntimeError('Invalid cached choice')
        append(self.out / 'judge_requests.jsonl', dict(stage=stage, cache_key=key, cache_hit=hit))
        return deepcopy(answers)


class RecommendationPolicy(IntentionPolicy):
    def __init__(self, models, compiler, out, *, routing='jev', query_policy='ranked', query_budget=3,
                 ablation='none'):
        super().__init__(models, compiler, out)
        if routing not in ('compiled', 'jev'):
            raise ValueError('Unknown routing mode')
        self.routing = routing
        self.scheduler = AblationScheduler(query_policy, query_budget, ablation)
        self.bindings = {}

    def prepare_active(self, active, allowed_channels):
        if self.routing == 'compiled':
            return active
        # Source versions are immutable. Observation routing is derived metadata,
        # never a rewrite of the user's intention or the compiler's draft.
        pending = [item for item in active if item.kind != 'clock'
                   and (item.id, item.revision, tuple(allowed_channels)) not in self.bindings]
        if pending:
            records = [dict(id=item.id, revision=item.revision, **item.spec(),
                            source_quotes=[source['quote'] for source in item.sources]) for item in pending]
            state = dict(intentions=records, available_channels=list(allowed_channels))
            criteria = {channel: f'Observe the {channel.replace("_", " ")} state channel.'
                        for channel in allowed_channels}
            criteria.update(NONE='This trigger is observable in the scene narrative without a hidden channel.',
                            UNRESOLVED='The available sources do not determine an observation channel.')
            questions = {f'route_{i}': dict(type='choice', criteria=criteria,
                instructions=f'Which available observation channel supplies evidence for the CURRENT '
                f'trigger in intentions[{i}]? Use its source quotes and latest trigger. '
                'The compiled channel is a fallible suggestion, not evidence. '
                'Match what must be observed, not the action that will be performed. '
                'Visible physical events and SMS/text notices can use NONE. '
                'A hidden portal or numeric state requires its corresponding available channel. '
                'Do not predict whether the trigger has occurred; only identify how to observe it.')
                for i in range(len(pending))}
            choices = self.models.evaluate('route_observation_channel', state, questions)
            verification = {f'support_{i}': dict(type='noul', instructions=
                f'For intentions[{i}], is the selected route {choices[f"route_{i}"]["choice"]} '
                'supported as a way to observe the current trigger, using the source quotes and '
                'available channel names? For NONE, the condition must be observable in the visible '
                'scene without hidden state. A merely related topic or the compiled channel alone '
                'is insufficient. UNRESOLVED is not a supported route.') for i in range(len(pending))}
            support = self.models.evaluate('verify_observation_channel', state, verification)
            bindings = []
            for i, item in enumerate(pending):
                choice = choices[f'route_{i}']['choice']
                accepted = choice != 'UNRESOLVED' and support[f'support_{i}']['noul'] >= self.threshold
                # A numeric predicate cannot operate without a channel. Preserve
                # its compiled route if semantic routing claims otherwise.
                if item.kind == 'numeric' and choice == 'NONE':
                    accepted = False
                effective = (None if choice == 'NONE' else choice) if accepted else item.channel
                key = (item.id, item.revision, tuple(allowed_channels))
                self.bindings[key] = effective
                bindings.append(dict(id=item.id, revision=item.revision, compiled=item.channel,
                                     proposed=choice, accepted=accepted, effective=effective,
                                     support=support[f'support_{i}']['noul'], sources=item.sources))
            append(self.out / 'observation_bindings.jsonl', dict(step=self.step, state=state, bindings=bindings))
        result = []
        for original in active:
            item = deepcopy(original)
            item.channel = ('clock' if item.kind == 'clock' else
                            self.bindings[(item.id, item.revision, tuple(allowed_channels))])
            result.append(item)
        return result

    def next_query(self, active, observations, allowed_channels):
        groups = defaultdict(list)
        for item in active:
            channel = 'clock' if item.kind == 'clock' else item.channel
            if channel:
                if channel not in allowed_channels:
                    raise RuntimeError('Observation binding used an unavailable channel')
                groups[channel].append(item)
        clock_required = any(item.kind == 'clock' for item in active)
        candidates = [ObservationCandidate(channel, tuple(item.id for item in items))
                      for channel, items in groups.items()]
        preliminary = self.scheduler.rank(candidates, observations, self.step, self.day, clock_required)
        missing = [item for item in active if item.channel and item.channel not in observations]
        scores = {}
        reserve_clock = clock_required and 'clock' not in observations
        # Avoid an extra model call when no ranking decision can affect selection.
        contested = (preliminary['remaining'] is not None and
                     len(preliminary['ranked_channels']) > preliminary['remaining'] > 0)
        if (self.scheduler.mode == 'ranked' and self.scheduler.ablation != 'opportunity'
                and contested and not reserve_clock):
            state = dict(today=self.day, vignette=self.vignette, observed_channels=observations,
                         intentions=[dict(id=i.id, **i.spec()) for i in missing])
            questions = {f'opportunity_{i}': dict(type='noul', instructions=
                f'Does the visible current vignette or observed channel replies contain a concrete '
                f'cue that makes checking intentions[{i}].channel useful NOW for this intention? '
                'Judge only the visible contextual support for checking. Hidden channel contents '
                'remain unknown; an intention by itself is not evidence its trigger occurred. '
                'Indirect scene cues can support checking. A low score does not establish absence '
                'of the hidden event and will not authorize or forbid execution.')
                for i in range(len(missing))}
            answers = self.models.evaluate('rank_observation_opportunities', state, questions)
            scores = {item.id: answers[f'opportunity_{i}']['noul'] for i, item in enumerate(missing)}
            candidates = [ObservationCandidate(channel, tuple(i.id for i in items),
                            max((scores.get(i.id, 0.) for i in items), default=0.))
                          for channel, items in groups.items()]
        decision = self.scheduler.rank(candidates, observations, self.step, self.day, clock_required)
        append(self.out / 'recommendations.jsonl', dict(step=self.step, day=self.day,
               query_policy=self.scheduler.mode, opportunity_scores=scores, **decision))
        return decision['selected_channel']

    def eligible_for_decision(self, item, observations):
        # A ranking score is not trigger evidence. Budget-excluded hidden state
        # cannot silently be replaced with guesses or earlier-step snapshots.
        eligible = not item.channel or item.channel in observations
        if not eligible:
            append(self.out / 'unobserved_intentions.jsonl', dict(step=self.step, id=item.id,
                   channel=item.channel, reason='required channel not observed in this step'))
        return eligible


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--judge', choices=['jev', 'llm'], default='jev')
    parser.add_argument('--routing', choices=['compiled', 'jev'], default='jev')
    parser.add_argument('--query-policy', choices=['all', 'round_robin', 'ranked'], default='ranked')
    parser.add_argument('--ablation', choices=ABLATIONS, default='none')
    parser.add_argument('--query-budget', type=int, default=3, help='Includes clock; ignored only by all policy')
    parser.add_argument('--days', type=int, choices=range(1, 8), default=7)
    parser.add_argument('--scenario', type=Path, default=ROOT / 'data/synthetic_week_v9.json')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--compiler-cache', type=Path, required=True)
    parser.add_argument('--judge-cache', type=Path, required=True)
    parser.add_argument('--require-compiler-cache', action='store_true')
    parser.add_argument('--require-judge-cache', action='store_true')
    parser.add_argument('--model', default='gpt-4o-mini-2024-07-18')
    parser.add_argument('--base-url', default=os.environ.get('OPENAI_BASE_URL', 'https://relay.example.invalid/v1'))
    parser.add_argument('--max-api-calls', type=int, default=800)
    parser.add_argument('--max-input-tokens', type=int, default=3000000)
    parser.add_argument('--max-jev-usd', type=float, default=1.)
    args = parser.parse_args(argv)
    if args.query_budget < 1 or args.max_api_calls < 1 or args.max_input_tokens < 1 or args.max_jev_usd <= 0:
        parser.error('Budgets must be positive')
    if args.routing == 'jev' and args.judge != 'jev':
        parser.error('Jev routing requires the Jev judge; use compiled for an LLM control')
    if args.ablation != 'none' and args.query_policy != 'ranked':
        parser.error('Score ablations require --query-policy ranked')
    return args


def main(*, args=None, policy_factory=RecommendationPolicy, model_factory=CachedModels,
         extra_manifest=None, extra_sources=(), llm_timeout=60):
    args = parse_args() if args is None else args
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise SystemExit('Use a new empty output directory')
    amem = WORKSPACE / 'a-mem'
    source = args.scenario.resolve()
    scenario = json.loads(source.read_text())
    if args.days > len(scenario['days']):
        raise SystemExit('Requested more days than available')
    scenario['days'] = scenario['days'][:args.days]
    write_json(out / 'scenario.json', scenario)
    manifest = dict(status='initializing', protocol='jev-observation-recommendation-v1',
        judge=args.judge, routing=args.routing, query_policy=args.query_policy, ablation=args.ablation,
        query_budget=None if args.query_policy == 'all' else args.query_budget,
        scope='full-week' if args.days == 7 else 'initial-complete-days',
        source_is_official=source == (ROOT / 'data/synthetic_week_v9.json').resolve(),
        budget_constrained=args.query_policy != 'all', days=args.days,
        expected_steps=sum(len(day['steps']) for day in scenario['days']),
        compiler_model=args.model, base_url=args.base_url, threshold=.5,
        judge_model_requested='typesafe/jev-1.13' if args.judge == 'jev' else 'openai/gpt-4o-mini',
        jev_model='typesafe/jev-1.13-20260917', llm_seed=None,
        source_scenario=str(source), scenario_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        subset_sha256=hashlib.sha256((out/'scenario.json').read_bytes()).hexdigest(),
        compiler_cache=str(args.compiler_cache.resolve()), judge_cache=str(args.judge_cache.resolve()),
        require_compiler_cache=args.require_compiler_cache, require_judge_cache=args.require_judge_cache,
        command=sys.argv, started_utc=pm.now_utc_iso(), fixed_ongoing_choice='A',
        completion_policy='attempt only; no success inferred', native_amem_graph_used=False,
        official_runtime_and_scorer_unchanged=True, ranking_weights=ranking_weights(args.ablation),
        opportunity_scoring_enabled=args.query_policy == 'ranked' and args.ablation != 'opportunity',
        budget=dict(max_api_calls=args.max_api_calls, max_input_tokens=args.max_input_tokens,
                    max_jev_usd=args.max_jev_usd))
    if extra_manifest:
        if set(extra_manifest) & set(manifest):
            raise ValueError('Extra manifest fields cannot overwrite the baseline protocol')
        manifest.update(extra_manifest)
    code = out / 'source'; code.mkdir()
    for file in [Path(__file__), Path(__file__).with_name('run_intentions.py'),
                 Path(__file__).with_name('run_amem_small.py'), SOURCE, RECOMMENDATION_SOURCE,
                 Path(__file__).with_name('analyze_recommendation.py'), *extra_sources]:
        shutil.copyfile(file, code/file.name)
    for repo, label in [(ROOT, 'pmbench'), (amem, 'amem')]:
        manifest[label+'_commit'] = git(repo, 'rev-parse', 'HEAD')
        manifest[label+'_branch'] = git(repo, 'branch', '--show-current')
        (out/f'{label}.patch').write_text(git(repo, 'diff', 'HEAD'))
        (out/f'{label}.status.txt').write_text(git(repo, 'status', '--short'))
    manifest['source_hashes'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in code.iterdir()}
    manifest['official_runtime_sha256'] = hashlib.sha256(Path(pm.__file__).read_bytes()).hexdigest()
    (out/'packages.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
    write_json(out/'manifest.json', manifest)
    models = compiler = policy = None
    started = time.monotonic()
    try:
        configure(amem, args.model, args.base_url)
        # Read locally without exposing the secret to logs or command arguments.
        script = 'from dotenv import dotenv_values; import sys; print(dotenv_values(sys.argv[1])["OPENROUTER_API_KEY"])'
        os.environ['OPENROUTER_API_KEY'] = subprocess.check_output(
            [sys.executable, '-c', script, str(amem/'.env')], text=True).strip()
        from openai import OpenAI
        client = OpenAI(timeout=llm_timeout, max_retries=1)
        judge_client = OpenAI(api_key=os.environ['OPENROUTER_API_KEY'], base_url='https://openrouter.ai/api/v1',
                              timeout=60, max_retries=1)
        models = model_factory(out, args.model, client, args.judge, judge_client=judge_client,
                              judge_cache=args.judge_cache, require_compiler_cache=args.require_compiler_cache,
                              require_judge_cache=args.require_judge_cache, **manifest['budget'])
        compiler = Compiler(models, args.compiler_cache.resolve())
        policy = policy_factory(models, compiler, out, routing=args.routing,
                                      query_policy=args.query_policy, query_budget=manifest['query_budget'],
                                      ablation=args.ablation)
        manifest['status'] = 'running'; write_json(out/'manifest.json', manifest)
        with (out/'runtime.log').open('w') as log, redirect_stdout(log), \
                patch.object(pm, 'build_llm_client', return_value=client), \
                patch.object(pm, 'request_model_action', side_effect=policy):
            entries = pm.run_llm(scenario, str(out/'actions.jsonl'), args.model, None, 5,
                                 'sglang', args.base_url, None)
        summary, per_day, steps = pm.score_log(scenario, entries)
        if steps != manifest['expected_steps']:
            raise RuntimeError('Incomplete trajectory')
        write_json(out/'score.json', dict(summary=summary, per_day=per_day, steps=steps))
        (out/'score.md').write_text(pm.build_markdown_report(summary, per_day, steps))
        manifest.update(status='complete', steps=steps)
        print(json.dumps(summary), flush=True)
    except Exception as exc:
        # Store type, not exception text that might contain a credential-bearing request.
        manifest.update(status='failed', error_type=type(exc).__name__)
        print(f'Run failed: {type(exc).__name__}; see recorded model/API responses.', file=sys.stderr)
        raise
    finally:
        if policy:
            write_json(out/'intention_final.json', policy.store.dump())
        if compiler:
            write_json(out/'compiler_final.json', compiler.draft.dump())
            manifest.update(compiler_cache_hits=compiler.hits, compiler_cache_misses=compiler.misses)
        if models:
            manifest.update(api_calls=models.calls, input_tokens=models.input_tokens,
                output_tokens=models.output_tokens, jev_reported_cost=models.jev_cost,
                judge_cache_hits=models.judge_hits, judge_cache_misses=models.judge_misses)
        manifest.update(seconds=time.monotonic()-started, finished_utc=pm.now_utc_iso())
        write_json(out/'manifest.json', manifest)


if __name__ == '__main__':
    main()
