#!/usr/bin/env python3
"""Post-run verification and comparable physical/cache-inclusive usage.

This is the only new component that reads evaluation labels. Its output must
never be passed to the online policy or used to tune a reported test run.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from analyze_amem_small import trace_scores
from run_amem_small import pm, write_json


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def usage(records):
    result = dict(responses=len(records), input_tokens=0, output_tokens=0, seconds=0.,
                  reported_cost=0., responses_without_cost=0)
    for record in records:
        value = record['response'].get('usage') or {}
        result['input_tokens'] += value.get('prompt_tokens', value.get('input_tokens', 0))
        result['output_tokens'] += value.get('completion_tokens', value.get('output_tokens', 0))
        result['seconds'] += record['seconds']
        if value.get('cost') is None:
            result['responses_without_cost'] += 1
        else:
            result['reported_cost'] += value['cost']
    return result


def analyze(out):
    manifest = json.loads((out/'manifest.json').read_text())
    if manifest['status'] != 'complete':
        raise ValueError(f'Incomplete run: {out.name}')
    scenario = json.loads((out/'scenario.json').read_text())
    if hashlib.sha256((out/'scenario.json').read_bytes()).hexdigest() != manifest['subset_sha256']:
        raise ValueError('Scenario snapshot changed')
    if hashlib.sha256(Path(pm.__file__).read_bytes()).hexdigest() != manifest['official_runtime_sha256']:
        raise ValueError('Scorer no longer matches the runtime used')
    for name, expected in manifest['source_hashes'].items():
        if hashlib.sha256((out/'source'/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Source snapshot changed: '+name)
    diagnostics, summary, _, steps = trace_scores(scenario, pm.read_log(str(out/'actions.jsonl')))
    if summary != json.loads((out/'score.json').read_text())['summary']:
        raise ValueError('Official score replay differs')
    decisions = rows(out/'decisions.jsonl')
    if steps != manifest['expected_steps'] or len(decisions) != steps:
        raise ValueError('Missing decisions')
    query_counts = [sum(item['queries'].values()) for item in diagnostics]
    if manifest['query_budget'] is not None and max(query_counts, default=0) > manifest['query_budget']:
        raise ValueError('Per-step query budget violated')
    physical = rows(out/'model_calls.jsonl')
    compiler_log = rows(out/'compilation.jsonl')
    compiler_records = []
    for entry in compiler_log:
        cache = json.loads((Path(manifest['compiler_cache'])/(entry['cache_key']+'.json')).read_text())
        if not cache.get('model_records'):
            raise ValueError('Compiler cache lacks model usage provenance')
        compiler_records.extend(cache['model_records'])
    judge_log = rows(out/'judge_requests.jsonl')
    judge_records = []
    by_stage = Counter()
    for entry in judge_log:
        cache = json.loads((Path(manifest['judge_cache'])/(entry['cache_key']+'.json')).read_text())
        if not cache.get('model_records'):
            raise ValueError('Judge cache lacks model usage provenance')
        judge_records.extend(cache['model_records'])
        by_stage[entry['stage']] += 1
    logical = compiler_records + judge_records
    tp, fp, fn = [summary[key] for key in ('set_tp', 'set_fp', 'set_fn')]
    binding_rows = [b for record in rows(out/'observation_bindings.jsonl') for b in record['bindings']]
    missed_channels = Counter()
    for step in diagnostics:
        for task in step['fn']:
            channel = step['tasks'][task]['required_channel']
            if channel and not step['queries'].get(channel):
                missed_channels[channel] += 1
    result = dict(name=out.name, steps=steps, routing=manifest['routing'],
        query_policy=manifest['query_policy'], query_budget=manifest['query_budget'],
        summary=summary, set_f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
        max_queries_per_step=max(query_counts, default=0), query_count=sum(query_counts),
        seconds=manifest['seconds'], physical_usage=usage(physical),
        logical_compiler_usage=usage(compiler_records), logical_judge_usage=usage(judge_records),
        logical_system_usage=usage(logical), logical_stages=dict(by_stage),
        models=dict(Counter(r['response']['model'] for r in logical)),
        compiler_requests=[r['cache_key'] for r in compiler_log],
        channel_binding_changes=[r for r in binding_rows if r['effective'] != r['compiled']],
        missed_without_query=dict(missed_channels),
        checks=dict(official_score_replayed=True, snapshots_verified=True, query_budget_verified=True))
    write_json(out/'step_diagnostics.json', diagnostics)
    write_json(out/'analysis.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    results = [analyze(path.resolve()) for path in args.runs]
    manifests = [json.loads((path/'manifest.json').read_text()) for path in args.runs]
    checks = dict(same_scenario=len({m['subset_sha256'] for m in manifests}) == 1,
                  same_sources=len({json.dumps(m['source_hashes'], sort_keys=True) for m in manifests}) == 1,
                  same_compiler_requests=all(r['compiler_requests'] == results[0]['compiler_requests'] for r in results))
    if not all(checks.values()):
        raise ValueError('Comparison arms are not controlled: '+str(checks))
    write_json(args.out, dict(runs=results, checks=checks))
    for result in results:
        print(json.dumps({key: result[key] for key in ('name', 'steps', 'set_f1', 'query_count',
                                                       'physical_usage', 'logical_system_usage')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
