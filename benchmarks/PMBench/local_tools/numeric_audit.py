"""Audit-only amendment: exact behavior, bounded Mem0 diagnostic score error."""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import sys

import run_memory_baseline as runner
from memory_transport import rows, sha, write, ExperimentStop

SCORE_ATOL = 1e-7


def compare_memory(original, replay, method):
    """Only float scores at rows[*].recall[*].score may differ; order is exact."""
    if method != 'mem0':
        assert original == replay, 'Non-Mem0 memory changed'
        return []
    left, right = deepcopy(original), deepcopy(replay)
    assert len(left) == len(right), 'Memory row count changed'
    differences = []
    for i, (a, b) in enumerate(zip(left, right)):
        assert len(a['recall']) == len(b['recall']), 'Recall length changed'
        for j, (x, y) in enumerate(zip(a['recall'], b['recall'])):
            assert set(x) == set(y), 'Recall fields changed'
            u, v = x.pop('score'), y.pop('score')
            assert type(u) in (float, int) and type(v) in (float, int), 'Score not numeric'
            assert math.isfinite(u) and math.isfinite(v), 'Score not finite'
            delta = abs(u - v)
            assert delta <= SCORE_ATOL, 'Diagnostic score exceeds absolute tolerance'
            if delta:
                differences.append(dict(row=i, rank=j, original=u, replay=v, abs_error=delta))
    assert left == right, 'Memory content, identity, order or metadata changed'
    return differences


def audit(run, cache):
    run, cache = Path(run).resolve(), Path(cache).resolve()
    manifest = json.loads((run / 'manifest.json').read_text())
    dest = run.parent / (run.name + '-audit')
    if not dest.exists():
        try:
            return runner.audit(run, cache)
        except AssertionError as exc:
            if str(exc) != 'memory.jsonl' or manifest['method'] != 'mem0':
                raise
    if (dest / 'validation.json').exists():
        prior = json.loads((dest / 'validation.json').read_text())
        assert prior['action_sha256'] == sha(run / 'actions.jsonl')
        return prior
    assert manifest['status'] == 'complete'
    assert all(sha(p) == h for p, h in manifest['source_hashes'].items())
    replay = dest / 'replay'
    replay_manifest = json.loads((replay / 'manifest.json').read_text())
    assert replay_manifest['status'] == 'complete' and replay_manifest['offline'] is True
    assert replay_manifest['source_hashes'] == manifest['source_hashes']
    assert replay_manifest['identities_consumed'] == manifest['identities_consumed']
    scene = json.loads((run / 'scenario.json').read_text())
    perturbed = deepcopy(scene)
    for day in perturbed['days']:
        for step in day['steps']:
            step['groundtruth'] = {}
            step['private_audit_canary'] = 'DO NOT QUERY OR ACT FROM THIS PRIVATE VALUE'
    assert json.loads((replay / 'scenario.json').read_text()) == perturbed
    entries = runner.pm.read_log(str(run / 'actions.jsonl'))
    assert entries == runner.pm.read_log(str(replay / 'actions.jsonl'))
    score, per_day, steps = runner.pm.score_log(scene, entries)
    assert json.loads((run / 'score.json').read_text()) == dict(summary=score, per_day=per_day, steps=steps)
    for name in ('decisions.jsonl', 'ingestion.jsonl', 'budget_events.jsonl', 'pma_events.jsonl', 'mem0_writes.jsonl'):
        assert rows(run / name) == rows(replay / name), name
    differences = compare_memory(rows(run / 'memory.jsonl'), rows(replay / 'memory.jsonl'), manifest['method'])
    assert json.loads((run / 'store.json').read_text()) == json.loads((replay / 'store.json').read_text())
    original_requests, replay_requests = rows(run / 'transport/requests.jsonl'), rows(replay / 'transport/requests.jsonl')
    assert [r['cache_key'] for r in original_requests] == [r['cache_key'] for r in replay_requests]
    assert all(r['cache_hit'] for r in replay_requests)
    assert not rows(replay / 'transport/model_calls.jsonl')
    assert not rows(replay / 'transport/api_errors.jsonl')
    write(dest / 'numeric_differences.json', differences)
    result = dict(status='passed', official_score_recomputed=True, offline_exact_replay=True,
        private_groundtruth_perturbed=True, action_sha256=sha(run / 'actions.jsonl'),
        steps=len(entries), requests=len(original_requests), query_total=manifest['query_total'],
        audit_semantics='Exact actions, ordered memory contents, ingestion, stores and model requests; Mem0 diagnostic score absolute tolerance only',
        diagnostic_score_atol=SCORE_ATOL, diagnostic_score_differences=len(differences),
        diagnostic_score_max_abs_error=max((d['abs_error'] for d in differences), default=0.0),
        diagnostic_differences_sha256=sha(dest / 'numeric_differences.json'),
        amendment_source=str(Path(__file__).resolve()), amendment_sha256=sha(__file__))
    write(dest / 'validation.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.out, args.cache), ensure_ascii=False))
