"""Run one PMA / dense RAG / Mem0 trajectory and exact offline audit."""
import argparse
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
from unittest.mock import patch

from memory_transport import (ROOT, WORKSPACE, MODEL, EMBED_MODEL, Transport,
    ExperimentStop, append, rows, write, sha)

sys.path.insert(0, str(ROOT / 'sim'))
import pm_bench as pm
from step3_budget import QueryBudget, instrument, guarded_resolver
from memory_methods import MemoryPolicy, PMA_ROOT, MEM0_ROOT, COMMON

PMA_COMMIT = '89e5c0d6aadfe531a1aee42fd290d48be89973dd'
MEM0_COMMIT = '0724291bd864cf807e88ee8981c8226644e8fb42'


def source_files():
    paths = [ROOT / 'local_tools' / n for n in (
        'memory_transport.py', 'embedding_redirect.py', 'memory_methods.py', 'run_memory_baseline.py',
        'step3_budget.py', 'matched_budget.py', 'run_official_baselines.py')]
    paths += [ROOT / 'sim/pm_bench.py', ROOT / 'sim/run_eval.py', COMMON]
    paths += list((PMA_ROOT / 'src/memory_agent/memory').glob('*.py'))
    paths += list((MEM0_ROOT / 'mem0').rglob('*.py'))
    paths += [MEM0_ROOT / 'mem0/memory/oss_notices_config.json']
    return paths


def verify_upstreams():
    for repo, expected in ((PMA_ROOT, PMA_COMMIT), (MEM0_ROOT, MEM0_COMMIT)):
        actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
        dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo, text=True).strip()
        if actual != expected or dirty:
            raise ExperimentStop('Upstream source changed: ' + str(repo))


def execute(method, scenario_path, out, cache, offline=False):
    verify_upstreams()
    scenario_path, out, cache = map(lambda p: Path(p).resolve(), (scenario_path, out, cache))
    out.mkdir(parents=True, exist_ok=False)
    cache.mkdir(parents=True, exist_ok=True)
    scene = json.loads(scenario_path.read_text())
    shutil.copyfile(scenario_path, out / 'scenario.json')
    transport = Transport(out / 'transport', cache / 'requests', offline)
    budget = QueryBudget(out)
    manifest = dict(status='running', method=method, model=MODEL, embedding_model=EMBED_MODEL if method != 'pma' else None,
        embedding_api_base_url=transport.embedding_base, embedding_wire_model='text-embedding-3-small' if method != 'pma' else None,
        api_base_url=transport.base, credential_source='chat: selected release profile; embeddings: a-mem/.env:OPENAI_API_KEY',
        step_cap=3, episode_cap=None, scenario_sha256=sha(scenario_path),
        source_hashes={str(p): sha(p) for p in source_files()},
        runtime_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        pma_commit=PMA_COMMIT, mem0_commit=MEM0_COMMIT, started_utc=pm.now_utc_iso(),
        offline=offline, configuration=dict(top_k=5, max_inject_chars=8000, temperature=0,
            action_context='system + standing plan + current day plan + current step queries + memory',
            rag_chunk_chars=1000, rag_overlap_chars=100, pma_window=8,
            pma_original_system_prompts=True, mem0_infer=True, mem0_threshold=0.0))
    write(out / 'manifest.json', manifest)
    policy = None
    try:
        with ExitStack() as stack:
            if offline:
                def deny(*args, **kwargs):
                    raise ExperimentStop('Offline audit forbids network access')
                stack.enter_context(patch.object(socket.socket, 'connect', deny))
            policy = MemoryPolicy(method, transport, budget, out, cache, stack)
            policy.original_request = pm.request_model_action
            stack.enter_context(patch.object(pm, 'build_llm_client', return_value=None))
            stack.enter_context(patch.object(pm, 'request_model_action', side_effect=policy))
            stack.enter_context(patch.object(pm, '_matched_budget', budget, create=True))
            stack.enter_context(patch.object(pm, 'resolve_state_query_items', guarded_resolver(pm.resolve_state_query_items, budget)))
            run = instrument(pm, 'run_llm', budget, out)
            with (out / 'runtime.log').open('w') as f, redirect_stdout(f):
                entries = run(scene, str(out / 'actions.jsonl'), MODEL, None, 5, 'openrouter', None, None, max_tokens=1024)
            expected = [(d['name'], s['id']) for d in scene['days'] for s in d['steps']]
            if [(e['day'], e['step_id']) for e in entries] != expected:
                raise ExperimentStop('Incomplete trajectory')
            counts = [sum(pm.get_action_state_query_counts(e).values()) for e in entries]
            if max(counts, default=0) > 3 or sum(counts) != budget.total:
                raise ExperimentStop('Observation accounting mismatch')
            score, per_day, steps = pm.score_log(scene, entries)
            write(out / 'score.json', dict(summary=score, per_day=per_day, steps=steps))
            write(out / 'store.json', policy.memory.snapshot())
            if transport.logical != len(transport.trace) or policy.tape.position != len(policy.tape.values):
                raise ExperimentStop('Recovery terminated before the recorded request/native prefix')
            manifest.update(status='complete', steps=len(entries), query_total=budget.total,
                identities_consumed=policy.tape.position, finished_utc=pm.now_utc_iso())
    except BaseException as exc:
        manifest.update(status='failed', error_type=type(exc).__name__, error=str(exc), finished_utc=pm.now_utc_iso())
        write(out / 'manifest.json', manifest)
        raise
    finally:
        if policy is not None:
            policy.memory.close()
    write(out / 'manifest.json', manifest)
    return out


def audit(run, cache):
    run, cache = Path(run).resolve(), Path(cache).resolve()
    manifest = json.loads((run / 'manifest.json').read_text())
    if manifest['status'] != 'complete':
        raise ExperimentStop('Cannot audit an incomplete result')
    assert all(sha(p) == h for p, h in manifest['source_hashes'].items())
    scene = json.loads((run / 'scenario.json').read_text())
    entries = pm.read_log(str(run / 'actions.jsonl'))
    score, per_day, steps = pm.score_log(scene, entries)
    assert json.loads((run / 'score.json').read_text()) == dict(summary=score, per_day=per_day, steps=steps)
    dest = run.parent / (run.name + '-audit')
    if (dest / 'validation.json').exists():
        prior = json.loads((dest / 'validation.json').read_text())
        assert prior['action_sha256'] == sha(run / 'actions.jsonl')
        return prior
    dest.mkdir(exist_ok=False)
    for day in scene['days']:
        for step in day['steps']:
            step['groundtruth'] = {}
            step['private_audit_canary'] = 'DO NOT QUERY OR ACT FROM THIS PRIVATE VALUE'
    write(dest / 'perturbed.json', scene)
    replay = execute(manifest['method'], dest / 'perturbed.json', dest / 'replay', cache, True)
    assert entries == pm.read_log(str(replay / 'actions.jsonl'))
    for name in ('decisions.jsonl', 'memory.jsonl', 'ingestion.jsonl', 'budget_events.jsonl', 'pma_events.jsonl', 'mem0_writes.jsonl'):
        assert rows(run / name) == rows(replay / name), name
    assert json.loads((run / 'store.json').read_text()) == json.loads((replay / 'store.json').read_text())
    original_requests, replay_requests = rows(run / 'transport/requests.jsonl'), rows(replay / 'transport/requests.jsonl')
    assert [r['cache_key'] for r in original_requests] == [r['cache_key'] for r in replay_requests]
    assert all(r['cache_hit'] for r in replay_requests)
    result = dict(status='passed', official_score_recomputed=True, offline_exact_replay=True,
        private_groundtruth_perturbed=True, action_sha256=sha(run / 'actions.jsonl'),
        steps=len(entries), requests=len(original_requests), query_total=manifest['query_total'])
    write(dest / 'validation.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['run', 'audit'])
    parser.add_argument('--method', choices=['pma', 'rag', 'mem0'])
    parser.add_argument('--scenario', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    if args.command == 'run':
        execute(args.method, args.scenario, args.out, args.cache, args.offline)
    else:
        print(json.dumps(audit(args.out, args.cache), ensure_ascii=False))
