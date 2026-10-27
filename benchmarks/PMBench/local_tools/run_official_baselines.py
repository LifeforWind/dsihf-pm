#!/usr/bin/env python3
"""Run the unmodified upstream entry point with audited transport and recovery."""
import argparse
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'sim'))
import run_eval as official

from openmodel_config import MODEL
PROVIDER = dict(only=['openai'], allow_fallbacks=False)
CONDITIONS = {
    'single-baseline': ('single_baseline', None),
    'single-todo-ledger': ('todo_ledger', None),
    'heartbeat-proactive': ('single_heartbeat', 'optional'),
    'heartbeat-auto-30m': ('single_heartbeat', 'auto30'),
    'heartbeat-auto-60m': ('single_heartbeat', 'auto60'),
    'hier-union-query': ('multi_union_query', None),
}
LIMITS = dict(responses=1500, input_tokens=15000000, reported_cost=5.)


class StudyStop(BaseException):
    """Bypass upstream handlers that otherwise turn service failure into actions."""


def write(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def append(path, value):
    with path.open('a') as f:
        f.write(json.dumps(value, ensure_ascii=False) + '\n')


def usage(records):
    result = dict(responses=0, input_tokens=0, output_tokens=0, reported_cost=0., missing_cost=0)
    for r in records:
        u = r['response'].get('usage') or {}
        result['responses'] += 1
        result['input_tokens'] += u.get('prompt_tokens', 0)
        result['output_tokens'] += u.get('completion_tokens', 0)
        if u.get('cost') is None:
            result['missing_cost'] += 1
        else:
            result['reported_cost'] += u['cost']
    return result


class AuditedTransport:
    def __init__(self, client, out, cache):
        self.client, self.out, self.cache = client, out, cache
        self.lock = threading.Lock()
        self.used = dict(responses=0, input_tokens=0, output_tokens=0, reported_cost=0., missing_cost=0)
        self.physical = dict(self.used)
        self.calls = self.hits = 0
        self.occurrences = {}
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        from openai.types.chat import ChatCompletion
        original = deepcopy(kwargs)
        wire = deepcopy(kwargs)
        wire.setdefault('extra_body', {})['provider'] = deepcopy(PROVIDER)
        request_key = digest(wire)
        with self.lock:
            self.calls += 1
            ordinal = self.calls
            occurrence = self.occurrences.get(request_key, 0) + 1
            self.occurrences[request_key] = occurrence
            if any(self.physical[k] >= v for k, v in LIMITS.items()):
                raise StudyStop('Protection budget reached; condition is incomplete')
        # Preserve distinct upstream retries, including empty/invalid output retries.
        key = digest([wire, occurrence])
        cache_path = self.cache / (key + '.json')
        cache_hit = cache_path.exists()
        if cache_hit:
            record = json.loads(cache_path.read_text())
            if record['wire_request'] != wire or record['official_request'] != original:
                raise StudyStop('Exact cache request mismatch')
            response = ChatCompletion.model_validate(record['response'])
            with self.lock:
                self.hits += 1
        else:
            for attempt in range(4):
                started = time.monotonic()
                try:
                    response = self.client.chat.completions.create(**wire)
                except Exception as exc:
                    status = getattr(exc, 'status_code', None)
                    record = dict(call=ordinal, attempt=attempt+1, cache_key=key, occurrence=occurrence,
                        official_request=original, wire_request=wire, error_type=type(exc).__name__,
                        status_code=status, error_body=getattr(exc, 'body', None), seconds=time.monotonic()-started)
                    with self.lock:
                        append(self.out/'api_errors.jsonl', record)
                    retryable = status in (429, 500, 502, 503, 504, 520, 522, 524) or type(exc).__name__ in (
                        'APIConnectionError', 'APITimeoutError')
                    if not retryable or attempt == 3:
                        raise StudyStop('Recorded API failure: '+type(exc).__name__) from exc
                    time.sleep(min(2**attempt, 4))
                    continue
                data = response.model_dump(mode='json')
                record = dict(call=ordinal, attempt=attempt+1, cache_key=key, occurrence=occurrence,
                    official_request=original, wire_request=wire, response=data,
                    seconds=time.monotonic()-started)
                accepted = data['model'].split('/')[-1] == MODEL.split('/')[-1] and data.get('provider', '').lower() == 'openai'
                record['identity_accepted'] = accepted
                with self.lock:
                    append(self.out/'model_calls.jsonl', record)
                    delta = usage([record])
                    for k in self.physical:
                        self.physical[k] += delta[k]
                if not accepted:
                    if attempt == 2:
                        raise StudyStop('Returned model/provider identity mismatch')
                    continue
                write(cache_path, record)
                break
            else:
                raise StudyStop('No accepted model response')
        if response.model.split('/')[-1] != MODEL.split('/')[-1]:
            raise StudyStop('Cached model identity mismatch')
        with self.lock:
            append(self.out/'requests.jsonl', dict(call=ordinal, cache_key=key, occurrence=occurrence, cache_hit=cache_hit,
                official_request=original, wire_request=wire))
            delta = usage([record])
            for k in self.used:
                self.used[k] += delta[k]
            write(self.out/'usage.json', dict(logical=self.used, physical=self.physical, cache_hits=self.hits, calls=self.calls))
        return response


def guarded_action(original):
    def call(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        except official.PM_BENCH.InvalidModelResponseError as exc:
            raise StudyStop('Invalid model output; upstream ground-truth fallback prohibited') from exc
    return call


def upstream_command(condition, scenario, out):
    setup, heartbeat = CONDITIONS[condition]
    command = ['sim/run_eval.py', '--setup', setup, '--scenario', str(scenario),
        '--backend', 'openrouter', '--model', MODEL, '--out-dir', str(out/'official'), '--score']
    if heartbeat:
        command += ['--heartbeat-mode', heartbeat]
    return command


def execute(condition, scenario, out, cache, client=None):
    out.mkdir(parents=True, exist_ok=False)
    cache.mkdir(parents=True, exist_ok=True)
    command = upstream_command(condition, scenario, out)
    manifest = dict(status='running', condition=condition, command=command,
        runtime_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        scenario=str(scenario), scenario_sha256=hashlib.sha256(scenario.read_bytes()).hexdigest(),
        model=MODEL, provider=PROVIDER, protection_limits=LIMITS, cache=str(cache),
        started_at_utc=official.PM_BENCH.now_utc_iso())
    write(out/'manifest.json', manifest)
    if client is None:
        raise StudyStop('Use scripts/run.py: an explicitly routed client is required')
    transport = AuditedTransport(client, out, cache)
    modules = [official.PM_BENCH, official.TODO_LEDGER.PM_BENCH,
        official.HIER_UNION.PM_BENCH, official.HIER_BASELINE.PM_BENCH]
    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(sys, 'argv', command))
            for module in modules:
                stack.enter_context(patch.object(module, 'build_llm_client', return_value=transport))
            stack.enter_context(patch.object(official.PM_BENCH, 'request_model_action',
                guarded_action(official.PM_BENCH.request_model_action)))
            code = official.main()
            if code:
                raise StudyStop('Upstream nonzero exit')
        action_logs = list((out/'official').glob('*/*/*.jsonl'))
        action_logs = [p for p in action_logs if not p.name.endswith(('.debug.jsonl', '.ledger.jsonl'))]
        if len(action_logs) != 1:
            raise StudyStop('Missing or ambiguous action log')
        log = action_logs[0]
        entries, metadata = official.PM_BENCH.read_log_with_metadata(str(log))
        scenario_data = official.PM_BENCH.load_scenario(str(scenario))
        expected = [(d['name'],s['id']) for d in scenario_data['days'] for s in d['steps']]
        if [(e['day'],e['step_id']) for e in entries] != expected:
            raise StudyStop('Incomplete or reordered trajectory')
        summary, per_day, steps = official.PM_BENCH.score_log(scenario_data, entries)
        write(out/'score.json', dict(summary=summary, per_day=per_day, steps=steps))
        manifest.update(status='complete', steps=steps, action_log=str(log), run_metadata=metadata)
    except BaseException as exc:
        manifest.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        manifest.update(finished_at_utc=official.PM_BENCH.now_utc_iso(), logical_usage=transport.used,
            physical_usage=transport.physical, cache_hits=transport.hits)
        write(out/'manifest.json', manifest)
    return manifest
