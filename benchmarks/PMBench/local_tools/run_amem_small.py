#!/usr/bin/env python3
"""Run unchanged PM-Bench through a visible-message-only A-mem adapter.

PM-Bench's venv hosts the simulator/scorer. A separate A-mem venv process hosts
the unmodified memory system. Model errors raise RuntimeError, never the
InvalidModelResponseError caught by upstream's label-derived fallback.
"""
import argparse
from contextlib import redirect_stdout
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
sys.path.insert(0, str(ROOT / 'sim'))
import pm_bench as pm


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()


def configure(amem_root, model, base_url):
    # Read credentials without logging them or copying them to experiment records.
    script = 'from dotenv import dotenv_values; import json,sys; print(json.dumps(dotenv_values(sys.argv[1])))'
    values = json.loads(subprocess.check_output([
        sys.executable, '-c', script, str(amem_root / '.env')], text=True))
    from openmodel_config import BASE, KEY
    if base_url.rstrip('/') != BASE:
        raise RuntimeError('Endpoint differs from the selected release profile')
    os.environ['OPENAI_API_KEY'] = KEY
    os.environ['OPENAI_BASE_URL'] = base_url.rstrip('/')
    os.environ['PMBENCH_MODEL'] = model
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['ANONYMIZED_TELEMETRY'] = 'False'


class MemoryWorker:
    def __init__(self, amem_root, out, runtime_root=None, expected_model=None,
                 max_memory_calls=800, max_memory_input_tokens=3000000, *,
                 restore_state=None, initial_calls=0, initial_input_tokens=0,
                 model_identity_retries=0):
        self.stderr = (out / 'memory_stderr.log').open('w')
        command = [str((runtime_root or amem_root) / '.venv/bin/python'), '-u',
                   str(Path(__file__).with_name('amem_worker.py')),
                   '--amem-root', str(amem_root), '--out', str(out)]
        if expected_model:
            command += ['--expected-model', expected_model, '--max-model-calls', str(max_memory_calls),
                        '--max-input-tokens', str(max_memory_input_tokens)]
        if restore_state:
            command += ['--restore-state', str(restore_state), '--initial-calls', str(initial_calls),
                        '--initial-input-tokens', str(initial_input_tokens)]
        if model_identity_retries:
            command += ['--model-identity-retries', str(model_identity_retries)]
        self.process = subprocess.Popen(command,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr, text=True, bufsize=1)
        ready = self.process.stdout.readline()
        if not ready or not json.loads(ready).get('ready'):
            raise RuntimeError('Memory worker initialization failed; see memory_stderr.log')

    def rpc(self, **request):
        self.process.stdin.write(json.dumps(request) + '\n')
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('Memory worker terminated')
        reply = json.loads(line)
        if not reply['ok']:
            raise RuntimeError(reply['error'])
        return reply['result']

    def close(self):
        self.process.stdin.close()
        self.process.wait(timeout=30)
        self.stderr.close()


class VisiblePolicy:
    """No scenario, ground truth, hidden state, or future steps are passed here."""
    def __init__(self, mode, client, model, out, memory=None, history_steps=2, k=5,
                 max_decision_calls=220, max_input_tokens=1500000, expected_model=None):
        self.mode, self.client, self.model, self.out = mode, client, model, out
        self.memory, self.history_steps, self.k = memory, history_steps, k
        self.step_start = None
        self.step_starts = []
        self.day = ''
        self.day_header = ''
        self.global_header = ''
        self.calls = 0
        self.input_tokens = self.output_tokens = 0
        self.max_step_rounds = 16
        self.rounds = 0
        self.seen_headers = set()
        self.max_decision_calls = max_decision_calls
        self.max_input_tokens = max_input_tokens
        self.expected_model = expected_model

    def context(self, messages):
        starts = [i for i, m in enumerate(messages) if m['role'] == 'user'
                  and '\nStep action menu' in m['content']]
        start = starts[-1]
        if start != self.step_start:
            self.step_start = start
            self.step_starts.append(start)
            self.rounds = 0
            self.global_header = messages[1]['content']
            headers = [m['content'] for m in messages[2:start]
                       if m['role'] == 'user' and m['content'].startswith('=== ')]
            self.day_header = headers[-1]
            self.day = self.day_header.splitlines()[0].strip('= ')
            if self.memory:
                for header in [self.global_header, self.day_header]:
                    if header not in self.seen_headers:
                        # Preserve raw instructions; split only at existing line boundaries.
                        for line in header.splitlines():
                            if line.strip() and not line.startswith('=== '):
                                self.memory.rpc(op='add', content=f'Instructions received on {self.day}: {line}')
                        self.seen_headers.add(header)
        current = copy.deepcopy(messages[start:])
        visible = current[0]['content'].split('\nStep action menu')[0]
        query_replies = [m['content'] for m in current if m['content'].startswith('State [')]
        query = f'{self.day}\n{visible}\n' + '\n'.join(query_replies)
        retrieved = self.memory.rpc(op='search', query=query, k=self.k) if self.memory else []
        if self.mode == 'single':
            return copy.deepcopy(messages), retrieved, query
        old_starts = self.step_starts[:-1]
        history_start = old_starts[-self.history_steps] if len(old_starts) >= self.history_steps else (old_starts[0] if old_starts else start)
        result = [copy.deepcopy(messages[0]), {'role': 'user', 'content': self.global_header},
                  {'role': 'user', 'content': self.day_header}]
        if retrieved:
            result.append({'role': 'user', 'content': 'Retrieved past observations (may be outdated; follow latest instructions):\n' + json.dumps(retrieved)})
        result.extend(copy.deepcopy(messages[history_start:]))
        return result, retrieved, query

    def __call__(self, client, model, messages, allowed_task_ids, allowed_channels, backend, **kwargs):
        actual, retrieved, query = self.context(messages)
        self.calls += 1
        self.rounds += 1
        if (self.calls > self.max_decision_calls or self.input_tokens > self.max_input_tokens
                or self.rounds > self.max_step_rounds):
            raise RuntimeError('Predeclared call/token/query budget reached; run is incomplete')
        schema = pm.build_action_schema(allowed_task_ids, allowed_channels)
        started = time.monotonic()
        try:
            response = self.client.chat.completions.create(
                model=model, messages=actual, temperature=0, max_tokens=256,
                response_format={'type': 'json_schema', 'json_schema': {
                    'name': 'pm_action', 'strict': True, 'schema': schema}})
            if self.expected_model:
                with (self.out / 'decision_responses.jsonl').open('a') as f:
                    f.write(json.dumps(dict(call=self.calls, request_model=model,
                        response=response.model_dump(), seconds=time.monotonic()-started)) + '\n')
                if response.model != self.expected_model:
                    raise ValueError('Unexpected decision model: ' + response.model)
            raw = response.choices[0].message.content
            if response.choices[0].finish_reason != 'stop':
                raise ValueError('Model response truncated')
            action = json.loads(raw)
            validate_action(action, allowed_task_ids, allowed_channels)
        except Exception as exc:
            with (self.out / 'decision_errors.jsonl').open('a') as f:
                f.write(json.dumps(dict(call=self.calls, error_type=type(exc).__name__,
                    status_code=getattr(exc, 'status_code', None))) + '\n')
            # Intentionally NOT pm.InvalidModelResponseError: upstream uses labels on that path.
            raise RuntimeError('Model action failed (no ground-truth fallback): ' + type(exc).__name__) from exc
        self.input_tokens += response.usage.prompt_tokens
        self.output_tokens += response.usage.completion_tokens
        record = {'call': self.calls, 'step_ordinal': len(self.step_starts), 'day': self.day,
                  'round': self.rounds, 'messages': actual, 'retrieval_query': query,
                  'retrieved': retrieved, 'action': action, 'usage': response.usage.model_dump(),
                  'model_returned': response.model, 'seconds': time.monotonic() - started}
        with (self.out / 'decisions.jsonl').open('a') as f:
            f.write(json.dumps(record) + '\n')
        if action['action'] == 'choose':
            if self.memory:
                current = messages[self.step_start:]
                vignette = current[0]['content'].split('\nStep action menu')[0]
                states = '\n'.join(m['content'] for m in current if m['content'].startswith('State ['))
                menu = dict(re.findall(r'(task_\d+): ([^\n]+)', current[0]['content']))
                attempted = [menu[t] for t in action['task_ids']]
                self.memory.rpc(op='add', content=f"Past step {len(self.step_starts)} on {self.day}:\n{vignette}\n{states}\nAttempted actions (not confirmation of success): {json.dumps(attempted)}")
            print(f"{self.mode}: step {len(self.step_starts)} done; {self.calls} decision calls; {self.input_tokens} input tokens", file=sys.__stdout__, flush=True)
        return action


def validate_action(action, ids, channels):
    if set(action) != {'action', 'choice', 'task_ids', 'channel'}:
        raise ValueError('Invalid action fields')
    tasks = action['task_ids']
    if not isinstance(tasks, list) or len(set(tasks)) != len(tasks) or len(tasks) > pm.MAX_ACTION_TASK_IDS or any(t not in ids for t in tasks):
        raise ValueError('Invalid action handles')
    if action['action'] == 'choose':
        if action['choice'] not in ['A', 'B', 'C'] or action['channel'] != 'NONE':
            raise ValueError('Invalid choose action')
    elif action['action'] in ['query_state', 'check_time']:
        if action['choice'] != 'NONE' or tasks or action['channel'] not in channels:
            raise ValueError('Invalid query action')
    else:
        raise ValueError('Unsupported action')
