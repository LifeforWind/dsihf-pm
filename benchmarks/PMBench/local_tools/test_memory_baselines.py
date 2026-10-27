"""Independent checks for visibility, recovery, and hard query boundaries."""
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import numpy as np

from memory_transport import NativeTape, ExperimentStop, Transport, MODEL, write, append
from memory_methods import MemoryPolicy, RagMemory, chunk_text, visible_note
from step3_budget import QueryBudget, guarded_resolver
from run_official_baselines import StudyStop


def test_fourth_query_blocked_before_lookup_and_next_step_reset(tmp_path):
    budget = QueryBudget(tmp_path)
    calls = []
    resolve = guarded_resolver(lambda ch: calls.append(ch), budget)
    budget.begin('Monday', 's1')
    for ch in ['clock', 'clock', 'email']:
        resolve(ch)
    with pytest.raises(StudyStop):
        resolve('calendar')
    assert calls == ['clock', 'clock', 'email']
    budget.begin('Monday', 's2')
    resolve('clock')
    assert budget.total == 4 and budget.remaining == 2
    budget.begin('Tuesday', 's1')
    assert budget.total == 4 and budget.remaining == 3


def test_exhausted_action_schema_and_query_notice(tmp_path):
    budget = QueryBudget(tmp_path)
    budget.begin('Monday', 's1')
    for _ in range(3):
        budget.charge('clock')
    from run_memory_baseline import pm
    request = dict(messages=[], response_format=dict(json_schema=dict(schema=pm.build_action_schema(['task_1'], ['clock']))))
    result = budget.request(request)
    assert result['response_format']['json_schema']['schema']['properties']['action']['enum'] == ['choose']
    assert '0 queries remain' in result['messages'][-1]['content']
    assert request['messages'] == []


def test_tape_exact_recovery_and_order_guard(tmp_path):
    path = tmp_path / 'tape.jsonl'
    tape = NativeTape(path)
    uid = tape.uuid_module('native').uuid4()
    now = tape.datetime_class('native').now()
    replay = NativeTape(path, True)
    assert replay.uuid_module('native').uuid4() == uid
    assert replay.datetime_class('native').now() == now
    with pytest.raises(ExperimentStop):
        replay.uuid_module('native').uuid4()
    with pytest.raises(ExperimentStop):
        NativeTape(path, True).datetime_class('native').now()


def test_rag_preserves_original_chunks_and_cosine_order(tmp_path):
    memory = RagMemory(None, tmp_path)
    vector_map = {'red': [1, 0], 'blue': [0, 1], 'purple': [.6, .8]}
    memory.vectors_for = lambda texts: [vector_map[t] for t in texts]
    memory.add('red', {'day': 'Monday'})
    memory.add('blue', {'day': 'Monday'})
    assert memory.recall('purple', {}) == 'blue\n\nred'
    assert [r['id'] for r in memory.last_recall] == [1, 0]
    text = 'A' * 999 + 'future commitment' + 'B' * 1050
    chunks = chunk_text(text)
    assert chunks[0] + ''.join(c[100:] for c in chunks[1:]) == text


def test_public_note_contains_queries_without_local_action_handles():
    messages = [dict(role='user', content='At the desk.\nA) Work\nB) Rest\nC) Leave\nStep action menu\n- task_1: Send report.'),
        dict(role='user', content='State [email]: A manager asked to send the report.'),
        dict(role='user', content='State [clock]: Time: 15:05')]
    text = visible_note(messages)
    assert 'task_1' not in text and 'A) Work' not in text
    assert '15:05' in text and 'manager' in text and 'At the desk' in text


def test_shared_policy_retains_instep_replies_but_drops_previous_scene(tmp_path):
    class Memory:
        last_recall = []
        def __init__(self, *args):
            self.writes, self.queries = [], []
        def add(self, text, metadata):
            self.writes.append(text)
        def recall(self, query, context):
            self.queries.append(query)
            return 'remembered past event'
    with ExitStack() as stack:
        stack.enter_context(patch('memory_methods.RagMemory', Memory))
        budget = QueryBudget(tmp_path)
        policy = MemoryPolicy('rag', SimpleNamespace(offline=False), budget, tmp_path, tmp_path, stack)
        captured = []
        def choose(c, m, messages, ids, channels, backend, **kwargs):
            captured.append(deepcopy(messages))
            return dict(action='choose', choice='A', task_ids=['task_1'], channel='NONE')
        policy.original_request = choose
        first = [dict(role='system', content='Public instructions'),
            dict(role='user', content='Regular tasks for every day\nDrink water at noon.'),
            dict(role='user', content='=== Monday ===\nSend the report when the manager asks.'),
            dict(role='user', content='Old scene.\nA) Work\nStep action menu\n- task_1: Send report.')]
        budget.begin('Monday', 's1')
        policy(None, MODEL, first, ['task_1'], ['clock'], 'unused')
        second = first + [dict(role='assistant', content='Previous choice'),
            dict(role='user', content='New scene.\nA) Work\nStep action menu\n- task_1: Drink water.')]
        budget.begin('Monday', 's2')
        policy(None, MODEL, second, ['task_1'], ['clock'], 'unused')
        second += [dict(role='user', content='State [clock]: Time: 12:00'),
            dict(role='user', content='Invalid choice; use A, B, or C.')]
        policy(None, MODEL, second, ['task_1'], ['clock'], 'unused')
        prompt = json.dumps(captured[-1])
        assert 'Old scene' not in prompt and 'Previous choice' not in prompt
        assert 'New scene' in prompt and 'Time: 12:00' in prompt and 'Invalid choice' in prompt
        assert 'Drink water' in policy.memory.writes[-1]
        assert all('task_1' not in x for x in policy.memory.writes)
        assert 'Old scene' not in policy.memory.queries[-1]


def test_transport_offline_does_not_call_network(tmp_path):
    transport = Transport(tmp_path / 'out', tmp_path / 'cache', offline=True)
    with pytest.raises(ExperimentStop):
        transport.create(model=MODEL, messages=[dict(role='user', content='test')])
    assert not (tmp_path / 'out/model_calls.jsonl').exists()


def test_transport_prohibits_nonfrozen_model(tmp_path):
    transport = Transport(tmp_path / 'out', tmp_path / 'cache', offline=True)
    with pytest.raises(ExperimentStop):
        transport.create(model='other-model', messages=[])


def test_native_invalid_choice_does_not_write_completed_step(tmp_path):
    class Memory:
        last_recall = []
        def __init__(self, *args):
            self.writes = []
        def add(self, text, metadata):
            self.writes.append(text)
        def recall(self, *args):
            return ''
    with ExitStack() as stack:
        stack.enter_context(patch('memory_methods.RagMemory', Memory))
        budget = QueryBudget(tmp_path)
        budget.begin('Monday', 's1')
        policy = MemoryPolicy('rag', SimpleNamespace(offline=False), budget, tmp_path, tmp_path, stack)
        policy.original_request = lambda *a, **kw: dict(action='choose', choice='NONE', channel='NONE', task_ids=['task_1'])
        policy(None, MODEL, [dict(role='user', content='Desk.\nStep action menu\n- task_1: Send report.')], ['task_1'], ['clock'], '')
        assert policy.memory.writes == []


def test_recovery_rejects_changed_request_before_network_or_cache(tmp_path):
    cache = tmp_path / 'cache'
    cache.mkdir()
    append(tmp_path / 'request_tape.jsonl', dict(kind='chat', stage='unspecified', cache_key='wrong-prior-key'))
    transport = Transport(tmp_path / 'out', cache, offline=True)
    with pytest.raises(ExperimentStop, match='prefix differs'):
        transport.create(model=MODEL, messages=[dict(role='user', content='changed request')])
    assert transport.logical == 0
    assert not (tmp_path / 'out/model_calls.jsonl').exists()

