"""Three memory backends over the same public PM-Bench stream."""
import asyncio
from copy import deepcopy
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from memory_transport import (WORKSPACE, MODEL, EMBED_MODEL, ExperimentStop,
                              NativeTape, append, write, digest)

PMA_ROOT = WORKSPACE / 'benchmarks/proactive-memory-agent'
MEM0_ROOT = WORKSPACE / 'benchmarks/mem0'
COMMON = WORKSPACE / 'benchmarks/PIS/code/pm_memory/common.py'
spec = importlib.util.spec_from_file_location('public_memory_common', COMMON)
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)

TOP_K = 5
MAX_INJECT = 8000


def chunk_text(text, size=1000, overlap=100):
    return [text[start:start + size] for start in range(0, len(text), size - overlap)]


def visible_note(messages):
    """Only texts actually supplied by the simulator; omit local menu handles."""
    return '\n'.join(common.scrub_task_handles(common.strip_abc_options(
        common.strip_action_menu(m['content']))) for m in messages
        if m.get('role') == 'user' and isinstance(m.get('content'), str))


class RagMemory:
    def __init__(self, transport, out):
        self.transport, self.out = transport, out
        self.documents, self.vectors = [], []
        self.last_recall = []

    def vectors_for(self, texts):
        if not texts:
            return []
        with self.transport.role('rag.embedding'):
            response = self.transport.embed(model=EMBED_MODEL, input=texts,
                dimensions=1536, encoding_format='float')
        vectors = [r.embedding for r in sorted(response.data, key=lambda x: x.index)]
        if len(vectors) != len(texts) or any(len(v) != 1536 for v in vectors):
            raise ExperimentStop('RAG embedding shape mismatch')
        return vectors

    def add(self, text, metadata):
        chunks = chunk_text(text)
        vectors = self.vectors_for(chunks)
        for chunk, vector in zip(chunks, vectors):
            self.documents.append(dict(id=len(self.documents), text=chunk, metadata=deepcopy(metadata)))
            self.vectors.append(vector)

    def recall(self, query, context):
        if not self.documents:
            self.last_recall = []
            return ''
        q = np.asarray(self.vectors_for([query])[0], dtype=np.float64)
        matrix = np.asarray(self.vectors, dtype=np.float64)
        scores = matrix @ q / np.maximum(np.linalg.norm(matrix, axis=1) * np.linalg.norm(q), 1e-30)
        order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))[:TOP_K]
        self.last_recall = [dict(self.documents[i], score=float(scores[i])) for i in order]
        return common.truncate('\n\n'.join(r['text'] for r in self.last_recall), MAX_INJECT)

    def snapshot(self):
        return dict(documents=self.documents, vector_sha256=digest(self.vectors))

    def close(self):
        pass


class Mem0Memory:
    def __init__(self, transport, out, tape, stack):
        # Load the pinned checkout, not a different installed release.
        sys.path.insert(0, str(MEM0_ROOT))
        os.environ['MEM0_DIR'] = str(out / 'mem0_config')
        from mem0 import Memory
        import mem0.memory.main as main
        import mem0.memory.storage as storage
        import mem0.configs.prompts as prompts
        from mem0.utils.spacy_models import get_nlp_full, get_nlp_lemma
        if get_nlp_full() is None or get_nlp_lemma() is None:
            raise ExperimentStop('Mem0 NLP dependencies unavailable; no silent degraded run')
        for module, label in ((main, 'mem0.main'), (storage, 'mem0.storage')):
            stack.enter_context(patch.object(module, 'uuid', tape.uuid_module(label)))
            stack.enter_context(patch.object(module, 'datetime', tape.datetime_class(label)))
        stack.enter_context(patch.object(prompts, 'datetime', tape.datetime_class('mem0.prompts')))
        config = dict(
            llm=dict(provider='openai', config=dict(model=MODEL, temperature=0, max_tokens=4096,
                openai_base_url=transport.base, api_key=__import__('qwen_wire').KEY)),
            embedder=dict(provider='openai', config=dict(model=EMBED_MODEL, embedding_dims=1536,
                openai_base_url=transport.embedding_base, api_key=os.environ['OPENAI_API_KEY'])),
            vector_store=dict(provider='qdrant', config=dict(path=str(out / 'qdrant'),
                collection_name='pmbench', embedding_model_dims=1536)),
            history_db_path=str(out / 'history.sqlite'))
        self.memory = Memory.from_config(config)
        if self.memory.vector_store._get_bm25_encoder() is None:
            raise ExperimentStop('Mem0 BM25 encoder unavailable; no silent degraded run')
        for client, base in ((self.memory.llm.client, transport.base), (self.memory.embedding_model.client, transport.embedding_base)):
            if str(client.base_url).rstrip('/') != base:
                raise ExperimentStop('Mem0 client route mismatch')
        # The official extraction/retrieval implementations remain unchanged.
        self.memory.llm.client = transport
        self.memory.embedding_model.client = transport
        self.transport, self.out = transport, out
        self.last_recall = []

    def add(self, text, metadata):
        with self.transport.role('mem0.write'):
            response = self.memory.add([dict(role='user', content=text)],
                user_id='pmbench-participant', metadata=deepcopy(metadata), infer=True)
        append(self.out / 'mem0_writes.jsonl', dict(text=text, metadata=metadata, response=response))

    def recall(self, query, context):
        with self.transport.role('mem0.search'):
            response = self.memory.search(query=query, filters={'user_id': 'pmbench-participant'},
                top_k=TOP_K, threshold=0.0)
        self.last_recall = response.get('results', [])
        return common.truncate('\n'.join('- ' + common.scrub_task_handles(r['memory'])
            for r in self.last_recall), MAX_INJECT)

    def snapshot(self):
        result = self.memory.get_all(filters={'user_id': 'pmbench-participant'}, top_k=100000)
        return sorted(result['results'], key=lambda r: r['id'])

    def close(self):
        self.memory.vector_store.client.close()
        if self.memory._entity_store is not None:
            self.memory._entity_store.client.close()
        self.memory.db.connection.close()


class PmaMemory:
    def __init__(self, transport, out, tape, stack):
        sys.path.insert(0, str(PMA_ROOT / 'src'))
        from memory_agent.memory.memory_agent import MemoryAgent
        import memory_agent.memory.universal_memory as bank
        import shortuuid
        stack.enter_context(patch.object(bank, 'shortuuid', SimpleNamespace(
            uuid=lambda: tape.get('pma.shortuuid', shortuuid.uuid))))
        stack.enter_context(patch.object(bank, 'datetime', tape.datetime_class('pma.bank')))
        class LLM:
            async def call(inner, prompt, system, tools=None):
                stage = 'pma.manage' if tools else 'pma.intervene'
                request = dict(model=MODEL, temperature=0, max_tokens=4096,
                    messages=[dict(role='system', content=system), dict(role='user', content=prompt)])
                if tools:
                    request.update(tools=tools, tool_choice='auto')
                with transport.role(stage):
                    return transport.create(**request).choices[0].message
        self.agent = MemoryAgent(llm=LLM())
        self.history, self.last_recall = [], {}
        self.calls = 0
        self.out = out

    def add(self, text, metadata):
        self.history.append(dict(text=text, metadata=deepcopy(metadata)))

    def recall(self, query, context):
        self.calls += 1
        # Original two-phase prompts/tools/bank and default BM25 prefilter.
        observation = ('[Task Description]\n' + context['task_description'] +
            '\n\n[Recent Trajectory]\n' + '\n\n'.join(x['text'] for x in self.history[-7:]) +
            '\n\n[Current Public Observation]\n' + context['current'])
        result = asyncio.run(self.agent.process(observation=observation, step_count=self.calls))
        self.last_recall = asdict(result)
        append(self.out / 'pma_events.jsonl', self.last_recall)
        return common.truncate(common.scrub_task_handles(result.context_for_action or ''), MAX_INJECT)

    def snapshot(self):
        return self.agent.memory.to_dict()

    def close(self):
        pass


class MemoryPolicy:
    """Common action context, read query, public ingestion, and action mapping."""
    def __init__(self, method, transport, budget, out, cache, stack):
        self.transport, self.budget, self.out = transport, budget, out
        self.tape = NativeTape(cache / 'native_tape.jsonl', transport.offline)
        self.memory = (RagMemory(transport, out) if method == 'rag' else
            Mem0Memory(transport, out, self.tape, stack) if method == 'mem0' else
            PmaMemory(transport, out, self.tape, stack))
        self.step, self.step_start = None, 0
        self.seen_plans = set()
        self.day = ''
        self.original_request = None

    def __call__(self, client, model, messages, ids, channels, backend, **kwargs):
        from run_memory_baseline import pm
        key = self.budget.step
        if key != self.step:
            self.step, self.step_start = key, len(messages) - 1
        plans = []
        for message in messages[:self.step_start]:
            text = message.get('content', '')
            if message.get('role') != 'user' or not isinstance(text, str):
                continue
            if text.startswith('=== ') or 'Regular tasks for every day' in text:
                if text.startswith('=== '):
                    self.day = text.splitlines()[0].strip('= ')
                if text not in self.seen_plans:
                    self.memory.add(text, dict(kind='public_plan', day=self.day))
                    append(self.out / 'ingestion.jsonl', dict(text=text, kind='public_plan', day=self.day))
                    self.seen_plans.add(text)
                plans.append(text)
        current_messages = messages[self.step_start:]
        current = visible_note(current_messages)
        query = f'Day: {self.day}\n{current}'
        systems = [deepcopy(m) for m in messages if m.get('role') == 'system']
        standing = next((p for p in plans if 'Regular tasks for every day' in p), None)
        day_plan = next((p for p in reversed(plans) if p.startswith('=== ')), None)
        head = systems + [dict(role='user', content=p) for p in (standing, day_plan) if p]
        compact = head + deepcopy(current_messages)
        task_description = '\n'.join(m['content'] for m in head if isinstance(m.get('content'), str))
        memory_text = self.memory.recall(query, dict(task_description=task_description, current=current))
        append(self.out / 'memory.jsonl', dict(day=key[0], step_id=key[1], query=query,
            memory_text=memory_text, recall=self.memory.last_recall))
        compact.append(dict(role='user', content=(
            '[Relevant memories from previously observed information]\n' + (memory_text or '(none)') +
            '\nThese records may describe earlier conditions. Match actions by meaning to the current menu.')))
        policy = self
        class DecisionClient:
            def __init__(inner):
                inner.chat = SimpleNamespace(completions=SimpleNamespace(create=inner.create))
            def create(inner, **request):
                request = policy.budget.request(request)
                request.pop('extra_body', None)  # provider-specific transport fields only
                with policy.transport.role('decision'):
                    return policy.transport.create(**request)
        try:
            action = self.original_request(DecisionClient(), MODEL, compact, ids, channels,
                'openrouter', allow_heartbeat=False, max_tokens=1024)
        except pm.InvalidModelResponseError as exc:
            raise ExperimentStop('Invalid decision; evaluator label fallback prohibited') from exc
        append(self.out / 'decisions.jsonl', dict(day=key[0], step_id=key[1], action=action))
        if (action.get('action') == 'choose' and action.get('choice') in ('A', 'B', 'C')
                and action.get('channel') in (None, 'NONE')):
            # Describe submitted action; the benchmark gives no success receipt.
            labels = common.resolve_action_labels(dict(action=action, step_raw=messages[self.step_start]['content']))
            note = f'Day: {self.day}\n{current}\nSubmitted prospective actions: ' + ('; '.join(labels) or '(none)')
            metadata = dict(day=self.day, kind='completed_environment_step')
            self.memory.add(note, metadata)
            append(self.out / 'ingestion.jsonl', dict(text=note, **metadata))
        return action
