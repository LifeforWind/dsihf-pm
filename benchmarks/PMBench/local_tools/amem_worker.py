#!/usr/bin/env python3
"""JSON-lines RPC in A-mem's own venv; only receives already-visible text."""
import argparse
from contextlib import redirect_stdout
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--amem-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--expected-model')
    parser.add_argument('--max-model-calls', type=int, default=800)
    parser.add_argument('--max-input-tokens', type=int, default=3000000)
    parser.add_argument('--restore-state', type=Path)
    parser.add_argument('--initial-calls', type=int, default=0)
    parser.add_argument('--initial-input-tokens', type=int, default=0)
    parser.add_argument('--model-identity-retries', type=int, default=0)
    args = parser.parse_args()
    sys.path.insert(0, str(args.amem_root))
    protocol = sys.stdout
    with redirect_stdout(sys.stderr):
        from agentic_memory.memory_system import AgenticMemorySystem
        imported = Path(inspect.getfile(AgenticMemorySystem)).resolve()
        if imported != (args.amem_root/'agentic_memory/memory_system.py').resolve():
            raise RuntimeError('Imported A-mem from an unexpected checkout')
        memory = AgenticMemorySystem(llm_model=os.environ['PMBENCH_MODEL'])
        restored = None
        if args.restore_state:
            from agentic_memory.memory_system import MemoryNote
            restored = json.loads(args.restore_state.read_text())
            memory.memories = {key: MemoryNote(**note) for key, note in restored['notes'].items()}
            memory.evo_cnt = restored['evolution_count']
            stored = restored['stored']
            # Preserve the stored metadata independently of evolved in-memory notes.
            # Old snapshots lack vectors; the unchanged local encoder rebuilds them.
            for index, key in enumerate(stored['ids']):
                kwargs = dict(ids=[key], documents=[stored['documents'][index]],
                              metadatas=[stored['metadatas'][index]])
                if stored.get('embeddings') is not None:
                    kwargs['embeddings'] = [stored['embeddings'][index]]
                memory.retriever.collection.add(**kwargs)
            if {key:vars(note) for key,note in memory.memories.items()} != restored['notes']:
                raise RuntimeError('Memory snapshot did not restore exactly')
            actual = memory.retriever.collection.get()
            for field in ['ids', 'documents', 'metadatas']:
                if actual[field] != stored[field]:
                    raise RuntimeError('Stored memory data did not restore exactly: '+field)
        loaded = {name: str(Path(module.__file__).resolve()) for name, module in sys.modules.items()
                  if name.startswith('agentic_memory') and getattr(module, '__file__', None)}
        runtime = {'loaded_modules': loaded, 'source_hashes': {
            name: hashlib.sha256(Path(path).read_bytes()).hexdigest() for name, path in loaded.items()},
            'embedding': memory.model_name, 'evo_threshold': memory.evo_threshold,
            'initial_memory_count': len(memory.memories)}
        if restored is not None:
            runtime.update(restored_from=str(args.restore_state),
                restored_sha256=hashlib.sha256(args.restore_state.read_bytes()).hexdigest(),
                restored_evolutions=memory.evo_cnt, snapshot_notes_and_metadata_equal=True,
                embeddings_recomputed=restored['stored'].get('embeddings') is None)
        (args.out/'memory_runtime.json').write_text(json.dumps(runtime, indent=2)+'\n')
        controller = memory.llm_controller.llm
        controller.client = controller.client.with_options(timeout=60, max_retries=1)
        create = controller.client.chat.completions.create
        if args.model_identity_retries:
            from model_identity import identity_checked
            create = identity_checked(create, args.expected_model, args.out, 'memory', args.model_identity_retries)
        counter = args.initial_calls
        input_tokens = args.initial_input_tokens
        failures = []

        def logged_create(**kwargs):
            nonlocal counter, input_tokens
            if args.expected_model and (counter >= args.max_model_calls or input_tokens >= args.max_input_tokens):
                failures.append('memory_budget')
                raise RuntimeError('Memory model protection budget exceeded')
            counter += 1
            started = time.monotonic()
            try:
                response = create(**kwargs)
            except Exception as exc:
                failures.append(type(exc).__name__)
                with (args.out / 'memory_errors.jsonl').open('a') as f:
                    f.write(json.dumps(dict(call=counter, error_type=type(exc).__name__,
                        status_code=getattr(exc, 'status_code', None)))+'\n')
                raise
            text = response.choices[0].message.content
            input_tokens += response.usage.prompt_tokens
            event = {'call': counter, 'request': kwargs, 'response': text,
                     'model': response.model, 'usage': response.usage.model_dump(),
                     'finish_reason': response.choices[0].finish_reason,
                     'seconds': time.monotonic() - started}
            with (args.out / 'memory_calls.jsonl').open('a') as f:
                f.write(json.dumps(event) + '\n')
            if args.expected_model and response.model != args.expected_model:
                failures.append('unexpected_model:' + response.model)
                raise RuntimeError('Unexpected memory model: ' + response.model)
            # Do not allow upstream's broad exception handlers to hide truncation.
            if response.choices[0].finish_reason != 'stop':
                failures.append('truncated')
                raise RuntimeError('Memory output did not finish normally')
            return response

        controller.client.chat.completions.create = logged_create
    print(json.dumps({'ready': True}), file=protocol, flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        try:
            with redirect_stdout(sys.stderr):
                op = request['op']
                if op == 'add':
                    before_calls = counter
                    content = request['content']
                    metadata = memory.analyze_content(content)
                    if not metadata.get('keywords') or metadata.get('context') == 'General':
                        raise RuntimeError('Native metadata analysis failed; refusing silent fallback')
                    key = f"note_{len(memory.memories):04d}"
                    before = {k: vars(v).copy() for k, v in memory.memories.items()}
                    key = memory.add_note(content, id=key, **metadata)
                    if failures:
                        raise RuntimeError('Native memory swallowed API failure: ' + ', '.join(failures))
                    note = memory.memories[key]
                    result = {'id': key, 'calls': counter - before_calls,
                              'evolution_count': memory.evo_cnt, 'note': vars(note),
                              'changed_neighbors': [k for k, v in before.items()
                                  if v != vars(memory.memories[k])]}
                    with (args.out / 'memory_writes.jsonl').open('a') as f:
                        f.write(json.dumps({'request': request, 'result': result}) + '\n')
                elif op == 'search':
                    result = memory.search_agentic(request['query'], request.get('k', 5))
                elif op == 'dump':
                    result = {'notes': {k: vars(v) for k, v in memory.memories.items()},
                              'stored': memory.retriever.collection.get(),
                              'evolution_count': memory.evo_cnt}
                else:
                    raise ValueError('Unknown operation')
            print(json.dumps({'ok': True, 'result': result}), file=protocol, flush=True)
        except Exception as exc:
            print(json.dumps({'ok': False, 'error': type(exc).__name__ + ': ' + str(exc)}),
                  file=protocol, flush=True)


if __name__ == '__main__':
    main()
