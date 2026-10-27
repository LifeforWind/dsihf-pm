#!/usr/bin/env python3
"""Evaluation-only diagnostics traced directly from the official scorer."""
import argparse
import collections
import inspect
import json
from pathlib import Path
import sys
from run_amem_small import pm, write_json


def trace_scores(scenario, entries):
    source, first_line = inspect.getsourcelines(pm.score_day)
    line_number = first_line + next(i for i, line in enumerate(source) if 'if chosen_set == due_now:' in line)
    records = []

    def trace(frame, event, arg):
        if frame.f_code is pm.score_day.__code__ and event == 'line' and frame.f_lineno == line_number:
            v = frame.f_locals
            due, chosen = v['due_now'], v['chosen_set']
            records.append({'day': v['day']['name'], 'step_id': v['step']['id'],
                            'time': v['step']['time'], 'text': v['step']['text'],
                            'due': sorted(due), 'chosen': sorted(chosen),
                            'tp': sorted(due & chosen), 'fp': sorted(chosen - due), 'fn': sorted(due - chosen),
                            'queries': v['query_counts'].copy(),
                            'tasks': {k: {'label': s['current']['label'], 'current': s['current'].copy(),
                                          'completed': s['completed'], 'canceled': s['canceled'],
                                          'required_channel': pm.required_monitor_channel(s['current'])}
                                      for k, s in v['task_states'].items() if k in due | chosen}})
        return trace

    previous = sys.gettrace()
    try:
        sys.settrace(trace)
        summary, per_day, count = pm.score_log(scenario, entries)
    finally:
        sys.settrace(previous)
    assert [sum(len(r[k]) for r in records) for k in ['tp', 'fp', 'fn']] == [summary['set_tp'], summary['set_fp'], summary['set_fn']]
    return records, summary, per_day, count


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run_root', type=Path)
    args = p.parse_args()
    combined = {}
    for mode in ['single', 'window', 'amem']:
        out = args.run_root / mode
        if not (out / 'score.json').exists():
            continue
        scenario = json.loads((out / 'scenario.json').read_text())
        records, summary, per_day, count = trace_scores(scenario, pm.read_log(str(out / 'actions.jsonl')))
        calls = [json.loads(line) for line in (out / 'decisions.jsonl').read_text().splitlines()]
        final_calls = {c['step_ordinal']: c for c in calls if c['action']['action'] == 'choose'}
        missed_channel = collections.Counter()
        missed_channel_unqueried = collections.Counter()
        for ordinal, record in enumerate(records, 1):
            decision = final_calls[ordinal]
            record.update(step_ordinal=ordinal, decision_call=decision['call'],
                          retrieved_ids=[n['id'] for n in decision['retrieved']],
                          retrieved=[{'id': n['id'], 'content': n['content'], 'context': n['context']} for n in decision['retrieved']])
            for task in record['fn']:
                channel = record['tasks'][task]['required_channel'] or 'visible_event'
                missed_channel[channel] += 1
                if channel != 'visible_event' and not record['queries'].get(channel):
                    missed_channel_unqueried[channel] += 1
        write_json(out / 'step_diagnostics.json', records)
        manifest = json.loads((out / 'manifest.json').read_text())
        memory_calls = [json.loads(line) for line in (out / 'memory_calls.jsonl').read_text().splitlines()] if (out / 'memory_calls.jsonl').exists() else []
        usage = {key: sum(c['usage'].get(key, 0) for c in calls + memory_calls) for key in ['prompt_tokens', 'completion_tokens', 'total_tokens']}
        tp, fp, fn = (summary['set_' + key] for key in ['tp', 'fp', 'fn'])
        item = {'set_f1': 2*tp/(2*tp+fp+fn), 'precision': tp/(tp+fp) if tp+fp else 0,
                'recall': tp/(tp+fn), 'summary': summary, 'missed_by_channel': dict(missed_channel),
                'missed_without_current_query': dict(missed_channel_unqueried),
                'decision_calls': len(calls), 'memory_calls': len(memory_calls), 'usage_all': usage,
                'seconds': manifest['seconds'], 'models_returned': sorted(set(c.get('model_returned', c.get('model')) for c in calls + memory_calls))}
        if memory_calls:
            final = json.loads((out / 'memory_final.json').read_text())
            notes = final['notes']
            stored = dict(zip(final['stored']['ids'], final['stored']['metadatas']))
            links = [(key, link) for key, note in notes.items() for link in note['links']]
            item['memory_audit'] = {'note_count': len(notes), 'evolutions': final['evolution_count'],
                                    'link_count': len(links), 'resolved_link_count': sum(link in notes for _, link in links),
                                    'stale_context_ids': [k for k, n in notes.items() if n['context'] != stored[k]['context']],
                                    'returned_linked_neighbors': sum(bool(n.get('is_neighbor')) for c in calls for n in c['retrieved'])}
        combined[mode] = item
    write_json(args.run_root / 'comparison.json', combined)
    print(json.dumps(combined, indent=2))


if __name__ == '__main__':
    main()
