"""Visible-only adapter: common task/evidence/execution, three candidate policies.

The controlled extension supplies structured public task contracts and provider
facts. It deliberately does not claim natural-language extraction performance.
"""
from dataclasses import asdict
import importlib.util
from pathlib import Path
import sys

from run_recommendation import ObservationCandidate, ObservationScheduler

SOURCE = Path(__file__).resolve().parents[3] / 'a-mem/agentic_memory/evidence_needs.py'
spec = importlib.util.spec_from_file_location('evidence_needs', SOURCE)
needs = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = needs
spec.loader.exec_module(needs)
Condition, TaskSpec, EvidenceNeeds = needs.Condition, needs.TaskSpec, needs.EvidenceNeeds
ARMS = ('semantic', 'task_channels', 'remaining')


def task_from_public(row):
    fields = dict(row)
    fields['conditions'] = tuple(Condition(**(dict(c) | {'channels': tuple(c['channels'])})) for c in row['conditions'])
    fields['receipt_channels'] = tuple(row['receipt_channels'])
    return TaskSpec(**fields)


class ResidualPolicy:
    def __init__(self, arm, models=None, budget=3):
        if arm not in ARMS:
            raise ValueError('Unknown arm')
        self.arm, self.models = arm, models
        self.store = EvidenceNeeds()
        self.scheduler = ObservationScheduler('round_robin', budget)
        self.serial = 0
        self.trace = []

    def begin(self, public):
        if set(public) != {'time', 'day', 'context', 'catalog', 'updates'}:
            raise ValueError('Policy interface accepts only explicit public fields')
        self.public = public
        self.store.advance(public['time'])
        self.observed = set()
        for i, update in enumerate(public['updates']):
            identity = f'instruction:{public["time"]}:{i}'
            if update['op'] == 'cancel':
                self.store.cancel(update['id'], update['revision'], identity, update['quote'], update['text'])
            else:
                self.store.install(task_from_public(update['task']), identity, update['text'])

    def receive(self, channel, response):
        if channel in self.observed or channel not in self.public['catalog']:
            raise ValueError('Only one requested response per channel and step')
        if set(response) != {'facts', 'receipts'}:
            raise ValueError('Unexpected provider response fields')
        self.store.observe(f'query:{self.store.now}:{channel}', channel, **response)
        self.observed.add(channel)

    def next_query(self):
        support = self.store.supports('remaining' if self.arm == 'remaining' else 'task_channels')
        semantic = {}
        if self.arm == 'semantic' and self.store.active() and len(self.observed) < self.scheduler.budget:
            # All arms have the same contracts, condition bindings, status and
            # observed evidence. This baseline receives them, including receipts.
            state = dict(context=self.public['context'], time=self.store.now,
                tasks=[dict(**asdict(t), status=self.store.status[t.id],
                            evidence_state=self.store.states(t)) for t in self.store.active()],
                catalog=self.public['catalog'], observed=sorted(self.observed),
                observed_facts=self.store.facts, attempts=self.store.attempts)
            channels = [c for c in self.public['catalog'] if c not in self.observed]
            questions = {f'relevant_{i}': dict(type='noul', instructions=
                f'Is the information source {c!r} semantically relevant to the active tasks and '
                'current context? Consider task goals, conditions, current progress and observed '
                'evidence. A source addressing an active task or its execution result is relevant. '
                'Use the catalog description and the full visible state. Do not invent hidden facts.')
                for i, c in enumerate(channels)}
            answers = self.models.evaluate('residual_semantic_relevance_v1', state, questions)
            semantic = {c: answers[f'relevant_{i}']['noul'] for i, c in enumerate(channels)}
            support = {c: [dict(task_id=t.id, revision=t.revision, reason='semantic_relevance')
                           for t in self.store.active()] for c, p in semantic.items() if p >= .5}
        candidates = [ObservationCandidate(c, tuple(sorted({r['task_id'] for r in rows})))
                      for c, rows in support.items()]
        decision = self.scheduler.rank(candidates, self.observed, self.store.now,
                                       self.public['day'], False)
        row = dict(time=self.store.now, arm=self.arm, support=support,
            observed=sorted(self.observed), semantic=semantic, decision=decision,
            condition_states={t.id: self.store.states(t) for t in self.store.active()},
            status=dict(self.store.status))
        self.trace.append(row)
        return decision['selected_channel']

    def act(self):
        actions = []
        for task in self.store.active():
            if self.store.ready(task):
                self.serial += 1
                attempt_id = f'attempt:{self.serial}:{task.id}:v{task.revision}'
                self.store.attempted(task.id, attempt_id)
                actions.append(dict(task_id=task.id, revision=task.revision, attempt_id=attempt_id))
        return actions
