"""Remaining evidence needs over explicit, versioned task contracts.

No benchmark labels, model APIs, or ranking weights live here. Callers translate
visible instructions into TaskSpec and deliver only actually observed facts and
execution receipts. Conjunctive conditions are the deliberately bounded v1 scope.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class Condition:
    id: str
    key: str
    description: str
    operator: str
    expected: object
    channels: tuple[str, ...]
    max_age: int = 0

    def __post_init__(self):
        if not self.id or not self.key or not self.description or not self.channels:
            raise ValueError('Conditions require identity, meaning, and evidence sources')
        if len(set(self.channels)) != len(self.channels) or any(not c for c in self.channels):
            raise ValueError('Invalid condition sources')
        if self.operator not in ('eq', 'ge', 'le', 'gt', 'lt'):
            raise ValueError('Unsupported predicate')
        if type(self.max_age) is not int or self.max_age < 0:
            raise ValueError('Invalid evidence age')
        if self.operator != 'eq' and not _number(self.expected):
            raise ValueError('Ordered predicates require finite numbers')


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


@dataclass(frozen=True)
class TaskSpec:
    id: str
    revision: int
    action: str
    source_quote: str
    conditions: tuple[Condition, ...]
    active_from: int
    expires_at: int
    receipt_channels: tuple[str, ...]

    def __post_init__(self):
        if not self.id or type(self.revision) is not int or self.revision < 1:
            raise ValueError('Invalid task version')
        if not self.action or not self.source_quote or not self.conditions or not self.receipt_channels:
            raise ValueError('Incomplete task contract')
        if type(self.active_from) is not int or type(self.expires_at) is not int or self.active_from > self.expires_at:
            raise ValueError('Invalid task window')
        if len({c.id for c in self.conditions}) != len(self.conditions):
            raise ValueError('Duplicate condition IDs')


class EvidenceNeeds:
    """Task lifecycle and evidence lifecycle are separate, auditable state.

    A failed/unknown predicate does not cancel a task. An attempt waits for a
    matching receipt; only observed success closes the task instance. Revision
    clears its attempt and version-bound evidence, never other tasks' evidence.
    """
    def __init__(self):
        self.tasks = {}
        self.status = {}
        self.attempts = {}
        self.facts = []
        self.observations = {}
        self.events = []
        self.now = -1

    def advance(self, now):
        if type(now) is not int or now < self.now:
            raise ValueError('Time must be monotonic')
        self.now = now

    def install(self, task, observation_id, text):
        if task.source_quote not in text:
            raise ValueError('Task requires an exact visible source quote')
        old = self.tasks.get(task.id)
        if old and task.revision <= old.revision:
            raise ValueError('Task revision must increase')
        self._observe(observation_id, dict(kind='instruction', text=text, at=self.now))
        self.tasks[task.id] = task
        self.status[task.id] = 'pending'
        self.attempts.pop(task.id, None)
        self.events.append(dict(type='revise' if old else 'add', task=task.id,
                                revision=task.revision, evidence=observation_id, at=self.now))

    def cancel(self, task_id, revision, observation_id, source_quote, text):
        if not source_quote or source_quote not in text:
            raise ValueError('Cancellation requires an exact visible source')
        if self.tasks[task_id].revision != revision:
            raise ValueError('Cancellation version mismatch')
        self._observe(observation_id, dict(kind='instruction', text=text, at=self.now))
        self.status[task_id] = 'canceled'
        self.attempts.pop(task_id, None)
        self.events.append(dict(type='cancel', task=task_id, revision=revision,
                                evidence=observation_id, at=self.now))

    def _observe(self, identity, payload):
        if not identity:
            raise ValueError('Observation identity required')
        if identity in self.observations and self.observations[identity] != payload:
            raise ValueError('Observation IDs are immutable')
        self.observations[identity] = deepcopy(payload)

    def observe(self, observation_id, channel, facts=(), receipts=()):
        """Apply a provider response atomically; empty/unknown is not false.

        Each fact has key/value/as_of/valid_until. The provider validity is
        intersected with each condition's max_age, never inferred from delivery.
        """
        payload = dict(channel=channel, facts=list(facts), receipts=list(receipts), at=self.now)
        trial = deepcopy(self)
        trial._observe(observation_id, payload)
        if observation_id in self.observations:
            return
        for fact in facts:
            if set(fact) != {'key', 'value', 'as_of', 'valid_until'} or not fact['key']:
                raise ValueError('Malformed provider fact')
            if (type(fact['as_of']) is not int or type(fact['valid_until']) is not int
                    or fact['as_of'] > self.now or fact['valid_until'] < fact['as_of']):
                raise ValueError('Invalid evidence timestamps')
            # Stamp support for versions actually present when the fact arrives.
            versions = {k: t.revision for k, t in trial.tasks.items()}
            trial.facts.append(dict(**deepcopy(fact), channel=channel,
                                    evidence=observation_id, versions=versions))
        for receipt in receipts:
            trial._receipt(receipt, channel, observation_id)
        trial.events.append(dict(type='observation', evidence=observation_id, channel=channel, at=self.now))
        self.__dict__.update(trial.__dict__)

    def _receipt(self, receipt, channel, observation_id):
        if set(receipt) != {'task_id', 'revision', 'attempt_id', 'outcome'}:
            raise ValueError('Malformed receipt')
        if receipt['outcome'] not in ('success', 'failed', 'pending'):
            raise ValueError('Unknown receipt outcome')
        key = receipt['task_id']
        task, attempt = self.tasks.get(key), self.attempts.get(key)
        matches = (task and attempt and self.status[key] == 'inflight'
                   and channel in task.receipt_channels and receipt['revision'] == task.revision
                   and receipt['attempt_id'] == attempt['attempt_id'])
        if not matches:
            self.events.append(dict(type='ignored_receipt', receipt=deepcopy(receipt), evidence=observation_id))
            return
        if receipt['outcome'] == 'success':
            self.status[key] = 'completed'
        elif receipt['outcome'] == 'failed':
            self.status[key] = 'pending'
            del self.attempts[key]
        self.events.append(dict(type='receipt', task=key, outcome=receipt['outcome'],
                                evidence=observation_id, at=self.now))

    def active(self):
        return [t for t in self.tasks.values() if self.status[t.id] in ('pending', 'inflight')
                and t.active_from <= self.now <= t.expires_at]

    def condition_state(self, task, condition):
        rows = [f for f in self.facts if f['key'] == condition.key
                and f['channel'] in condition.channels and f['versions'].get(task.id) == task.revision
                and f['as_of'] <= self.now]
        if not rows:
            return dict(value='unknown', evidence=[])
        newest = max(f['as_of'] for f in rows)
        rows = [f for f in rows if f['as_of'] == newest]
        # A newer expired/revoked value supersedes an older long-lived one.
        if any(self.now > min(f['valid_until'], f['as_of'] + condition.max_age) for f in rows):
            return dict(value='unknown', evidence=[f['evidence'] for f in rows])
        values = [f['value'] for f in rows]
        if any(type(v) != type(values[0]) or v != values[0] for v in values) or values[0] is None:
            return dict(value='unknown', evidence=[f['evidence'] for f in rows])
        value = values[0]
        if condition.operator == 'eq':
            result = type(value) == type(condition.expected) and value == condition.expected
        elif not _number(value):
            return dict(value='unknown', evidence=[f['evidence'] for f in rows])
        else:
            target = condition.expected
            result = {'ge': value >= target, 'le': value <= target,
                      'gt': value > target, 'lt': value < target}[condition.operator]
        return dict(value='true' if result else 'false', evidence=[f['evidence'] for f in rows])

    def states(self, task):
        return {c.id: self.condition_state(task, c) for c in task.conditions}

    def ready(self, task):
        return self.status[task.id] == 'pending' and task in self.active() and all(
            row['value'] == 'true' for row in self.states(task).values())

    def attempted(self, task_id, attempt_id):
        task = self.tasks[task_id]
        if not attempt_id or not self.ready(task):
            raise ValueError('Attempt requires active task and currently supported conditions')
        self.attempts[task_id] = dict(attempt_id=attempt_id, revision=task.revision, at=self.now)
        self.status[task_id] = 'inflight'
        self.events.append(dict(type='attempt', task=task_id, attempt=attempt_id, at=self.now))

    def supports(self, mode='remaining'):
        """channel -> distinct task/condition reasons. Ranking is a separate layer."""
        if mode not in ('remaining', 'task_channels'):
            raise ValueError('Unknown candidate mode')
        result = {}
        def add(channel, task, reason):
            result.setdefault(channel, []).append(dict(task_id=task.id, revision=task.revision, reason=reason))
        for task in self.active():
            states = self.states(task)
            if mode == 'task_channels':
                for condition in task.conditions:
                    for channel in condition.channels:
                        add(channel, task, condition.id)
                if self.status[task.id] == 'inflight':
                    for channel in task.receipt_channels:
                        add(channel, task, 'receipt')
            elif self.status[task.id] == 'inflight':
                for channel in task.receipt_channels:
                    add(channel, task, 'receipt')
            elif not any(row['value'] == 'false' for row in states.values()):
                for condition in task.conditions:
                    if states[condition.id]['value'] == 'unknown':
                        for channel in condition.channels:
                            add(channel, task, condition.id)
        return result

    def dump(self):
        return dict(now=self.now, tasks={k: asdict(t) for k, t in self.tasks.items()},
                    status=deepcopy(self.status), attempts=deepcopy(self.attempts),
                    facts=deepcopy(self.facts), observations=deepcopy(self.observations),
                    events=deepcopy(self.events))
