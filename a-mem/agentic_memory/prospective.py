"""Evidence-backed prospective memory, independent of vector retrieval and APIs.

The caller supplies only observed evidence and schema-validated proposals. A
semantic judge accepts proposals; this module owns revisions and execution state.
Original evidence is retained, including canceled and superseded instructions.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass, field
import re

DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']


def minutes(value):
    if not isinstance(value, str) or not re.fullmatch(r'(?:[01]\d|2[0-3]):[0-5]\d', value):
        raise ValueError('Expected HH:MM')
    hour, minute = map(int, value.split(':'))
    return hour * 60 + minute


@dataclass
class Intention:
    id: str
    action: str
    trigger: str
    day: str
    kind: str
    at: str | None
    channel: str | None
    comparator: str | None
    threshold: float | None
    parent_id: str | None
    dependency: str | None
    offset_minutes: int | None
    revision: int = 1
    status: str = 'pending'
    sources: list = field(default_factory=list)
    versions: list = field(default_factory=list)
    attempts: dict = field(default_factory=dict)
    confirmations: dict = field(default_factory=dict)
    blocked_by: list = field(default_factory=list)

    def spec(self):
        return {key: deepcopy(getattr(self, key)) for key in SPEC_FIELDS}


SPEC_FIELDS = ('action', 'trigger', 'day', 'kind', 'at', 'channel', 'comparator',
               'threshold', 'parent_id', 'dependency', 'offset_minutes')


class IntentionStore:
    def __init__(self):
        self.intents = {}
        self.evidence = {}
        self.events = []

    def observe(self, evidence_id, text, day, step):
        value = dict(text=text, day=day, step=step)
        if evidence_id in self.evidence and self.evidence[evidence_id] != value:
            raise ValueError('Evidence IDs are immutable')
        self.evidence[evidence_id] = value

    def validate(self, proposal, channels):
        if proposal['op'] not in ('add', 'revise', 'cancel'):
            raise ValueError('Unknown proposal operation')
        spec = proposal['spec']
        if set(spec) != set(SPEC_FIELDS):
            raise ValueError('Incomplete intention specification')
        if spec['day'] not in DAYS + ['EVERYDAY']:
            raise ValueError('Unknown intention day')
        if spec['kind'] not in ('event', 'clock', 'numeric'):
            raise ValueError('Unknown trigger kind')
        if not spec['action'].strip() or not spec['trigger'].strip():
            raise ValueError('Empty action or trigger')
        if spec['at'] is not None:
            minutes(spec['at'])
        if spec['kind'] == 'clock' and spec['at'] is None:
            raise ValueError('Clock trigger requires a time')
        if spec['channel'] is not None and spec['channel'] not in channels:
            raise ValueError('Unknown observation channel')
        if spec['kind'] == 'numeric' and (spec['channel'] is None or
                spec['comparator'] not in ('lt', 'le', 'eq', 'ge', 'gt') or
                not isinstance(spec['threshold'], (int, float))):
            raise ValueError('Numeric trigger requires a channel, operator and threshold')
        if spec['dependency'] not in (None, 'requires_confirmed', 'time_offset'):
            raise ValueError('Unknown dependency')
        if bool(spec['parent_id']) != bool(spec['dependency']):
            raise ValueError('Dependency must have a parent')
        if spec['dependency'] == 'time_offset' and not isinstance(spec['offset_minutes'], int):
            raise ValueError('Time dependency requires an integer offset')

    def apply(self, proposal, evidence_id, accepted=True):
        before = deepcopy((self.intents, self.events))
        try:
            self._apply(proposal, evidence_id, accepted)
        except Exception:
            self.intents, self.events = before
            raise

    def _apply(self, proposal, evidence_id, accepted):
        evidence = self.evidence[evidence_id]
        if not proposal['source_quote'] or proposal['source_quote'] not in evidence['text']:
            raise ValueError('Proposal lacks an exact observed source span')
        target, op = proposal['id'], proposal['op']
        self.events.append(dict(type='proposal', id=target, op=op, evidence=evidence_id, accepted=accepted))
        if not accepted:
            return
        if op == 'add':
            if target in self.intents:
                raise ValueError('Duplicate intention ID')
            item = Intention(id=target, **deepcopy(proposal['spec']))
            self.intents[target] = item
        else:
            item = self.intents.get(target)
            if item is None:
                self.events.append(dict(type='unapplied_missing_target', id=target, evidence=evidence_id))
                return
            item.versions.append(asdict(item) | {'versions': []})
            item.revision += 1
            if op == 'cancel':
                item.status = 'canceled'
            else:
                for key, value in proposal['spec'].items():
                    setattr(item, key, deepcopy(value))
                item.status = 'pending'
                # A revised commitment is eligible for a new attempt.
                item.attempts.clear()
                item.confirmations.clear()
        item.sources.append(dict(evidence_id=evidence_id, quote=proposal['source_quote']))
        self._validate_dependencies()
        self._propagate()

    def _validate_dependencies(self):
        for key in self.intents:
            seen, current = set(), key
            while current in self.intents:
                if current in seen:
                    raise ValueError('Cyclic intention dependency')
                seen.add(current)
                current = self.intents[current].parent_id

    def _propagate(self):
        # Bounded fixed point handles multiple dependency levels irrespective of order.
        for _ in range(len(self.intents)):
            changed = False
            for item in self.intents.values():
                parent = self.intents.get(item.parent_id)
                previous_blockers = item.blocked_by
                item.blocked_by = ([item.parent_id] if item.parent_id and
                    (parent is None or parent.status == 'canceled' or parent.blocked_by) else [])
                changed |= previous_blockers != item.blocked_by
                if not parent or item.blocked_by or item.status == 'canceled' or item.dependency != 'time_offset':
                    continue
                if parent.kind != 'clock':
                    item.blocked_by = [parent.id]
                    continue
                offset, minute = divmod(minutes(parent.at) + item.offset_minutes, 1440)
                day = 'EVERYDAY' if parent.day == 'EVERYDAY' else DAYS[(DAYS.index(parent.day) + offset) % 7]
                at = f'{minute // 60:02d}:{minute % 60:02d}'
                if (item.day, item.at, item.kind) != (day, at, 'clock'):
                    self.events.append(dict(type='dependency_update', id=item.id, parent_id=parent.id,
                                            parent_revision=parent.revision, before=item.spec(), day=day, at=at))
                    item.versions.append(asdict(item) | {'versions': []})
                    item.revision += 1
                    item.day, item.at, item.kind = day, at, 'clock'
                    item.attempts.clear()
                    item.confirmations.clear()
                    changed = True
            if not changed:
                break

    def active(self, day):
        items = []
        for item in self.intents.values():
            if item.day not in (day, 'EVERYDAY') or item.status != 'pending' or item.blocked_by:
                continue
            if day in item.attempts or day in item.confirmations:
                continue
            if item.dependency == 'requires_confirmed':
                parent = self.intents.get(item.parent_id)
                if not parent or not parent.confirmations or (parent.day == 'EVERYDAY' and day not in parent.confirmations):
                    continue
            items.append(item)
        return items

    def attempted(self, key, day, step, handle):
        item = self.intents[key]
        item.attempts[day] = dict(step=step, handle=handle, revision=item.revision)
        self.events.append(dict(type='attempt', id=key, day=day, step=step, revision=item.revision))

    def acknowledge(self, key, day, succeeded, evidence_id):
        if evidence_id not in self.evidence:
            raise ValueError('Execution feedback needs observed evidence')
        item = self.intents[key]
        if succeeded:
            item.confirmations[day] = evidence_id
        else:
            item.attempts.pop(day, None)
        self.events.append(dict(type='execution_feedback', id=key, succeeded=succeeded, evidence=evidence_id))

    def dump(self):
        return dict(intents={key: asdict(value) for key, value in self.intents.items()},
                    evidence=self.evidence, events=self.events)
