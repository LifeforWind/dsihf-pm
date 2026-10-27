"""Budgeted observation recommendation over evidence-backed intentions.

Semantic scores are supplied by the caller. This module knows no benchmark,
model API, hidden state, or action labels. Scores prioritize observation only;
they never authorize an intention's execution.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ObservationCandidate:
    channel: str
    intention_ids: tuple[str, ...]
    opportunity: float = 0.0

    def __post_init__(self):
        if not self.channel or not self.intention_ids:
            raise ValueError('A channel candidate needs at least one intention')
        if len(set(self.intention_ids)) != len(self.intention_ids):
            raise ValueError('Duplicate intentions inflate observation coverage')
        if (isinstance(self.opportunity, bool) or not isinstance(self.opportunity, (int, float))
                or not math.isfinite(self.opportunity) or not 0 <= self.opportunity <= 1):
            raise ValueError('Opportunity must be finite and in [0, 1]')


class ObservationScheduler:
    """A transparent, untrained ranker with a bounded stale-observation bonus.

    For ranked mode: max semantic opportunity per channel + 0.15 * age
    (capped at 6 steps) + 0.05 * extra covered intentions (capped at 4).
    Round-robin uses only last observed step. Both reserve one query for clock
    when the caller has pending exact-time intentions. Budget includes clock.
    """
    def __init__(self, mode='ranked', budget=3):
        if mode not in ('all', 'round_robin', 'ranked'):
            raise ValueError('Unknown observation policy')
        if budget is not None and (isinstance(budget, bool) or not isinstance(budget, int) or budget < 1):
            raise ValueError('Query budget must be positive or unlimited')
        if mode == 'all' and budget is not None:
            raise ValueError('All-channel policy requires an unlimited budget')
        self.mode, self.budget = mode, budget
        self.last_observed = {}
        self.scope = None
        self.scope_start_step = None
        self.last_step = None

    def rank(self, candidates, observed_channels, step, scope, clock_required=False):
        if self.last_step is not None and step < self.last_step:
            raise ValueError('Observation steps must be monotonic')
        if self.scope != scope:
            self.scope, self.scope_start_step = scope, step
            self.last_observed = {}
        self.last_step = step
        observed = set(observed_channels)
        for channel in observed:
            self.last_observed[channel] = step
        remaining = None if self.budget is None else max(0, self.budget - len(observed))
        if len({c.channel for c in candidates}) != len(candidates):
            raise ValueError('Group intentions by channel before scheduling')
        rows = []
        for candidate in candidates:
            if candidate.channel in observed:
                continue
            last = self.last_observed.get(candidate.channel, self.scope_start_step - 1)
            age = max(0, step - last)
            value = candidate.opportunity + .15 * min(age, 6) + .05 * min(len(candidate.intention_ids)-1, 4)
            rows.append(dict(channel=candidate.channel, intention_ids=list(candidate.intention_ids),
                             opportunity=candidate.opportunity, age_steps=age,
                             priority=value, clock_reserved=clock_required and candidate.channel == 'clock'))
        if self.mode == 'all':
            rows.sort(key=lambda r: r['channel'])
        elif self.mode == 'round_robin':
            rows.sort(key=lambda r: (not r['clock_reserved'], -r['age_steps'], r['channel']))
        else:
            rows.sort(key=lambda r: (not r['clock_reserved'], -r['priority'], r['channel']))
        selected = rows[0]['channel'] if rows and remaining != 0 else None
        return dict(selected_channel=selected, ranked_channels=rows,
                    budget=self.budget, remaining=remaining, observed_channels=sorted(observed))
