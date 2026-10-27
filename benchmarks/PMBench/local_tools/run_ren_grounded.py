#!/usr/bin/env python3
"""Development adapter: evidence support/refutation and a shared condition ledger.

The intention compiler, native demand lifecycle and round-robin are unchanged.
No benchmark task labels, private state, future text or scorer feedback enter it.
"""
from copy import deepcopy
from functools import partial
from pathlib import Path
import sys

import run_recommendation as base
from run_ren_natural import NaturalModels, NaturalRENPolicy, SCENE, NEEDS_SOURCE
from run_intentions import deterministic_trigger


def evidence_questions(state):
    questions = {}
    for n, row in enumerate(state['evidence_items']):
        if row.get('kind', 'event') != 'event':
            continue
        reference = f'evidence_items[{n}]'
        boundary = (f'Use ONLY {reference}.observed_text as evidence for its condition; '
            'visible_scene supplies current temporal context. The intention is a plan, not evidence. '
            'Match the subject and owner as well as object, occurrence and temporal qualifications. '
            'An event concerning a different person or their property is not support for this condition. '
            'A reminder, future plan, '
            'hypothetical, unrelated event or old notification alone does not establish a current event. ')
        questions[f'positive_{n}'] = dict(type='noul', instructions=boundary +
            'Is this event condition occurring or satisfied now? Interpret ordinary narrative '
            'paraphrases and indirect physical availability by their everyday meaning. The text '
            'need not repeat the condition literally or explicitly announce an event timestamp. '
            'Present-tense scene narration that frames an ongoing event establishes that occurrence '
            'even when it reflects on the event rather than describing each physical action. '
            'Require positive support from the indicated source, not merely lack of contradiction.')
        questions[f'negative_{n}'] = dict(type='noul', instructions=boundary +
            'Does the evidence positively establish that this condition is NOT satisfied now? '
            'Require an explicit denial or an incompatible observed state. Missing mention, an '
            'empty/no-update response and uncertainty are NOT refutation.')
    return questions


def evidence_value(answers, n, threshold=.5):
    positive = answers[f'positive_{n}']['noul'] >= threshold
    negative = answers[f'negative_{n}']['noul'] >= threshold
    return None if positive == negative else positive


class GroundedModels(NaturalModels):
    """Only verified responses enter caches; retries never select a better answer."""
    def chat(self, *args, **kwargs):
        for attempt in range(3):
            try:
                return super().chat(*args, **kwargs)
            except RuntimeError as exc:
                # Identity errors are logged by the parent BEFORE rejection.
                if str(exc) not in ('Compiler returned a different model version',
                                    'Compiler model mismatch') and not str(exc).startswith('Unexpected compiler/judge model'):
                    raise
                base.append(self.out/'rejected_models.jsonl', dict(
                    stage=args[0], attempt=attempt+1, error=str(exc)))
                if attempt == 2:
                    raise

    def evaluate(self, stage, state, questions):
        if stage != 'trigger_and_bind' or not getattr(self, 'ledger_ready', None):
            return super().evaluate(stage, state, questions)
        binding = {k:v for k,v in questions.items() if not k.startswith('cue_')}
        answers = super().evaluate('bind_grounded_actions', state, binding)
        reused = {}
        for key in questions:
            if key.startswith('cue_'):
                intent = state['intentions'][int(key.split('_')[1])]
                ready, provenance = self.ledger_ready(intent['id'])
                if not ready:
                    raise RuntimeError('Attempt to bind an unverified condition')
                answers[key] = dict(noul=1., provenance='reused verified demand ledger')
                reused[key] = provenance
        base.append(self.out/'ledger_decisions.jsonl', dict(state=state, reused=reused))
        return answers


class GroundedRENPolicy(NaturalRENPolicy):
    def __init__(self,*args,candidate_mode='remaining',shared_decision=True,**kwargs):
        self.filtered_control = candidate_mode == 'task_channels_filtered'
        super().__init__(*args,candidate_mode='task_channels' if self.filtered_control else candidate_mode,**kwargs)
        if shared_decision:
            self.models.ledger_ready = self.ledger_ready

    def ledger_ready(self, identity):
        task = self.needs.tasks[f'{identity}@{self.day}']
        return self.needs.ready(task), dict(task_id=task.id, revision=task.revision,
            now=self.needs.now, states=self.needs.states(task))

    def interpret(self, active, observations):
        sources = {SCENE:self.vignette, **observations}
        pending = []
        for item in active:
            task = self.needs.tasks[f'{item.id}@{self.day}']
            for condition in task.conditions:
                for channel in condition.channels:
                    token = (self.step, task.id, task.revision, condition.id, channel)
                    if channel in sources and token not in self.seen_evidence:
                        pending.append((token,item,condition,channel))
        state = dict(today=self.day,visible_scene=self.vignette,evidence_items=[dict(
            intention=i.action,condition=c.description,kind=i.kind,source=channel,
            observed_text=sources[channel]) for _,i,c,channel in pending])
        questions = evidence_questions(state)
        answers = self.models.evaluate('interpret_grounded_evidence',state,questions)
        for n,(token,item,condition,channel) in enumerate(pending):
            if item.kind == 'event':
                value = evidence_value(answers,n,self.threshold)
            else:
                value = deterministic_trigger(item,observations)
                if not observations.get(channel) or 'no new' in observations[channel].lower():
                    value = None
            self.needs.observe('judgment:'+base.digest(token),channel,[dict(key=condition.key,
                value=value,as_of=self.needs.now,valid_until=self.needs.now)])
            self.seen_evidence.add(token)
        if pending:
            base.append(self.out/'condition_evidence.jsonl',dict(step=self.step,state=state,
                questions=questions,answers=answers,values=[evidence_value(answers,n,self.threshold)
                    if item.kind=='event' else deterministic_trigger(item,observations)
                    for n,(_,item,_,_) in enumerate(pending)],
                provenance='fallible current-step judgment, not provider validity'))

    def next_query(self, active, observations, allowed_channels):
        if not self.filtered_control:
            return super().next_query(active,observations,allowed_channels)
        self.interpret(active,observations)
        active_ids = {f'{i.id}@{self.day}' for i in active}
        support = {c:[r for r in reasons if r['task_id'] in active_ids]
                   for c,reasons in self.needs.supports('task_channels').items()}
        support = {c:r for c,r in support.items() if r}
        candidates = [base.ObservationCandidate(c,tuple(sorted({r['task_id'] for r in reasons})))
                      for c,reasons in support.items() if c in allowed_channels]
        decision = self.scheduler.rank(candidates,observations,self.step,self.day,
                                      any(i.kind=='clock' for i in active))
        self.prefix += 1
        base.append(self.out/'recommendations.jsonl',dict(step=self.step,day=self.day,prefix=self.prefix,
            candidate_mode='task_channels_filtered',support=support,
            unavailable_sources={c:r for c,r in support.items() if c not in allowed_channels},
            states={t.id:self.needs.states(t) for t in self.needs.active()},**decision))
        return decision['selected_channel']


def main():
    argv = sys.argv[1:]
    index = argv.index('--candidate-mode'); mode = argv[index+1]; del argv[index:index+2]
    shared = '--separate-decision' not in argv
    if not shared: argv.remove('--separate-decision')
    args = base.parse_args(argv)
    base.main(args=args, policy_factory=partial(GroundedRENPolicy,candidate_mode=mode,shared_decision=shared),
        model_factory=GroundedModels, extra_sources=(Path(__file__),Path(__file__).with_name('run_ren_natural.py'),
        NEEDS_SOURCE,Path(__file__).with_name('residual_policy.py')),
        extra_manifest=dict(ren_adapter='grounded-v1',candidate_mode=mode,shared_condition_decision=shared,
            natural_language_entry=True,structured_oracle_contracts=False,
            model_identity_retries=2,evidence_validity='current public step only; inferred by adapter',
            receipt_interface_available=False,completion_feedback_exercised=False))


if __name__=='__main__': main()
