"""Matched candidate baselines and optional episode cap for held-out evaluation.

No new ranking signal: every arm retains the same round-robin scheduler.
The cap can stop acquisition, but never makes a condition true or marks success.
"""
from dataclasses import asdict
import run_recommendation as base
from run_ren_grounded import GroundedRENPolicy


class CappedRoundRobin(base.AblationScheduler):
    def __init__(self,budget,episode_cap=None):
        super().__init__('round_robin',budget,'none')
        if episode_cap is not None and (isinstance(episode_cap,bool) or not isinstance(episode_cap,int) or episode_cap<1):
            raise ValueError('Episode cap must be a positive integer')
        self.episode_cap=episode_cap;self.spent=0

    def rank(self,*args,**kwargs):
        result=super().rank(*args,**kwargs)
        capped=self.episode_cap is not None and self.spent>=self.episode_cap
        if capped:result['selected_channel']=None
        result.update(episode_cap=self.episode_cap,episode_spent_before=self.spent,episode_exhausted=capped)
        if result['selected_channel'] is not None:self.spent+=1
        return result


class ConfirmationPolicy(GroundedRENPolicy):
    def __init__(self,*args,candidate_mode='remaining',episode_cap=None,**kwargs):
        self.semantic=candidate_mode=='semantic'
        super().__init__(*args,candidate_mode='remaining' if self.semantic else candidate_mode,**kwargs)
        self.scheduler=CappedRoundRobin(self.scheduler.budget,episode_cap)

    def next_query(self,active,observations,allowed_channels):
        if not self.semantic:return super().next_query(active,observations,allowed_channels)
        self.interpret(active,observations)
        live={f'{i.id}@{self.day}' for i in active}
        tasks=[t for t in self.needs.active() if t.id in live]
        channels=[c for c in allowed_channels if c not in observations]
        state=dict(day=self.day,visible_scene=self.vignette,observed_channels=observations,
            tasks=[dict(**asdict(t),status=self.needs.status[t.id],evidence_state=self.needs.states(t)) for t in tasks],
            channel_catalog={c:c.replace('_',' ') for c in allowed_channels})
        available=self.scheduler.episode_cap is None or self.scheduler.spent<self.scheduler.episode_cap
        questions={f'relevant_{n}':dict(type='noul',instructions=
            f'Is source {c!r} semantically relevant to the active tasks and current context? '
            'Consider task goals, conditions, current progress and observed evidence. A source '
            'addressing an active task or its execution result is relevant. Use the full visible '
            'state and public channel names; do not invent hidden facts.') for n,c in enumerate(channels)} if (
                tasks and len(observations)<self.scheduler.budget and available) else {}
        answers=self.models.evaluate('native_semantic_relevance_v1',state,questions)
        support={c:[dict(task_id=t.id,revision=t.revision,reason='semantic_relevance') for t in tasks]
            for n,c in enumerate(channels) if f'relevant_{n}' in answers and answers[f'relevant_{n}']['noul']>=.5}
        clock_required=any(i.kind=='clock' for i in active)
        # The existing shared adapter reserves the clock for pending clock tasks.
        # Keep that interface convention identical in all candidate arms.
        if clock_required and 'clock' in allowed_channels:
            support['clock']=[dict(task_id=f'{i.id}@{self.day}',revision=i.revision,reason='shared_clock_reservation')
                              for i in active if i.kind=='clock']
        candidates=[base.ObservationCandidate(c,tuple(sorted({r['task_id'] for r in reasons})))
                    for c,reasons in support.items()]
        decision=self.scheduler.rank(candidates,observations,self.step,self.day,clock_required)
        self.prefix+=1
        base.append(self.out/'recommendations.jsonl',dict(step=self.step,day=self.day,prefix=self.prefix,
            candidate_mode='semantic',support=support,unavailable_sources={},
            states={t.id:self.needs.states(t) for t in self.needs.active()},**decision))
        base.append(self.out/'semantic_candidates.jsonl',dict(step=self.step,state=state,questions=questions,answers=answers))
        return decision['selected_channel']
