"""Independent evidence/lifecycle reference, extracted unchanged for single-run audits."""
from copy import deepcopy

class Reference:
    def __init__(self):
        self.tasks={};self.status={};self.attempts={};self.facts=[];self.instructed={};self.now=-1;self.day=None

    def begin(self,public):
        self.now,self.day=public['time'],public['day']
        for u in public['updates']:
            if u['op']=='cancel':
                key=u['id'];self.status[key]='canceled';self.attempts.pop(key,None)
            else:
                t=deepcopy(u['task']);key=t['id'];self.tasks[key]=t
                self.status[key]='pending';self.attempts.pop(key,None)
            self.instructed[key]=self.day

    def receive(self,channel,response,omit_success=False):
        for f in response['facts']:
            self.facts.append(dict(**f,channel=channel,evidence=f'query:{self.now}:{channel}',
                                  versions={k:t['revision'] for k,t in self.tasks.items()}))
        for r in response['receipts']:
            key=r['task_id'];t=self.tasks.get(key)
            if omit_success and r['outcome']=='success':continue
            if not(t and self.status[key]=='inflight' and self.attempts.get(key)==r['attempt_id']
                   and t['revision']==r['revision'] and channel in t['receipt_channels']):continue
            if r['outcome']=='success':self.status[key]='completed'
            elif r['outcome']=='failed':self.status[key]='pending';del self.attempts[key]

    def active(self):
        return [t for t in self.tasks.values() if self.status[t['id']] in ('pending','inflight')
                and t['active_from']<=self.now<=t['expires_at']]

    def states(self,t):
        result={}
        for c in t['conditions']:
            evidence=[f for f in self.facts if f['key']==c['key'] and f['channel'] in c['channels']
                      and f['versions'].get(t['id'])==t['revision'] and f['as_of']<=self.now]
            state='unknown'
            if evidence:
                newest=max(f['as_of'] for f in evidence);evidence=[f for f in evidence if f['as_of']==newest]
                value=evidence[0]['value']
                valid=all(self.now<=f['valid_until'] and self.now-f['as_of']<=c['max_age']
                          and type(value)==type(f['value']) and value==f['value'] for f in evidence)
                if valid and value is not None:
                    assert c['operator']=='eq'
                    state='true' if type(value)==type(c['expected']) and value==c['expected'] else 'false'
            result[c['id']]=dict(value=state,evidence=[f['evidence'] for f in evidence])
        return result

    def support(self,mode='remaining',persistence=True):
        result={}
        def add(channel,t,reason):result.setdefault(channel,[]).append(dict(task_id=t['id'],revision=t['revision'],reason=reason))
        for t in self.active():
            if not persistence and self.instructed[t['id']]!=self.day:continue
            state=self.states(t)
            if mode=='task_channels':
                for c in t['conditions']:
                    for source in c['channels']:add(source,t,c['id'])
                if self.status[t['id']]=='inflight':
                    for source in t['receipt_channels']:add(source,t,'receipt')
            elif self.status[t['id']]=='inflight':
                for source in t['receipt_channels']:add(source,t,'receipt')
            elif not any(r['value']=='false' for r in state.values()):
                for c in t['conditions']:
                    if state[c['id']]['value']=='unknown':
                        for source in c['channels']:add(source,t,c['id'])
        return result

    def ready(self):
        return {t['id'] for t in self.active() if self.status[t['id']]=='pending'
                and all(r['value']=='true' for r in self.states(t).values())}

    def act(self,actions):
        for a in actions:self.status[a['task_id']]='inflight';self.attempts[a['task_id']]=a['attempt_id']
