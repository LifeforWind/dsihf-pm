"""Controlled extension, separate from official PM-Bench data and scoring.

Structured task contracts are PUBLIC inputs. Signal schedules, service outcomes,
and the evaluator's required-task list stay private. No policy reads a scenario.
"""
from copy import deepcopy
import random

FAMILIES = ('multi_condition', 'cross_day', 'update_shared', 'retry_receipt', 'freshness', 'single_condition')
SEEDS = (3101, 3102)
STEPS = 12


def make_scenario(family, seed):
    rng = random.Random(seed)
    names = [f'source_{i}' for i in range(1, 8)]
    rng.shuffle(names)
    kinds = ('funds', 'approval', 'inventory', 'registry', 'receipt', 'guide', 'news')
    route = dict(zip(kinds, names))
    descriptions = dict(funds='Current account funding confirmations for individual orders.',
        approval='Signed invoice approval records for individual orders.',
        inventory='Current inventory reservations for individual orders.',
        registry='Current compliance clearance certificates for individual orders.',
        receipt='Execution receipts: exact order, task revision and attempt identity, success or failure.',
        guide='General payment and order-processing instructions; no live order statuses.',
        news='News and discussion of purchasing, invoices and payment; no individual order statuses.')
    catalog = {route[k]: dict(description=descriptions[k], kind=k,
        freshness='Immutable signed approval; valid through this episode.' if k == 'approval'
        else 'Snapshots certified for blocks 0-3, 4-7, 8-11; may change only at block boundaries.')
        for k in kinds}
    scenario = dict(id=f'{family}-{seed}', family=family, seed=seed, steps=STEPS,
        catalog=catalog, updates={}, signals={}, execution={}, required=[], cancellations=[])
    prefix = f'order-{seed}-'
    for n in range(3):
        key = prefix + str(n)
        selected = ['approval', 'funds', 'inventory']
        if family in ('cross_day', 'retry_receipt', 'freshness', 'update_shared'):
            selected = ['approval', 'funds']
        if family == 'single_condition': selected = ['funds']
        conditions = []
        for kind in selected:
            signal_key = f'{key}:{kind}'
            values = [True, True, True]
            if kind == 'funds' and family in ('multi_condition', 'cross_day', 'update_shared'):
                values = [False, True, True] if n != 2 else [False, False, True]
            if kind == 'inventory' and family == 'multi_condition': values = [False, False, True]
            if kind == 'funds' and family == 'freshness': values = [True, False, True]
            conditions.append(dict(id=kind, key=signal_key,
                description=f'{key} has a confirmed {kind} record authorizing processing.',
                operator='eq', expected=True, channels=[route[kind]], max_age=11 if kind == 'approval' else 3))
            scenario['signals'][signal_key] = dict(channel=route[kind], values=values,
                                                  immutable=kind == 'approval')
        active = 4 * n if family == 'cross_day' else 0
        quote = (f'Process {key} once all of the following are confirmed: ' + ', '.join(selected)
                 + f'. This commitment remains active from tick {active} through tick 11 across days. '
                 'A submitted attempt is not success; confirm its receipt and retry only after explicit failure.')
        task = dict(id=key, revision=1, action=f'Process payment and order {key}', source_quote=quote,
            conditions=conditions, active_from=active, expires_at=11, receipt_channels=[route['receipt']])
        scenario['updates'].setdefault('0', []).append(dict(op='add', task=task, text=quote))
        scenario['execution'][key] = dict(failures=1 if family in ('retry_receipt', 'freshness') else 0,
            receipt_delay=2 if family in ('retry_receipt', 'freshness') else 1)
        scenario['required'].append(key)
    if family == 'update_shared':
        canceled = prefix + '0'
        quote = f'Cancel the processing commitment for {canceled}.'
        scenario['updates']['2'] = [dict(op='cancel', id=canceled, revision=1, quote=quote, text=quote)]
        scenario['required'].remove(canceled); scenario['cancellations'].append(canceled)
        new = deepcopy(scenario['updates']['0'][1]['task']); new['revision'] = 2
        new['conditions'][0] = dict(id='clearance', key=f'{new["id"]}:registry',
            description=f'Current compliance clearance for {new["id"]}', operator='eq', expected=True,
            channels=[route['registry']], max_age=3)
        new['source_quote'] = f'Revise {new["id"]}: require registry clearance and funding, replacing invoice approval.'
        scenario['updates']['4'] = [dict(op='revise', task=new, text=new['source_quote'])]
        scenario['signals'][f'{new["id"]}:registry'] = dict(channel=route['registry'], values=[False, True, True], immutable=False)
    scenario['provider_anomalies'] = ({'1': [route['funds']], '5': [route['funds']]}
                                      if family == 'freshness' else {})
    return scenario


class Environment:
    def __init__(self, scenario):
        self.scene = deepcopy(scenario)
        self.tasks, self.status, self.pending = {}, {}, []
        self.actions, self.delivered, self.queries = [], [], []
        self.attempt_counts = {}
        self.completed = {}
        self.now = -1

    def begin(self, tick):
        self.now = tick
        updates = deepcopy(self.scene['updates'].get(str(tick), []))
        for update in updates:
            if update['op'] == 'cancel':
                self.status[update['id']] = 'canceled'
            else:
                task = update['task']; self.tasks[task['id']] = task
                self.status[task['id']] = 'pending'
        return dict(time=tick, day=('Monday' if tick < 4 else 'Tuesday' if tick < 8 else 'Wednesday'),
                    context='Working through payment, purchasing and order-processing commitments.',
                    catalog=deepcopy(self.scene['catalog']), updates=updates)

    def query(self, channel):
        if channel not in self.scene['catalog']:
            raise ValueError('Unknown source')
        facts = []
        if channel not in self.scene['provider_anomalies'].get(str(self.now), []):
            for key, signal in self.scene['signals'].items():
                if signal['channel'] != channel: continue
                block = self.now // 4
                facts.append(dict(key=key, value=signal['values'][block],
                    as_of=0 if signal['immutable'] else block * 4,
                    valid_until=11 if signal['immutable'] else block * 4 + 3))
        receipts = [deepcopy(x['receipt']) for x in self.pending
                    if x['available_at'] <= self.now and x['channel'] == channel]
        # Receipts are persistent snapshots, including older attempts. The policy
        # must match the current task version and attempt rather than latest text.
        self.queries.append(dict(time=self.now, channel=channel, facts=deepcopy(facts), receipts=deepcopy(receipts)))
        for r in receipts:
            if r['outcome'] == 'success':
                task = self.tasks.get(r['task_id'])
                if task and task['revision'] == r['revision'] and self.status[r['task_id']] != 'canceled':
                    self.delivered.append(dict(time=self.now, **r))
        return dict(facts=facts, receipts=receipts)

    def true_conditions(self, task):
        # Independent evaluator uses provider schedules, never policy's ledger.
        result = []
        for c in task['conditions']:
            signal = self.scene['signals'][c['key']]
            value = signal['values'][self.now // 4]
            op, expected = c['operator'], c['expected']
            result.append((type(value) == type(expected) and value == expected) if op == 'eq' else
                {'ge': value >= expected, 'le': value <= expected,
                 'gt': value > expected, 'lt': value < expected}[op])
        return all(result)

    def execute(self, actions):
        for action in actions:
            key = action['task_id']; task = self.tasks.get(key)
            valid = bool(task and self.status[key] != 'canceled'
                and action['revision'] == task['revision'] and task['active_from'] <= self.now <= task['expires_at']
                and self.true_conditions(task) and (key, task['revision']) not in self.completed)
            if any(p['receipt']['attempt_id'] == action['attempt_id'] for p in self.pending):
                raise ValueError('Duplicate attempt identity')
            version = (key, action['revision'])
            count = self.attempt_counts.get(version, 0) + 1; self.attempt_counts[version] = count
            config = self.scene['execution'].get(key, dict(failures=0, receipt_delay=1))
            success = valid and count > config['failures']
            if success:
                self.completed[version] = self.now
            receipt = dict(**action, outcome='success' if success else 'failed')
            self.pending.append(dict(available_at=self.now + config['receipt_delay'],
                channel=task['receipt_channels'][0], receipt=receipt))
            self.actions.append(dict(time=self.now, **action, valid=valid, success=success))

    def score(self, final_status):
        required = self.scene['required']
        completed = [k for k in required if (k, self.tasks[k]['revision']) in self.completed]
        confirmed = [k for k in completed if any(r['task_id'] == k and r['revision'] == self.tasks[k]['revision']
            and r['time'] <= self.tasks[k]['expires_at'] for r in self.delivered)]
        correct_closures = all(status != 'completed' or key in confirmed for key, status in final_status.items())
        return dict(required=len(required), completed=len(completed), confirmed=len(confirmed),
            completed_ids=completed, confirmed_ids=confirmed, queries=len(self.queries), attempts=len(self.actions),
            invalid_attempts=sum(not a['valid'] for a in self.actions),
            failed_attempts=sum(not a['success'] for a in self.actions),
            correct_closures=correct_closures, completion_rate=len(completed)/len(required),
            confirmed_rate=len(confirmed)/len(required))
