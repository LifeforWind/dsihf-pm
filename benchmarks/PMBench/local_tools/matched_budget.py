"""Public query budget overlay; official source files and scoring stay untouched."""
import ast
from copy import deepcopy
import inspect
from pathlib import Path
import textwrap
import types
from run_official_baselines import append, StudyStop


class QueryBudget:
    def __init__(self, out, step_cap=2, episode_cap=126):
        self.out, self.step_cap, self.episode_cap = Path(out), step_cap, episode_cap
        self.step = None
        self.used = self.total = 0
        self.history = []

    def begin(self, day, step_id):
        key = (day, step_id)
        if key != self.step:
            if key in self.history:
                raise StudyStop('Repeated budget step')
            self.step, self.used = key, 0
            self.history.append(key)
            append(self.out/'budget_events.jsonl', dict(event='step', day=day, step_id=step_id,
                episode_used=self.total))

    @property
    def remaining(self):
        return max(0, min(self.step_cap-self.used, self.episode_cap-self.total))

    def charge(self, channel):
        if self.step is None or not self.remaining:
            raise StudyStop('Query would exceed the frozen observation budget')
        self.used += 1
        self.total += 1
        append(self.out/'budget_events.jsonl', dict(event='query', day=self.step[0],
            step_id=self.step[1], channel=channel, step_used=self.used, episode_used=self.total))

    def limit_channels(self, channels):
        selected = list(channels)[:self.remaining]
        append(self.out/'budget_events.jsonl', dict(event='union_cap', day=self.step[0],
            step_id=self.step[1], proposed=list(channels), selected=selected,
            rule='original upstream proposal order, no new ranking'))
        return selected

    def request(self, kwargs):
        result = deepcopy(kwargs)
        notice = (f'Observation budget: at most {self.step_cap} channel queries per step and '
            f'{self.episode_cap} for the whole episode, including clock and repeated queries. '
            f'{self.remaining} queries remain available at this step; '
            f'{max(0,self.episode_cap-self.total)} remain in the episode. '
            'Use your existing observations to choose actions when no queries remain. '
            'A lack of query budget is not evidence that any task condition holds.')
        result['messages'].append(dict(role='user', content=notice))
        if self.remaining == 0:
            schema = result.get('response_format', {}).get('json_schema', {}).get('schema')
            if schema:
                properties = schema.get('properties', {})
                if 'action' in properties:
                    properties['action']['enum'] = [v for v in properties['action']['enum']
                        if v not in ('query_state', 'check_time')]
                    if 'channel' in properties:
                        properties['channel']['enum'] = ['NONE']
        return result


class BudgetTransform(ast.NodeTransformer):
    def __init__(self):
        self.steps = self.unions = 0

    def visit_For(self, node):
        self.generic_visit(node)
        if ast.unparse(node.target) == '(step_idx, step)':
            node.body.insert(0, ast.parse("_matched_budget.begin(day['name'], step['id'])").body[0])
            self.steps += 1
        if isinstance(node.iter, ast.Name) and node.iter.id == 'union_query_channels':
            node.iter = ast.Call(func=ast.Attribute(value=ast.Name(id='_matched_budget', ctx=ast.Load()),
                attr='limit_channels', ctx=ast.Load()), args=[node.iter], keywords=[])
            self.unions += 1
        return node


def instrument(module, name, budget, out, expect_union=False):
    original = getattr(module, name)
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    transform = BudgetTransform()
    tree = transform.visit(tree)
    ast.fix_missing_locations(tree)
    assert transform.steps == 1 and transform.unions == int(expect_union)
    path = Path(out)/f'overlay_{name}.py'
    path.write_text(ast.unparse(tree)+'\n')
    namespace = dict(module.__dict__, _matched_budget=budget)
    exec(compile(tree, str(path), 'exec'), namespace)
    compiled = namespace[name]
    result = types.FunctionType(compiled.__code__, module.__dict__, name, compiled.__defaults__)
    result.__kwdefaults__ = compiled.__kwdefaults__
    return result


def guarded_resolver(original, budget):
    def resolve(channel, *args, **kwargs):
        budget.charge(channel)  # Must happen BEFORE any hidden-state lookup.
        return original(channel, *args, **kwargs)
    return resolve
