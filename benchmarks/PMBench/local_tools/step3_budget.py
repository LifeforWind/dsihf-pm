"""User-selected observation protocol: three per step, no episode cap."""
from copy import deepcopy
from matched_budget import QueryBudget as PreviousBudget, instrument, guarded_resolver


class QueryBudget(PreviousBudget):
    def __init__(self,out,step_cap=3,episode_cap=None):
        if step_cap != 3 or episode_cap is not None:
            raise ValueError('This protocol requires step cap 3 and no episode cap')
        super().__init__(out,step_cap,episode_cap)

    @property
    def remaining(self):
        return max(0,self.step_cap-self.used)

    def request(self,kwargs):
        result=deepcopy(kwargs)
        notice=(f'Observation budget: at most 3 channel queries per step, including clock '
                f'and repeated queries. There is no episode-level query cap. '
                f'{self.remaining} queries remain available at this step. '
                'Use your existing observations to choose actions when no queries remain. '
                'A lack of query budget is not evidence that any task condition holds.')
        result['messages'].append(dict(role='user',content=notice))
        if self.remaining==0:
            schema=result.get('response_format',{}).get('json_schema',{}).get('schema')
            if schema:
                properties=schema.get('properties',{})
                if 'action' in properties:
                    properties['action']['enum']=[v for v in properties['action']['enum']
                                                  if v not in ('query_state','check_time')]
                    if 'channel' in properties:properties['channel']['enum']=['NONE']
        return result
