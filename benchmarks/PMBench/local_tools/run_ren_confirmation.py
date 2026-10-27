#!/usr/bin/env python3
"""Native-interface held-out runner; frozen adapter, candidate arm and budgets."""
from functools import partial
from pathlib import Path
import sys
from unittest.mock import patch
import run_recommendation as base
from run_ren_grounded import GroundedModels
from run_ren_situated import SituatedModels,SourceCompiler
from run_ren_natural import NEEDS_SOURCE
from ren_confirmation_policy import ConfirmationPolicy
from run_ren_verified import VerifiedCompiler

class WeekBudget:
    def check_week_budget(self):
        if self.week_totals['calls']>=4000 or self.week_totals['input_tokens']>=12000000 or self.week_totals['cost']>=2:
            raise RuntimeError('Frozen confirmation-week physical-call budget reached')

class ConfirmationGroundedModels(WeekBudget,GroundedModels):pass
class ConfirmationSituatedModels(WeekBudget,SituatedModels):pass

class ConfirmationVerifiedPolicy(ConfirmationPolicy):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if not isinstance(self.compiler,VerifiedCompiler):raise TypeError('Verified confirmation requires authoritative compiler')
        self.compiler.authoritative_store=lambda:self.store

def main():
    argv=sys.argv[1:]
    def take(name,default=None):
        if name not in argv:
            if default is not None:return default
            raise ValueError('Missing '+name)
        i=argv.index(name);value=argv[i+1];del argv[i:i+2];return value
    adapter=take('--adapter');mode=take('--candidate-mode');cap=int(take('--episode-cap','0')) or None
    if adapter not in ('grounded','situated','verified'):raise ValueError('Unknown frozen adapter')
    if mode not in ('semantic','task_channels_filtered','remaining'):raise ValueError('Unknown candidate arm')
    args=base.parse_args(argv)
    compiler=VerifiedCompiler if adapter=='verified' else SourceCompiler if adapter=='situated' else base.Compiler
    model=ConfirmationGroundedModels if adapter=='grounded' else ConfirmationSituatedModels
    policy=ConfirmationVerifiedPolicy if adapter=='verified' else ConfirmationPolicy
    sources=[Path(__file__),NEEDS_SOURCE]+[Path(__file__).with_name(n) for n in
        ('run_ren_verified.py','run_ren_grounded.py','run_ren_natural.py','run_ren_situated.py','ren_public_frames.py',
         'ren_source_compiler.py','ren_confirmation_policy.py','residual_policy.py','compiler_failure.py')]
    with patch.object(base,'Compiler',compiler):
        base.main(args=args,policy_factory=partial(policy,candidate_mode=mode,episode_cap=cap),
            model_factory=model,extra_sources=sources,extra_manifest=dict(ren_adapter=adapter+'-confirmation-v1',
                candidate_mode=mode,episode_query_cap=cap,shared_condition_decision=True,
                compiler_context='accepted demand ledger' if adapter=='verified' else 'compiler draft',
                confirmation_per_week_limits=dict(responses=4000,input_tokens=12000000,returned_usd=2),
                natural_language_entry=True,structured_oracle_contracts=False,private_step_timestamp_passed=False,
                model_identity_retries=2,evidence_validity='current public step only; inferred by adapter',
                intention_compile_failure_policy='reject-truncated-intention-batch-v1',
                receipt_interface_available=False,completion_feedback_exercised=False))

if __name__=='__main__':
    try:main()
    except Exception as exc:
        transport=type(exc).__name__ in ('APIConnectionError','APITimeoutError','RateLimitError','InternalServerError','URLError','TimeoutError') or str(exc).startswith(
            ('Decisions transport failed','Decisions HTTP 429','Decisions HTTP 500','Decisions HTTP 502','Decisions HTTP 503','Decisions HTTP 504'))
        if '--out' in sys.argv:
            out=Path(sys.argv[sys.argv.index('--out')+1])
            if out.exists():base.write_json(out/'failure_kind.json',dict(error_type=type(exc).__name__,retryable_transport=transport))
        raise
