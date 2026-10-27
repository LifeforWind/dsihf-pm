#!/usr/bin/env python3
"""Make the verified demand ledger the sole persistent compiler context.

The compiler may propose changes; rejected proposals never become its future
memory. The prompts, evidence adapter, lifecycle and round-robin are unchanged.
"""
from copy import deepcopy
from functools import partial
from pathlib import Path
import sys
from unittest.mock import patch
import run_recommendation as base
from ren_source_compiler import SourceCompiler
from run_ren_situated import SituatedModels
from run_ren_grounded import GroundedRENPolicy
from run_ren_natural import NEEDS_SOURCE

class VerifiedCompiler(SourceCompiler):
    def compile(self,evidence_id,text,day,step,channels):
        if not callable(getattr(self,'authoritative_store',None)):
            raise RuntimeError('Verified compiler requires an authoritative demand store')
        self.draft=deepcopy(self.authoritative_store())
        base.append(self.models.out/'compiler_contexts.jsonl',dict(evidence_id=evidence_id,
            accepted_store_sha256=base.digest(self.draft.dump()),intentions=[dict(id=i.id,
                revision=i.revision,status=i.status,**i.spec()) for i in self.draft.intents.values()]))
        return super().compile(evidence_id,text,day,step,channels)

class VerifiedPolicy(GroundedRENPolicy):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if not isinstance(self.compiler,VerifiedCompiler):raise TypeError('Verified policy requires VerifiedCompiler')
        self.compiler.authoritative_store=lambda:self.store

def main():
    argv=sys.argv[1:];index=argv.index('--candidate-mode');mode=argv[index+1];del argv[index:index+2]
    args=base.parse_args(argv)
    sources=[Path(__file__),NEEDS_SOURCE]+[Path(__file__).with_name(n) for n in
        ('run_ren_grounded.py','run_ren_natural.py','run_ren_situated.py','ren_public_frames.py','ren_source_compiler.py','residual_policy.py','compiler_failure.py')]
    with patch.object(base,'Compiler',VerifiedCompiler):
        base.main(args=args,policy_factory=partial(VerifiedPolicy,candidate_mode=mode),model_factory=SituatedModels,
            extra_sources=sources,extra_manifest=dict(ren_adapter='verified-ledger-v1',candidate_mode=mode,
                compiler_context='accepted demand ledger refreshed before every source batch',
                shared_condition_decision=True,natural_language_entry=True,structured_oracle_contracts=False,
                source_span_repair='unique case-insensitive match restored to exact observed span',
                condition_scope='only trigger; date already enforced',observation_frames='quote-grounded public text only',
                model_identity_retries=2,evidence_validity='current public step only; inferred by adapter',
                compiler_failure_policy='reject-truncated-compiler-batch-v1',
                receipt_interface_available=False,completion_feedback_exercised=False))

if __name__=='__main__':main()
