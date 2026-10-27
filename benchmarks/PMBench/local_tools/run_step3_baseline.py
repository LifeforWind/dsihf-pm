"""Three queries per step without an episode cap; user relay only."""
import argparse
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from unittest.mock import patch

import run_official_baselines as official_run
import run_amem_small as amem_run
from step3_budget import QueryBudget, instrument, guarded_resolver
from relay_transport import RelayTransport as MatchedTransport, EXPECTED, configure, make_client
from run_official_baselines import ROOT, MODEL, CONDITIONS, StudyStop, write, rows

WORKSPACE = ROOT.parents[1]
NATIVE = WORKSPACE/'.worktrees/amem-native-ceffb86'
RUNTIME = WORKSPACE/'a-mem'


def environment():
    configure()


def budget_overlay(stack, budget, out):
    official = official_run.official
    modules = [official.PM_BENCH, official.TODO_LEDGER.PM_BENCH,
        official.HIER_UNION.PM_BENCH, official.HIER_BASELINE.PM_BENCH, amem_run.pm]
    unique = list({id(m):m for m in modules}.values())
    for module in unique:
        # Upstream's backend name selects Chat Completions and checks a key
        # before calling the injected client. Supply an unusable sentinel only;
        # the real credential and endpoint live exclusively in make_client().
        stack.enter_context(patch.object(module, 'load_openrouter_api_key',
            return_value='unused-injected-user-relay-client'))
        stack.enter_context(patch.object(module, 'resolve_state_query_items',
            guarded_resolver(module.resolve_state_query_items, budget)))
        stack.enter_context(patch.object(module, '_matched_budget', budget, create=True))
    for module, name, union in [(official.PM_BENCH,'run_llm',False),
            (official.TODO_LEDGER,'run_todo_ledger',False),
            (official.HIER_UNION,'run_hierarchical_agent',True)]:
        stack.enter_context(patch.object(module,'_matched_budget',budget,create=True))
        fn = instrument(module, name, budget, out, union)
        stack.enter_context(patch.object(module, name, fn))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--condition',choices=list(CONDITIONS),required=True)
    p.add_argument('--scenario',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--step-cap',type=int,choices=[3],default=3)
    args=p.parse_args()
    args.episode_cap=None
    out,cache,scenario=args.out.resolve(),args.cache.resolve(),args.scenario.resolve()
    if out.exists():raise RuntimeError('Refusing to overwrite an attempt directory')
    environment()
    client=make_client()
    # Official runner creates its output itself. Overlay files are staged separately.
    overlay=out.parent/(out.name+'-overlay');overlay.mkdir(parents=True,exist_ok=False)
    budget=QueryBudget(overlay,args.step_cap,args.episode_cap)
    try:
        def transport(c,o,k):return MatchedTransport(c,o,k,budget)
        with ExitStack() as stack:
            budget_overlay(stack,budget,overlay)
            stack.enter_context(patch.object(official_run,'AuditedTransport',transport))
            official_run.execute(args.condition,scenario,out,cache,client=client)
        manifest=json.loads((out/'manifest.json').read_text())
        manifest.update(provider='explicit-release-profile',api_base_url=os.environ['OPENAI_BASE_URL'],
            query_step_cap=args.step_cap,query_episode_cap=args.episode_cap,
            query_overlay=str(overlay),query_total=budget.total,
            model_comparison=MODEL+'; REN uses a separate Jev judge')
        entries=amem_run.pm.read_log(manifest['action_log'])
        counts=[sum(amem_run.pm.get_action_state_query_counts(e).values()) for e in entries]
        assert max(counts)<=args.step_cap and sum(counts)==budget.total
        write(out/'manifest.json',manifest)
    except BaseException as exc:
        cause=exc
        kinds=[]
        while cause is not None:
            kinds.append(type(cause).__name__);cause=cause.__cause__
        errors=[r for f in out.rglob('api_errors.jsonl') for r in rows(f)] if out.exists() else []
        retryable=any(k in ('APIConnectionError','APITimeoutError','RemoteDisconnected') for k in kinds)
        if errors:
            last=errors[-1]
            retryable |= last.get('status_code') in (429,500,502,503,504,520,522,524) or last.get('error_type') in ('APIConnectionError','APITimeoutError')
        write(overlay/'failure.json',dict(error_type=type(exc).__name__,cause_types=kinds,retryable_transport=retryable))
        if (out/'manifest.json').exists():
            manifest=json.loads((out/'manifest.json').read_text())
            manifest.update(status='failed',error_type=type(exc).__name__)
            write(out/'manifest.json',manifest)
        raise


if __name__=='__main__':main()
