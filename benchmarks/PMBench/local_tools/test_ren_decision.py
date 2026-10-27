import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from run_ren_grounded import GroundedModels
from run_ren_natural import NaturalModels

class SharedDecisionTests(unittest.TestCase):
    def test_action_binding_never_rejudges_verified_event(self):
        with tempfile.TemporaryDirectory() as folder:
            m=GroundedModels(Path(folder),'gpt-4o-mini-2024-07-18',None,'jev',judge_client=object(),judge_cache=Path(folder)/'cache')
            m.ledger_ready=lambda identity:(identity=='pay',{'task_id':identity,'states':{'0':'true'}})
            state={'intentions':[{'id':'pay'}]}
            questions={'cue_0':dict(type='noul'),'action_0':dict(type='choice',criteria={'task_1':'Pay'})}
            with patch.object(NaturalModels,'evaluate',return_value={'action_0':{'choice':'task_1'}}) as call:
                answers=m.evaluate('trigger_and_bind',state,questions)
                self.assertEqual(call.call_args.args,('bind_grounded_actions',state,{'action_0':questions['action_0']}))
                self.assertEqual(answers['cue_0']['noul'],1.)
            m.ledger_ready=lambda identity:(False,{})
            with patch.object(NaturalModels,'evaluate',return_value={'action_0':{'choice':'task_1'}}):
                with self.assertRaises(RuntimeError):m.evaluate('trigger_and_bind',state,questions)

