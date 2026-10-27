from openmodel_config import MODEL as ACTIVE_MODEL, BASE as ACTIVE_BASE
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
from run_official_baselines import StudyStop
from step3_budget import QueryBudget,guarded_resolver


class Step3Tests(unittest.TestCase):
    def test_no_episode_cutoff_and_step_guard_before_read(self):
        with tempfile.TemporaryDirectory() as folder:
            b=QueryBudget(Path(folder));hidden=Mock(return_value=['visible response'])
            query=guarded_resolver(hidden,b)
            for step in range(45):
                b.begin('Monday' if step<44 else 'Tuesday',str(step))
                for channel in ['clock','email','email']:query(channel)
                with self.assertRaises(StudyStop):query('bank_balance')
            self.assertEqual(hidden.call_count,135)
            self.assertEqual(b.total,135);self.assertIsNone(b.episode_cap)
            b.begin('Tuesday','next');self.assertEqual(b.remaining,3)

    def test_notice_and_schema(self):
        with tempfile.TemporaryDirectory() as folder:
            b=QueryBudget(Path(folder));b.begin('Monday','1')
            request=dict(messages=[],response_format=dict(json_schema=dict(schema=dict(properties={
                'action':dict(enum=['choose','query_state','check_time']),
                'channel':dict(enum=['clock','NONE']), 'task_ids':dict(type='array')}))))
            for _ in range(3):b.charge('clock')
            changed=b.request(request)
            self.assertIn('no episode-level query cap',changed['messages'][-1]['content'])
            self.assertNotIn('126',changed['messages'][-1]['content'])
            self.assertEqual(changed['response_format']['json_schema']['schema']['properties']['action']['enum'],['choose'])
            self.assertEqual(request['messages'],[])

    def test_reject_old_protocol(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):QueryBudget(folder,2,126)


if __name__=='__main__':unittest.main()

class RouteTests(unittest.TestCase):
    def test_ren_routes_by_model_before_send(self):
        from run_step3_ren import check_route
        relay='https://configured-relay.example/v1'
        self.assertEqual(check_route(relay+'/chat/completions',dict(model=ACTIVE_MODEL),relay),'user-relay')
        self.assertEqual(check_route('https://openrouter.ai/api/alpha/decisions',dict(model='typesafe/jev-1.13'),relay),'jev-openrouter')
        with self.assertRaises(AssertionError):check_route('https://openrouter.ai/api/v1/chat/completions',dict(model=ACTIVE_MODEL),relay)
        with self.assertRaises(AssertionError):check_route(relay+'/chat/completions',dict(model='typesafe/jev-1.13'),relay)
