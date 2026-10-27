import json,tempfile
from pathlib import Path
import unittest
from run_ren_verified import VerifiedCompiler
from run_intentions import IntentionStore

class Model:
    model='gpt-4o-mini-2024-07-18'
    def __init__(self,out):self.out=out;self.records=[];self.contexts=[]
    def chat(self,stage,messages,schema,max_tokens):
        state=json.loads(messages[-1]['content']);self.contexts.append(state)
        if state['observation']=='A proposed task.':
            return dict(proposals=[dict(op='add',id='draft',source_quote=state['observation'],spec=dict(
                action='An unverified task',trigger='Event',kind='event',day='Monday',at=None,channel=None,
                comparator=None,threshold=None,parent_id=None,dependency=None,offset_minutes=None))])
        return dict(proposals=[])

class VerifiedContextTests(unittest.TestCase):
    def test_rejected_proposal_never_enters_next_compiler_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model=Model(root);compiler=VerifiedCompiler(model,root/'cache');accepted=IntentionStore()
            compiler.authoritative_store=lambda:accepted
            proposed=compiler.compile('a','A proposed task.','Monday',1,['email'])
            self.assertEqual(len(proposed),1);self.assertTrue(compiler.draft.intents);self.assertFalse(accepted.intents)
            compiler.compile('b','Next observation.','Monday',2,['email'])
            self.assertEqual(model.contexts[-1]['existing'],[])
    def test_accepted_changes_and_cancellation_visible_on_next_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model=Model(root);compiler=VerifiedCompiler(model,root/'cache');accepted=IntentionStore()
            compiler.authoritative_store=lambda:accepted
            proposal=compiler.compile('a','A proposed task.','Monday',1,['email'])[0]
            accepted.observe('a','A proposed task.','Monday',1);accepted.apply(proposal,'a')
            compiler.compile('b','Next observation.','Monday',2,['email'])
            self.assertEqual(model.contexts[-1]['existing'][0]['id'],proposal['id'])
            accepted.observe('cancel','Cancel task.','Monday',2)
            accepted.apply(dict(proposal,op='cancel',source_quote='Cancel task.'),'cancel')
            compiler.compile('c','Another observation.','Monday',3,['email'])
            self.assertEqual(model.contexts[-1]['existing'][0]['status'],'canceled')
    def test_missing_authority_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);compiler=VerifiedCompiler(Model(root),root/'cache')
            with self.assertRaises(RuntimeError):compiler.compile('a','Text','Monday',1,['email'])

if __name__=='__main__':unittest.main()
