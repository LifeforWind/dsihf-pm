from test_embedding_route import RouteTest
from memory_transport import MODEL,ExperimentStop,digest,write
from qwen_wire import wire
from types import SimpleNamespace
import json

class NativeFinishTest(RouteTest):
    def chat_case(self,accepted=False,response_model=MODEL):
        t=self.setup_transport();t.stage='pma.intervene'
        request=wire(dict(model=MODEL,messages=[dict(role='user',content='Current public observation')],max_tokens=4096))
        key=digest([digest(['chat',t.stage,request,t.base]),1]);marker=dict(kind='chat',stage=t.stage,cache_key=key);t.trace=[marker]
        response=dict(model=response_model,choices=[dict(index=0,finish_reason='length',message=dict(role='assistant',content='Unfinished original reply'))])
        rec=dict(kind='chat',stage=t.stage,request=request,response=response,api_base_url=t.base,cache_key=key,accepted=accepted)
        write(t.cache/(key+'.json'),rec);return t,request,response,rec
    def test_original_rejected_length_reply_replayed_byte_exact(self):
        t,req,res,rec=self.chat_case();p=t.cache/(rec['cache_key']+'.json');before=p.read_bytes()
        self.assertEqual(t.request('chat',req),res);self.assertEqual(t.physical,0);self.assertEqual(p.read_bytes(),before)
    def test_wrong_model_still_rejected(self):
        t,req,_,_=self.chat_case(response_model='wrong')
        with self.assertRaisesRegex(ExperimentStop,'provenance'):t.request('chat',req)
    def test_new_length_reply_not_regenerated(self):
        t=self.setup_transport(offline=False);t.stage='pma.intervene';calls=[]
        req=dict(model=MODEL,messages=[dict(role='user',content='test')]);res=dict(model=MODEL,choices=[dict(finish_reason='length',message=dict(content='truncated'))])
        def create(**kwargs):calls.append(kwargs);return SimpleNamespace(model_dump=lambda **kw:res)
        t.client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        self.assertEqual(t.request('chat',req),res);self.assertEqual(len(calls),1)
if __name__=='__main__':
 import unittest;unittest.main()
