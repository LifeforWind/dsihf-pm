import json, tempfile, unittest
from pathlib import Path
from collections import Counter
from types import SimpleNamespace
from copy import deepcopy
from memory_transport import Transport, ExperimentStop, digest, write
from openmodel_config import BASE

class RouteTest(unittest.TestCase):
    def setup_transport(self, trace=(), offline=True):
        t=object.__new__(Transport);p=Path(self.tmp.name)
        t.out=p/'out';t.cache=p/'cache';t.out.mkdir(exist_ok=True);t.cache.mkdir(exist_ok=True)
        t.base=BASE;t.embedding_base='https://user-relay.example/v1'
        t.offline=offline;t.occurrences=Counter();t.stage='mem0.write';t.physical=t.logical=t.request_chars=0
        t.trace_path=p/'tape.jsonl';t.trace=deepcopy(list(trace));t.trace_path.write_text(''.join(json.dumps(x)+'\n' for x in trace))
        return t
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.req=dict(model='text-embedding-3-small',input=['test'],dimensions=1536,encoding_format='float')
    def tearDown(self):self.tmp.cleanup()
    def marker(self,t):
        key=digest([digest(['embedding',t.stage,self.req,t.embedding_base]),1]);return dict(kind='embedding',stage=t.stage,cache_key=key)
    def test_exact_relay_prefix_without_network(self):
        t=self.setup_transport();mark=self.marker(t);t.trace=[mark]
        response=dict(model='text-embedding-3-small',data=[])
        write(t.cache/(mark['cache_key']+'.json'),dict(request=self.req,response=response,api_base_url=t.embedding_base,kind='embedding',accepted=True))
        self.assertEqual(t.request('embedding',self.req),response);self.assertEqual(t.physical,0)
    def test_openrouter_cache_cannot_be_reused(self):
        t=self.setup_transport(offline=False)
        key=digest([digest(['embedding',t.stage,self.req,'https://openrouter.ai/api/v1']),1])
        t.trace=[dict(kind='embedding',stage=t.stage,cache_key=key)]
        with self.assertRaisesRegex(ExperimentStop,'prefix differs'):t.request('embedding',self.req)
    def test_new_route_actual_wire_key_and_replay(self):
        t=self.setup_transport(offline=False);calls=[]
        response=dict(model='text-embedding-3-small',data=[dict(index=0,embedding=[0.1]*1536)])
        def create(**req):calls.append(req);return SimpleNamespace(model_dump=lambda **kw:response)
        t.embedding_client=SimpleNamespace(embeddings=SimpleNamespace(create=create))
        self.assertEqual(t.request('embedding',self.req),response)
        self.assertEqual(calls[0]['model'],'text-embedding-3-small')
        trace=deepcopy(t.trace);key=trace[0]['cache_key'];rec=json.loads((t.cache/(key+'.json')).read_text())
        self.assertEqual(rec['api_base_url'],t.embedding_base);self.assertIn('OPENAI_API_KEY',rec['credential_source'])
        self.assertEqual(key,digest([digest(['embedding',t.stage,calls[0],t.embedding_base]),1]))
        r=self.setup_transport(trace);self.assertEqual(r.request('embedding',self.req),response);self.assertEqual(r.physical,0)
    def test_changed_input_rejected_before_call(self):
        t=self.setup_transport(offline=False);t.trace=[self.marker(t)]
        with self.assertRaisesRegex(ExperimentStop,'prefix differs'):t.request('embedding',dict(self.req,input=['changed']))
    def test_changed_dimensions_refused(self):
        t=self.setup_transport(offline=False)
        with self.assertRaisesRegex(ExperimentStop,'shape/encoding'):t.request('embedding',dict(self.req,dimensions=1024))
if __name__=='__main__':unittest.main()
