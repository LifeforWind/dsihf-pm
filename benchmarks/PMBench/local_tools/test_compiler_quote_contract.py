"""The prompt candidate keeps the original strict quote acceptance contract."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from ren_source_compiler import SourceCompiler, source_span
from run_intentions import COMPILER_PROMPT, IntentionStore, PROPOSALS_SCHEMA


def proposal(identity, action, trigger, quote):
    return dict(op='add', id=identity, source_quote=quote, spec=dict(
        action=action, trigger=trigger, day='Monday', kind='event', at=None,
        channel=None, comparator=None, threshold=None, parent_id=None,
        dependency=None, offset_minutes=None))


class Model:
    model = 'offline-fixture'
    def __init__(self, out, proposals):
        self.out, self.proposals, self.records = out, proposals, []
        self.requests = []
    def chat(self, stage, messages, schema, max_tokens):
        self.requests.append((stage, messages, schema, max_tokens))
        return dict(proposals=deepcopy(self.proposals))


class QuoteContractTests(unittest.TestCase):
    observation = 'Inspect the filter before opening and after closing.'

    def compile(self, quotes):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        proposals = [proposal('draft_' + str(n), action, trigger, quote)
            for n,(action,trigger,quote) in enumerate([
                ('Inspect the filter before opening.', 'Opening occurs.', quotes[0]),
                ('Inspect the filter after closing.', 'Closing occurs.', quotes[1])])]
        model = Model(root, proposals)
        compiler = SourceCompiler(model, root / 'cache')
        result = compiler.compile('observed', self.observation, 'Monday', 1, [])
        return root, model, compiler, result

    def test_same_complete_sentence_can_support_two_distinct_occurrences(self):
        _,_,compiler,result = self.compile([self.observation,self.observation])
        self.assertEqual(len(result), 2)
        self.assertEqual(len(compiler.draft.intents), 2)
        self.assertNotEqual(result[0]['id'],result[1]['id'])
        self.assertEqual([r['source_quote'] for r in result], [self.observation]*2)

    def test_ellipsis_and_unmarked_omissions_still_fail_original_gate(self):
        root,_,compiler,result = self.compile([
            'Inspect the filter ... after closing.', 'Inspect the filter after closing.'])
        self.assertEqual(result, [])
        self.assertEqual(compiler.draft.intents, {})
        log = json.loads((root/'compilation.jsonl').read_text())
        self.assertEqual([r['reason'] for r in log['rejected']],
                         ['Proposal lacks an exact observed source span']*2)
        self.assertIsNone(source_span('Inspect the filter ... after closing.', self.observation))
        self.assertIsNone(source_span('Inspect the filter after closing.', self.observation))

    def test_prompt_change_uses_a_different_cache_key_with_same_schema_and_cap(self):
        root,model,_,_ = self.compile([self.observation,self.observation])
        stage,messages,schema,cap = model.requests[0]
        self.assertEqual((stage,schema,cap), ('compile',PROPOSALS_SCHEMA,5500))
        self.assertEqual(messages[0]['content'],COMPILER_PROMPT)
        key = hashlib.sha256(json.dumps([model.model,messages,schema],sort_keys=True).encode()).hexdigest()
        cached = json.loads((root/'cache'/(key+'.json')).read_text())
        self.assertEqual(cached['result']['proposals'],model.proposals)
        self.assertEqual(cached['request'],messages)


if __name__ == '__main__':
    unittest.main()
