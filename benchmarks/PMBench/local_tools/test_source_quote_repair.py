"""Grounding and ambiguity regressions for model-generated abbreviated quotes."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from ren_source_compiler import SourceCompiler, repair_proposal, source_span


def proposal(quote,action='Take medication at dinner'):
    return dict(op='add',id='draft',source_quote=quote,spec=dict(
        action=action,trigger='Dinner occurs.',day='EVERYDAY',kind='event',at=None,
        channel=None,comparator=None,threshold=None,parent_id=None,dependency=None,
        offset_minutes=None))


class QuoteGroundingTests(unittest.TestCase):
    def test_split_instruction_restores_full_original_span(self):
        examples = [
            ('Take medication at ... dinner.','Take medication at breakfast and dinner.'),
            ('Take medication at ... 21:00.','Take medication at 11:00 and 21:00.'),
            ('Collect … counter.','Collect the blue item at the counter.'),
            ('send ... receipt ... Friday.','Send the signed receipt before Friday.'),
            ('Pay ... $12.50.','Pay invoice A for $12.50.'),
        ]
        for quote,text in examples:
            with self.subTest(quote=quote):
                self.assertEqual(source_span(quote,text),text)

    def test_exact_and_unique_case_insensitive_quotes_keep_existing_behavior(self):
        self.assertEqual(source_span('Wait... then leave.','Wait... then leave.'),'Wait... then leave.')
        self.assertEqual(source_span('PAY THE INVOICE.','Pay the invoice.'),'Pay the invoice.')
        self.assertEqual(source_span('Pay it.','Pay it. Pay it.'),'Pay it.')
        self.assertIsNone(source_span('PAY IT.','Pay it. Pay it.'))

    def test_missing_or_unanchored_or_reordered_quotes_are_rejected(self):
        for quote in ['Call ... Alice.','... dinner.','Take ...','...','Take ... .','dinner ... Take']:
            with self.subTest(quote=quote):
                self.assertIsNone(source_span(quote,'Take medication at dinner.'))

    def test_multiple_occurrences_or_expansions_are_ambiguous(self):
        for quote,text in [
            ('Take ... dinner.','Take A at dinner. Take B at dinner.'),
            ('Take ... dinner.','Take A at dinner. Take A at dinner.'),
            ('A ... B','A B B'),
            ('A ... B','A A B'),
            ('aba ... Z','ababa Z'),
        ]:
            with self.subTest(text=text):
                self.assertIsNone(source_span(quote,text))

    def test_gaps_do_not_join_lines_sentences_or_clauses(self):
        for text in ['Take A. Then dinner.', 'Take A! Then dinner.',
                     'Take A? Then dinner.', 'Take A; then dinner.',
                     'Take A\nThen dinner.', 'Take A\r\nThen dinner.']:
            with self.subTest(text=text):
                self.assertIsNone(source_span('Take ... dinner.',text))

    def test_internal_anchor_paths_with_same_outer_span_are_not_ambiguous(self):
        self.assertEqual(source_span('A ... B ... C','A B B C'),'A B B C')

    def test_one_unmarked_omission_restores_unique_original(self):
        for quote,text in [
            ('Take medication at dinner.','Take medication at breakfast and dinner.'),
            ('Take asthma medication at 21:00.','Take asthma medication at 11:00 and 21:00.'),
            ('Send the receipt Friday.','Send the receipt before Friday.'),
        ]:
            with self.subTest(quote=quote):self.assertEqual(source_span(quote,text),text)

    def test_unmarked_omission_rejects_weak_or_unseen_or_reordered_anchors(self):
        for quote,text in [
            ('Do send invoice.','Do not send invoice.'),
            ('Take medication at lunch.','Take medication at breakfast and dinner.'),
            ('Send receipt the Friday.','Send the receipt before Friday.'),
            ('Send the receipt Friday.','Send the receipt. Then wait until Friday.'),
            ('Send the receipt Friday.','Send the receipt before Friday. Send the receipt after Friday.'),
        ]:
            with self.subTest(quote=quote,text=text):self.assertIsNone(source_span(quote,text))

    def test_repair_preserves_proposal_semantics_and_original_response(self):
        original=proposal('Take medication at ... dinner.');before=deepcopy(original)
        fixed,changes=repair_proposal(original,'Take medication at breakfast and dinner.')
        self.assertEqual(original,before)
        self.assertEqual(fixed['spec'],original['spec'])
        self.assertEqual(fixed['source_quote'],'Take medication at breakfast and dinner.')
        self.assertEqual(changes[0][1],'source_quote')
        self.assertIn('ellipsis expansion',changes[0][-1])

    def test_compiler_retains_raw_cache_and_does_not_trust_its_own_proposal(self):
        class Model:
            model='synthetic-model'
            def __init__(self,out):self.out=out;self.records=[];self.calls=0
            def chat(self,*args):
                self.calls+=1
                return dict(proposals=[proposal('Take medication at ... dinner.')])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model=Model(root);compiler=SourceCompiler(model,root/'cache')
            text='Take medication at breakfast and dinner.'
            fixed=compiler.compile('observation',text,'Monday',1,['clock'])
            self.assertEqual(fixed[0]['source_quote'],text)
            cache=next((root/'cache').glob('*.json'));raw=cache.read_bytes()
            self.assertEqual(json.loads(raw)['result']['proposals'][0]['source_quote'],
                             'Take medication at ... dinner.')
            compiler.draft=type(compiler.draft)()
            compiler.compile('observation',text,'Monday',1,['clock'])
            self.assertEqual(model.calls,1)
            self.assertEqual(cache.read_bytes(),raw)
            # Repaired syntax is only a proposal. The verified store can reject it.
            from run_intentions import IntentionStore
            store=IntentionStore();store.observe('observation',text,'Monday',1)
            store.apply(fixed[0],'observation',accepted=False)
            self.assertFalse(store.intents)


if __name__=='__main__':unittest.main()
