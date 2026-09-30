import json
import unittest
from unittest.mock import Mock
from types import SimpleNamespace as NS
from test_classifier import candidate
from classifier import GPCClassifier
from normalization import normalize_receipt_text
from selection import make_selector
from taxonomy import product_type_codes
from embedding_text import embedding_revision
from display_mapping import display_labels_for_gpc
from test_classifier import brick


class SelectionTests(unittest.TestCase):
    def classify(self, selector):
        db=Mock(); db.execute.return_value.all.return_value=[]
        describe=None
        embed=Mock(return_value=[0.0]*1536)
        classifier=GPCClassifier(db,describe,embed,select_candidate=selector)
        classifier._retrieve_and_rank=Mock(return_value=[candidate(),candidate(.45,.45,10005900)])
        result=classifier.classify('APPL',True)
        self.assertIn('Product: apple',embed.call_args.args[0])
        return result

    def test_model_can_select_a_valid_alternative(self):
        r=self.classify(lambda *_:{'code':10005900,'needs_review':False,'description':'An apple.'})
        self.assertEqual(r['code'],10005900)
        self.assertEqual(r['candidates'][0]['gpc_code'],r['code'])
        self.assertEqual(r['selection_source'],'model')

    def test_invalid_code_falls_back_without_failing_request(self):
        r=self.classify(lambda *_:{'code':12345,'needs_review':False})
        self.assertEqual(r['code'],10005897)
        self.assertTrue(r['selection_fallback'])
        self.assertTrue(r['needs_review'])

    def test_model_failure_and_malformed_result(self):
        for response in [None,{}, {'code':10005900,'needs_review':'false'}]:
            with self.subTest(response=response):
                r=self.classify(lambda *_:response)
                self.assertTrue(r['selection_fallback'])

    def test_timeout_falls_back(self):
        r=self.classify(Mock(side_effect=TimeoutError()))
        self.assertTrue(r['selection_fallback'])
        self.assertTrue(r['needs_review'])

    def test_valid_selector_result_is_cached(self):
        client=Mock();client.chat.completions.create.return_value=NS(choices=[NS(message=NS(content=json.dumps({'candidate_id':0,'needs_review':False,'description':'Banana'})))])
        select=make_selector(client,'test')
        for _ in range(2): select(normalize_receipt_text('Banana'),[candidate()])
        self.assertEqual(client.chat.completions.create.call_count,1)

    def test_invalid_selection_is_not_cached(self):
        client=Mock();client.chat.completions.create.return_value=NS(choices=[NS(message=NS(content='{"candidate_id":99,"needs_review":false}'))])
        select=make_selector(client,'test')
        for _ in range(2):
            with self.assertRaises(ValueError):select(normalize_receipt_text('Banana'),[candidate()])
        self.assertEqual(client.chat.completions.create.call_count,2)

    def test_product_type_retrieval_does_not_promote_ingredients(self):
        self.assertIn(10000165,product_type_codes('tortilla'))
        self.assertIn(10000165,product_type_codes('scone'))
        self.assertNotIn(10000165,product_type_codes('rice'))

    def test_missing_mapping_uses_existing_ancestor_label(self):
        self.assertEqual(display_labels_for_gpc(brick(10000573),{'Skin Products':'Skin Care'}).category,'Skin Care')
        self.assertEqual(display_labels_for_gpc(brick(10000522),{'Pet Food/Drinks':'Pet Food & Drinks'}).category,'Pet Food & Drinks')
        self.assertEqual(display_labels_for_gpc(brick(10005897),{'Fruits - Unprepared/Unprocessed (Fresh)':'Fresh Fruit'}).category,'Produce')

    def test_embedding_revision_changes_with_content(self):
        self.assertNotEqual(embedding_revision('old'),embedding_revision('new'))
        self.assertEqual(embedding_revision('new'),embedding_revision('new'))


if __name__=='__main__':unittest.main()
