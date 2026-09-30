import os
os.environ.setdefault('DATABASE_URL', 'postgresql://test:test@localhost/test')
os.environ.setdefault('OPENAI_API_KEY', 'test')
os.environ.setdefault('API_AUTH_TOKEN', 'test')

import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock
from sqlalchemy.dialects import postgresql

from classifier import (
    GPCClassifier, ClassificationCandidateRow, confidence_from_ranked_candidates,
    status_from_confidence, rerank_candidate,
)
from display_mapping import display_labels_for_gpc
from normalization import normalize_receipt_text
from taxonomy import BRICK_PATHS


def brick(code=10005897):
    path = BRICK_PATHS[code]
    return NS(id=code, code=code, level=4, title=path[-1], full_title=' '.join(path),
              definition='', active=True, level_2_category=None, level_3_category=None)


def candidate(score=.5, similarity=.5, code=10005897):
    item = brick(code)
    return ClassificationCandidateRow(item.id,item,similarity,score,item.title,code,4,[])


class NormalizationTests(unittest.TestCase):
    def test_meaning_and_quantities_are_preserved(self):
        for source, expected in {
            'POT ROAST':'pot roast', 'CAR WAX':'car wax', 'BAKING SODA':'baking soda',
            'CHOCOLATE THINS':'chocolate thins', 'WHL WHT BRD 500G':'whole wheat bread 500g',
            '2% MLK 1.5 L':'2% milk 1.5 l', 'GLUTEN-FREE':'gluten-free',
            'CHKN BRST':'chicken breast', 'FRZ BAGEL':'frozen bagel',
        }.items():
            with self.subTest(source=source):
                self.assertEqual(normalize_receipt_text(source).normalized_text,expected)


class RankingTests(unittest.TestCase):
    def test_tie_requires_review(self):
        confidence=confidence_from_ranked_candidates([candidate(),candidate(code=10005900)])
        self.assertTrue(status_from_confidence(confidence)[1])

    def test_single_weak_candidate_is_not_promoted(self):
        confidence=confidence_from_ranked_candidates([candidate(.25,.25)])
        self.assertEqual(confidence,.25)
        self.assertEqual(status_from_confidence(confidence),('uncertain',True))

    def test_keyword_bonuses_do_not_become_probability(self):
        confidence=confidence_from_ranked_candidates([candidate(.9,.25),candidate(.5,.2)])
        self.assertEqual(confidence,.25)

    def test_word_match_does_not_match_substrings(self):
        item=NS(title='Shampoo',full_title='Beauty Shampoo',definition='',active=True,level=4)
        _, reasons=rerank_candidate(item,item,.4,normalize_receipt_text('ham'))
        self.assertFalse(any('term_match' in r for r in reasons))

    def test_query_only_retrieves_active_bricks(self):
        db=Mock()
        db.execute.return_value.all.return_value=[(NS(id=1),brick(),.3)]
        c=GPCClassifier(db,Mock(),Mock())
        c._retrieve_and_rank([0.0]*1536,normalize_receipt_text('Banana'))
        statement=db.execute.call_args_list[0].args[0]
        sql=str(statement.compile(dialect=postgresql.dialect()))
        self.assertIn('gpc_level.level =',sql)
        self.assertIn('gpc_level.active IS true',sql)
        self.assertIn('gpc_level.code NOT IN',sql)
        self.assertLessEqual(db.execute.call_count,2)

    def test_pet_context_and_hot_dog_exception(self):
        for query, restricted in [('Cat Chow salmon',True),('One Cat chicken',True),('Dog biscuits',True),('Cat eye sunglasses',False),('hot dog buns',False)]:
            db=Mock(); db.execute.return_value.all.return_value=[(NS(id=1),brick(),.3)]
            GPCClassifier(db,Mock(),Mock())._retrieve_and_rank([0.0]*1536,normalize_receipt_text(query))
            sql=str(db.execute.call_args_list[0].args[0].compile(dialect=postgresql.dialect()))
            self.assertEqual('gpc_level.code IN (' in sql,restricted)

    def test_description_failure_uses_original(self):
        vector=Mock(return_value=[0.0]*1536)
        db=Mock(); db.execute.return_value.all.return_value=[]
        c=GPCClassifier(db,Mock(side_effect=TimeoutError()),vector)
        c._retrieve_and_rank=Mock(return_value=[candidate()])
        result=c.classify('ORG BNNA')
        self.assertTrue(result['description_fallback'])
        self.assertIn('organic banana',vector.call_args.args[0])
        self.assertEqual(result['category'],'Produce')

    def test_rewrite_never_replaces_original_in_embedding(self):
        vector=Mock(return_value=[0.0]*1536)
        db=Mock(); db.execute.return_value.all.return_value=[]
        c=GPCClassifier(db,Mock(return_value=NS(choices=[NS(message=NS(content='A container.'))])),vector)
        c._retrieve_and_rank=Mock(return_value=[candidate()])
        c.classify('Farmers 2% milk jug')
        self.assertIn('farmers 2% milk jug',vector.call_args.args[0])

    def test_existing_category_names(self):
        for code,label in [(10005897,'Produce'),(10005900,'Produce'),(10000165,'Bakery'),
                           (10000025,'Dairy & Eggs'),(10000211,'Pantry'),(10000047,'Snacks & Candy')]:
            self.assertEqual(display_labels_for_gpc(brick(code)).category,label)


if __name__ == '__main__': unittest.main()
