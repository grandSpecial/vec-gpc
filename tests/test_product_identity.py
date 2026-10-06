import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock

from sqlalchemy.dialects import postgresql

from test_classifier import brick
from classifier import GPCClassifier
from normalization import normalize_receipt_text
from product_identity import FAMILIES, explicit_product_family


class ProductIdentityTests(unittest.TestCase):
    def test_explicit_receipt_products(self):
        for query, family in {
            'Chocolate chip muffins': 'sweet_muffins',
            'Banana chocolate chunk muffin': 'sweet_muffins',
            'Blueberry muffins 6 pack': 'sweet_muffins',
            'GSPP roasted pepper pizza': 'pizza',
            'GSPP roasted crust four cheese pizza': 'pizza',
            'Frozen pizza 500 g': 'pizza',
            'PC Blue Menu sparkling water, orange': 'sparkling_water',
            'PC sparkling water': 'sparkling_water',
            'Que Pasa nacho tortilla chips': 'snack_chips',
            'PC kettle chips': 'snack_chips',
            'Italian bread crumbs': 'bread_crumbs',
            'Hot pepper rings': 'pepper_rings',
        }.items():
            with self.subTest(query=query):
                self.assertEqual(explicit_product_family(query), family)

    def test_different_products_and_special_context_abstain(self):
        for query in (
            'Pizza maker', 'Pizza cutter', 'Pizza oven', 'Pizza flavoured chips',
            'Pizza rolls', 'Pizza dough', 'Muffin pan', 'Muffin mix',
            'English muffins', 'Cheddar savoury muffins', 'Corn muffins',
            'Cannabis chocolate chip muffins', 'THC muffins', 'CBD sparkling water',
            'Dog treat muffins', 'Sparkling water maker', 'Sparkling water machine',
            'Chocolate muffin scented candle', 'Pizza toy', 'T-Rex', 'OU Lavender',
            'Chocolate chips', 'Wood chips', 'Fresh hot pepper rings',
            'Frozen hot pepper rings', 'Bread crumb coated chicken',
            'Frozen tortilla chips',
        ):
            with self.subTest(query=query):
                self.assertIsNone(explicit_product_family(query))

    def test_retrieval_and_alias_queries_both_preserve_family(self):
        for query, family, code in (
            ('Chocolate chip muffins', 'sweet_muffins', 10000171),
            ('GSPP roasted pepper pizza', 'pizza', 10000249),
            ('PC Blue Menu sparkling water, orange', 'sparkling_water', 10008410),
        ):
            with self.subTest(query=query):
                db = Mock()
                db.execute.return_value.all.return_value = [(NS(id=code), brick(code), .5)]
                classifier = GPCClassifier(db, None, Mock())
                ranked = classifier._retrieve_and_rank([0.0] * 1536, normalize_receipt_text(query))
                for call in db.execute.call_args_list:
                    sql = str(call.args[0].compile(
                        dialect=postgresql.dialect(), compile_kwargs={'literal_binds': True}
                    ))
                    allowed = ', '.join(str(c) for c in FAMILIES[family])
                    self.assertIn(f'gpc_level.code IN ({allowed})', sql)
                self.assertIn(f'explicit_product_family:{family}', ranked[0].reasons)

    def test_bad_model_selection_falls_back_within_family(self):
        db = Mock()
        # retrieval, then ancestor display labels
        db.execute.return_value.all.side_effect = [
            [(NS(id=10000171), brick(10000171), .5)], [],
        ]
        classifier = GPCClassifier(db, None, Mock(return_value=[0.0] * 1536),
            select_candidate=lambda *_: {'code': 10008078, 'needs_review': False})
        result = classifier.classify('Chocolate chip muffins')
        self.assertEqual(result['category'], 'Bakery')
        self.assertTrue(result['selection_fallback'])
        self.assertTrue(result['needs_review'])
        self.assertEqual(result['identity_rule'], 'sweet_muffins')


if __name__ == '__main__':
    unittest.main()
