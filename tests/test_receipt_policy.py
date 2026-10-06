import unittest
from unittest.mock import Mock

from test_classifier import brick
from classifier import GPCClassifier
from display_mapping import display_labels_for_gpc
from normalization import normalize_receipt_text
from receipt_charges import receipt_charge_result


class ReceiptPolicyTests(unittest.TestCase):
    def test_display_categories_preserve_gpc_identity(self):
        for code, query, expected in (
            (10000166, 'Multigrain crackers', 'Snacks & Candy'),
            (10000161, 'Savoury crackers', 'Snacks & Candy'),
            (10000161, 'Chocolate cookies', 'Bakery'),
            (10000166, 'Croutons', 'Bakery'),
            (10000177, 'Que Pasa tortilla chips', 'Snacks & Candy'),
            (10000158, 'Italian bread crumbs', 'Pantry'),
            (10000157, 'Italian bread crumbs', 'Pantry'),
            (10000158, 'Baking soda', 'Bakery'),
            (10000244, 'Pickled pepper rings', 'Pantry'),
            (10000272, 'Canned vegetables', 'Pantry'),
            (10000006, 'Shelf stable vegetables', 'Pantry'),
            (10005900, 'Apples', 'Produce'),
            # Wrong bricks must remain visible as wrong product selections.
            (10001985, 'Multigrain crackers', 'Small Domestic Appliances'),
        ):
            with self.subTest(query=query, code=code):
                item = brick(code)
                labels = display_labels_for_gpc(item, receipt_text=query)
                self.assertEqual(labels.category, expected)
                self.assertEqual(item.code, code)

    def test_deposit_bypasses_model_and_database(self):
        db, describe, embed, select = Mock(), Mock(), Mock(), Mock()
        result = GPCClassifier(db, describe, embed, select_candidate=select).classify(
            'Nova Scotia beverage container deposit', include_candidates=True)
        self.assertEqual(result['category'], 'Deposits & Fees')
        self.assertEqual(result['level_2_category'], result['category'])
        self.assertEqual(result['status'], 'non_product')
        self.assertIsNone(result['code'])
        self.assertIsNone(result['confidence'])
        self.assertEqual(result['candidates'], [])
        db.execute.assert_not_called()
        describe.assert_not_called()
        embed.assert_not_called()
        select.assert_not_called()

    def test_non_deposit_products_do_not_match_charge_rule(self):
        for query in ('Bottle', 'Container deposit bag', 'Mineral deposit remover',
                      'Deposit', 'Tipping', 'Empty beverage container'):
            self.assertIsNone(receipt_charge_result(normalize_receipt_text(query)))


if __name__ == '__main__':
    unittest.main()
