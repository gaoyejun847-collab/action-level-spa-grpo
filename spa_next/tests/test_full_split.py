import unittest
from spa_next.prepare_data import select_all_splits


class FullSplit(unittest.TestCase):
    def test_all_tasks_disjoint_and_reserved_test_held_out(self):
        reserved = {0, 12, 1000, 12086}
        split = select_all_splits(12087, reserved)
        train, val, test = (set(split[k]) for k in ('train', 'val', 'test'))
        self.assertEqual(tuple(map(len, (train, val, test))), (9668, 1209, 1210))
        self.assertEqual(train | val | test, set(range(12087)))
        self.assertFalse(train & val or train & test or val & test)
        self.assertTrue(reserved <= test)
        self.assertEqual(len(train) % 4, 0)
        self.assertEqual(split, select_all_splits(12087, reserved))
        self.assertNotEqual(split, select_all_splits(12087, reserved, seed=43))

    def test_invalid_reserved_ids_rejected(self):
        with self.assertRaises(ValueError):
            select_all_splits(12087, {-1})
        with self.assertRaises(ValueError):
            select_all_splits(12087, set(range(2000)))
