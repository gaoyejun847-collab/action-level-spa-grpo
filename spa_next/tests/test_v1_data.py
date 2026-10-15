import unittest

from spa_next.training.prepare_v1 import select_rows


def row(task, name, price):
    return dict(extra_info={'task_id': task},
                prompt=[{'content': f'Instruction: [SEP] Find {name} under {price} dollars [SEP]'}])


class V1DataTests(unittest.TestCase):
    def test_budget_accepts_rollout_indices_and_rejects_duplicates(self):
        from spa_next.training.report_v1 import audit_budget
        records = [dict(task_id=task, episode_id=f'{task}-{sample}',
                        initial_state_id=str(task), model='base', sft_version='sft',
                        actor_version=dict(experiment='C'),
                        policy_version=dict(step=task // 4 + 1, rollout_n=sample, validate=False))
                   for task in range(512) for sample in range(4)]
        result = audit_budget(records, list(range(512)), arm='C', sft_version='sft')
        self.assertEqual(result['training_trajectories'], 2048)
        records[0]['policy_version']['rollout_n'] = 1
        with self.assertRaises(ValueError):
            audit_budget(records, list(range(512)), arm='C', sft_version='sft')

    def test_price_variants_are_excluded_and_order_is_shared(self):
        splits = dict(train=[row(1, 'shirt', 10), row(2, 'shoe', 10),
                             row(3, 'hat', 10), row(4, 'bag', 10), row(5, 'sock', 10)],
                      val=[row(6, 'shirt', 20)], test=[row(7, 'shoe', 30)])
        selected, eligible = select_rows(splits, train_count=3, val_count=1)
        self.assertEqual(eligible, 3)
        self.assertEqual({r['extra_info']['task_id'] for r in selected['train']}, {3, 4, 5})
        self.assertEqual(selected, select_rows(splits, train_count=3, val_count=1)[0])
        self.assertEqual(selected['val_full'], splits['val'])

    def test_insufficient_isolated_data_stops(self):
        splits = dict(train=[row(1, 'shirt', 10)], val=[row(2, 'shirt', 20)], test=[])
        with self.assertRaises(ValueError):
            select_rows(splits, train_count=1, val_count=1)


if __name__ == '__main__':
    unittest.main()
