import unittest
from types import SimpleNamespace
import torch
from spa_next.algorithms.losses import normalization_denominator, weighted_mean


class ThreeGPUChecks(unittest.TestCase):
    def test_padding_preserves_full_batch_gradient_and_incomplete_real_rows(self):
        mask = torch.ones(18, 3)
        mask[15:] = 0  # real trajectory 15 is incomplete; 16 and 17 are padding
        weights = mask / 3
        valid = torch.ones(18)
        valid[16:] = 0
        denominator, world = normalization_denominator(mask, weights, trajectory_valid=valid)
        self.assertEqual(denominator, 16)
        full = torch.tensor(.3, requires_grad=True)
        x = torch.arange(54).reshape(18, 3).float()
        expected = ((full*x[:15]).square().mean(-1)).sum()/16
        expected.backward()
        gradients = []
        for rank in range(3):
            p = torch.tensor(.3, requires_grad=True)
            for index in range(rank*6, (rank+1)*6):
                loss = weighted_mean((p*x[index:index+1]).square(), mask[index:index+1],
                    weights[index:index+1], global_denominator=denominator, world_size=3)
                loss.backward()
            gradients.append(p.grad)
        torch.testing.assert_close(torch.stack(gradients).mean(), full.grad)

    def test_worker_rpc_padding_is_removed_and_training_rows_masked(self):
        try:
            from verl import DataProto
        except ImportError:
            self.skipTest('verl required')
        from spa_next.training.verl_v1 import distributed_call
        data = DataProto.from_dict(tensors=dict(response_mask=torch.ones(16, 2),
            action_loss_weights=torch.full((16, 2), .5), advantages=torch.ones(16, 2)))
        trainer = SimpleNamespace(config=SimpleNamespace(algorithm=SimpleNamespace(adv_estimator='spa_v1')),
            actor_rollout_wg=None, _get_dp_size=lambda *args: 3)
        def inference(batch):
            self.assertEqual(len(batch), 18)
            return DataProto.from_dict(tensors=dict(old_log_probs=torch.ones(18, 2)))
        self.assertEqual(len(distributed_call(trainer, inference, data)), 16)
        def update(batch):
            self.assertEqual(len(batch), 18)
            self.assertEqual(batch.batch['spa_row_valid'].sum().item(), 16)
            self.assertEqual(batch.batch['response_mask'][16:].sum().item(), 0)
            self.assertEqual(batch.batch['action_loss_weights'][16:].sum().item(), 0)
            return 'updated'
        self.assertEqual(distributed_call(trainer, update, data, True), 'updated')
        self.assertEqual(len(data), 16)
        self.assertEqual(data.batch['response_mask'].sum().item(), 32)


if __name__ == '__main__':
    unittest.main()
