"""Directed numeric tests for the reviewed action-level SPA-GRPO contract."""

import math
import unittest

try:
    import torch
except ImportError:
    torch = None

from spa_next.agents.tokens import TokenTrace
if torch is not None:
    from spa_next.algorithms.action_grpo import (
        action_loss_weights, calibrate_returns, compute_action_advantages, pad_action_ids,
    )
    from spa_next.algorithms.losses import (
        clipped_policy_loss, low_variance_kl, normalization_denominator, weighted_mean,
    )


@unittest.skipIf(torch is None, "PyTorch is required; run in the training environment")
class ActionAdvantages(unittest.TestCase):
    def compute(self, rewards, ids, scores=None, reasons=None, **kwargs):
        ids = torch.tensor(ids, dtype=torch.long)
        return compute_action_advantages(torch.tensor(rewards, dtype=torch.float32), ids, ids.ge(0),
                                         ["same-task-and-state"] * len(rewards), scores, reasons, **kwargs)

    def test_calibration_hand_calculation_and_last_score_cancels(self):
        calibrated, returns, residual = calibrate_returns([0.2, 0.3, 0.1], 1.0)
        self.assertEqual(calibrated, [0.2, 0.3, 0.5])
        self.assertEqual(returns, [1.0, 0.8, 0.5])
        self.assertAlmostEqual(residual, 0.4)
        changed, changed_returns, _ = calibrate_returns([0.2, 0.3, 100.0], 1.0)
        self.assertEqual(changed, calibrated)
        self.assertEqual(changed_returns, returns)

    def test_signed_calibration_is_not_clipped(self):
        calibrated, returns, _ = calibrate_returns([1.2, -0.2], 0.1)
        self.assertAlmostEqual(sum(calibrated), 0.1)
        self.assertLess(calibrated[-1], 0)
        self.assertAlmostEqual(returns[0], 0.1)
        self.assertAlmostEqual(returns[1], -1.1)

    def test_sample_standard_deviation_and_first_action_equality(self):
        result = self.compute([0, 1], [[0, 0, 1], [0, 1, 1]], [[0.1, 0.1], [0.2, 0.3]])
        expected = 0.5 / (math.sqrt(0.5) + 1e-6)
        self.assertAlmostEqual(result["outcome_advantages"][0].item(), -expected, places=6)
        self.assertAlmostEqual(result["outcome_advantages"][1].item(), expected, places=6)
        for row in result["audit"]:
            self.assertEqual(row["action_returns"][0], row["outcome_reward"])
            self.assertAlmostEqual(row["spa_advantages"][0], row["outcome_advantage"])
            self.assertAlmostEqual(row["mixed_advantages"][0], row["outcome_advantage"])

    def test_nontrivial_four_trajectory_mixture_matches_formula(self):
        rewards = [0.0, 0.25, 0.5, 1.0]
        prefixes = [0.1, 0.2, -0.1, 0.8]
        result = self.compute(rewards, [[0, 1]] * 4, [[value, 0.0] for value in prefixes])
        returns = [reward - prefix for reward, prefix in zip(rewards, prefixes)]
        mean_r, mean_g = sum(rewards) / 4, sum(returns) / 4
        std_r = math.sqrt(sum((reward - mean_r) ** 2 for reward in rewards) / 3)
        std_g = math.sqrt(sum((value - mean_g) ** 2 for value in returns) / 3)
        for index, row in enumerate(result["audit"]):
            outcome = (rewards[index] - mean_r) / (std_r + 1e-6)
            spa = (returns[index] - mean_g) / (std_g + 1e-6)
            expected = 0.8 * outcome + 0.2 * spa
            self.assertAlmostEqual(row["mixed_advantages"][1], expected, places=6)
            self.assertAlmostEqual(result["advantages"][index, 1].item(), expected, places=6)

    def test_same_reward_disables_entire_group(self):
        result = self.compute([0.5, 0.5], [[0, 1], [0, 1]], [[0.1, 0.1], [0.4, 0.1]])
        self.assertTrue(torch.equal(result["advantages"], torch.zeros(2, 2)))
        for row in result["audit"]:
            self.assertIn("outcome_zero_variance", row["group_fallback_reasons"])
            self.assertEqual(row["mixing_coefficients"], [0, 0])

    def test_one_trajectory_group_has_zero_advantage(self):
        result = self.compute([1], [[0, 1]], [[0.2, 0.4]])
        self.assertTrue(torch.equal(result["advantages"], torch.zeros(1, 2)))

    def test_depth_uses_only_existing_actions_and_singleton_falls_back(self):
        result = self.compute([1, 0], [[0, 1, 2], [0, 1, -1]], [[0.2, 0.3, 0.1], [-0.1, 0.1]])
        first = result["audit"][0]
        self.assertEqual(first["depth_fallback_reasons"][2], "single_sample_depth")
        self.assertEqual(first["mixing_coefficients"], [0.2, 0.2, 0.0])
        self.assertAlmostEqual(first["mixed_advantages"][2], first["outcome_advantage"])
        self.assertEqual(result["advantages"][1, 2].item(), 0)
        self.assertEqual(result["returns"][1, 2].item(), 0)
        self.assertEqual(result["action_loss_weights"][1, 2].item(), 0)

    def test_zero_depth_variance_restores_full_outcome_not_point_eight(self):
        result = self.compute([1, 0], [[0, 1], [0, 1]], [[0.5, 0.2], [-0.5, 0.2]])
        for row in result["audit"]:
            self.assertEqual(row["depth_fallback_reasons"][1], "depth_zero_variance")
            self.assertEqual(row["mixing_coefficients"][1], 0)
            self.assertAlmostEqual(row["mixed_advantages"][1], row["outcome_advantage"])

    def test_invalid_prm_whole_group_fallback(self):
        cases = [
            (None, "missing_prm_scores"),
            ([float("nan"), 0], "nonfinite_prm_scores"),
            ([float("inf"), 0], "nonfinite_prm_scores"),
            ([0.2], "prm_score_count_mismatch"),
            ([-2.0, 0], "calibration_residual_exceeded"),
        ]
        for scores, reason in cases:
            with self.subTest(reason=reason):
                result = self.compute([1, 0], [[0, 1], [0, 1]], [scores, [0, 0]])
                for row in result["audit"]:
                    self.assertIn(reason, row["group_fallback_reasons"])
                    self.assertEqual(row["mixing_coefficients"], [0, 0])
                    self.assertEqual(row["mixed_advantages"], [row["outcome_advantage"]] * 2)

    def test_residual_exact_limit_is_allowed(self):
        result = self.compute([1, 0], [[0], [0]], [[0], [0]])
        self.assertEqual(result["metrics"]["spa/group_fallback_rate"], 0)

    def test_incomplete_and_context_endings_fallback_whole_group(self):
        for reason in ["incomplete_action", "context_budget"]:
            result = self.compute([1, 0], [[0, 1], [0, 1]], [[0.1, 0.2], [0, 0]], [reason, "terminated"])
            self.assertEqual(result["metrics"]["spa/group_fallback_rate"], 1)
            for row in result["audit"]:
                self.assertIn(reason, row["group_fallback_reasons"])
                self.assertEqual(row["mixed_advantages"], [row["outcome_advantage"]] * 2)

    def test_max_steps_is_complete_not_automatic_fallback(self):
        result = self.compute([1, 0], [[0, 1], [0, 1]], [[0.1, 0.2], [0, 0]], ["terminated", "max_steps"])
        self.assertEqual(result["metrics"]["spa/group_fallback_rate"], 0)

    def test_missing_model_is_not_modelled_as_silent_lambda_zero(self):
        # Service/version failures are raised by the scoring client upstream;
        # unknown/infra trajectory status is likewise refused by the math layer.
        with self.assertRaisesRegex(ValueError, "Infrastructure"):
            self.compute([1, 0], [[0], [0]], [[0], [0]], ["service_unavailable", "terminated"])

    def test_disabled_spa_needs_no_scores_and_lambda_zero_is_outcome(self):
        outcome = self.compute([1, 0], [[0, 1], [0, 1]], spa_enabled=False)
        zero = self.compute([1, 0], [[0, 1], [0, 1]], [[0.1, 0.2], [0, 0]], lambda_spa=0)
        torch.testing.assert_close(outcome["advantages"], zero["advantages"])
        self.assertEqual(outcome["metrics"]["spa/group_fallback_rate"], 0)

    def test_score_and_advantage_outputs_are_detached(self):
        rewards = torch.tensor([1.0, 0.0], requires_grad=True)
        scores = [torch.tensor([0.1, 0.2], requires_grad=True), torch.tensor([0.0, 0.1], requires_grad=True)]
        ids = torch.tensor([[0, 1], [0, 1]])
        result = compute_action_advantages(rewards, ids, torch.ones(2, 2), ["a", "a"], scores)
        for key in ["advantages", "returns", "action_loss_weights", "outcome_advantages"]:
            self.assertFalse(result[key].requires_grad)

    def test_groups_are_independent_and_reordering_is_equivariant(self):
        ids = torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]])
        rewards = torch.tensor([0.0, 1.0, 0.2, 0.8])
        scores = [[0.0, 0.0], [0.2, 0.3], [0.1, 0.0], [0.1, 0.1]]
        groups = ["a", "a", "b", "b"]
        original = compute_action_advantages(rewards, ids, ids.ge(0), groups, scores)
        order = torch.tensor([2, 0, 3, 1])
        shuffled = compute_action_advantages(rewards[order], ids[order], ids[order].ge(0),
                                             [groups[i] for i in order.tolist()], [scores[i] for i in order.tolist()])
        torch.testing.assert_close(shuffled["advantages"], original["advantages"][order])
        with self.assertRaisesRegex(ValueError, "Incomplete rollout group"):
            compute_action_advantages(rewards, ids, ids.ge(0), groups, scores, expected_group_size=4)


@unittest.skipIf(torch is None, "PyTorch is required; run in the training environment")
class ActionMasksAndLoss(unittest.TestCase):
    def test_generation_calls_determine_actions_not_thought_markers(self):
        trace = TokenTrace([1, 2, 3], 300)
        text = "Thought: first\nThought: second\nAction: click[buy]"
        generated = list(text.encode()) + [151645]
        trace.append(generated, action_id=0, train=True, logprobs=[-1.0] * len(generated))
        trace.append([9, 9])  # Observation tokens.
        trace.append([8, 8, 8])  # Incomplete subsequent generation.
        mask = torch.tensor([trace.mask + [0, 0]])
        ids = pad_action_ids([trace.action_ids], mask)
        weights = action_loss_weights(ids, mask)
        self.assertEqual(ids.max().item(), 0)
        self.assertAlmostEqual(weights.sum().item(), 1, places=6)
        self.assertTrue(torch.equal(weights[0, len(generated):], torch.zeros(7)))
        values = torch.arange(ids.shape[1], dtype=torch.float32, requires_grad=True)
        weighted_mean(values.unsqueeze(0), mask, weights).backward()
        self.assertTrue(torch.equal(values.grad[len(generated):], torch.zeros(7)))

    def test_weights_padding_reordering_and_split_preserve_semantics(self):
        ids = torch.tensor([[0, 0, -1, 1, -1], [0, 1, 1, 1, 1]])
        mask = ids.ge(0)
        weights = action_loss_weights(ids, mask)
        torch.testing.assert_close(weights, torch.tensor([[0.25, 0.25, 0, 0.5, 0], [0.5, 0.125, 0.125, 0.125, 0.125]]))
        padded_ids = torch.cat([ids, torch.full((2, 5), -1)], -1)
        torch.testing.assert_close(action_loss_weights(padded_ids, padded_ids.ge(0))[:, :5], weights)
        torch.testing.assert_close(action_loss_weights(ids.flip(0), mask.flip(0)), weights.flip(0))
        for index in range(2):
            torch.testing.assert_close(action_loss_weights(ids[index:index + 1], mask[index:index + 1]), weights[index:index + 1])

    def test_mask_drift_and_split_action_are_rejected(self):
        for raw, mask in [([[0, -1]], [[1, 1]]), ([[0, 0]], [[1, 0]]), ([[0, -1, 0]], [[1, 0, 1]]), ([[1, 1]], [[1, 1]])]:
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    pad_action_ids(raw, torch.tensor(mask))

    def test_first_action_incomplete_preserves_group_and_zero_loss(self):
        ids = pad_action_ids([[-1, -1], [0, 0]], torch.tensor([[0, 0], [1, 1]]))
        mask = ids.ge(0)
        result = compute_action_advantages(torch.tensor([0.0, 1.0]), ids, mask, ["same", "same"],
                                           [None, [0.5]], ["incomplete_action", "terminated"])
        self.assertEqual(result["audit"][0]["num_actions"], 0)
        self.assertEqual(result["audit"][0]["mixed_advantages"], [])
        self.assertIn("incomplete_action", result["audit"][1]["group_fallback_reasons"])
        self.assertTrue(torch.equal(result["action_loss_weights"][0], torch.zeros(2)))
        denominator, _ = normalization_denominator(mask, result["action_loss_weights"])
        self.assertEqual(denominator, 2)
        values = torch.tensor([[999.0, 999.0], [2.0, 2.0]], requires_grad=True)
        loss = weighted_mean(values, mask, result["action_loss_weights"])
        self.assertEqual(loss.item(), 1)
        loss.backward()
        torch.testing.assert_close(values.grad[0], torch.zeros(2))

    def test_whole_batch_incomplete_blocks_empty_optimizer_step_all_arms(self):
        ids = torch.full((2, 3), -1)
        mask = ids.ge(0)
        result = compute_action_advantages(torch.zeros(2), ids, mask, ["a", "a"],
                                           [None, None], ["incomplete_action", "incomplete_action"])
        self.assertTrue(torch.equal(result["advantages"], torch.zeros(2, 3)))
        for weights in [None, result["action_loss_weights"]]:
            with self.assertRaisesRegex(ValueError, "complete action token"):
                normalization_denominator(mask, weights)

    def test_standard_clip_without_dual_clip_and_detached_old_policy(self):
        old = torch.zeros(1, 2, requires_grad=True)
        current = torch.log(torch.tensor([[10.0, 2.0]])).requires_grad_()
        advantage = torch.tensor([[-1.0, 1.0]], requires_grad=True)
        loss, _ = clipped_policy_loss(old, current, advantage, torch.ones(1, 2))
        self.assertAlmostEqual(loss.item(), (10.0 - 1.2) / 2, places=6)
        loss.backward()
        self.assertIsNone(old.grad)
        self.assertIsNone(advantage.grad)
        torch.testing.assert_close(current.grad, torch.tensor([[5.0, 0.0]]))

    def test_policy_and_kl_use_identical_hand_computed_action_weights(self):
        ids = torch.tensor([[0, 0, 1, -1], [0, 1, 1, 1]])
        mask = ids.ge(0)
        weights = action_loss_weights(ids, mask)
        ratio = torch.tensor([[1.1, 0.9, 1.3, 100], [0.7, 1.0, 1.1, 1.2]])
        current = ratio.log()
        advantage = torch.tensor([[1.0, 1.0, -1.0, 300], [-0.5, 0.5, 0.5, 0.5]])
        loss, _ = clipped_policy_loss(torch.zeros(2, 4), current, advantage, mask, weights)
        token_loss = [[-1.1, -0.9, 1.3], [0.4, -0.5, -0.55, -0.6]]
        hand = (((token_loss[0][0] + token_loss[0][1]) / 2 + token_loss[0][2]) / 2
                + (token_loss[1][0] + sum(token_loss[1][1:]) / 3) / 2) / 2
        self.assertAlmostEqual(loss.item(), hand, places=6)
        kl = low_variance_kl(current, torch.zeros(2, 4))
        penalty = weighted_mean(kl, mask, weights)
        hand_kl = 0.0
        for row in range(2):
            for col in range(4):
                if mask[row, col]:
                    d = max(-20, min(20, -float(current[row, col])))
                    hand_kl += min(10, max(0, math.exp(d) - d - 1)) * float(weights[row, col]) / 2
        self.assertAlmostEqual(penalty.item(), hand_kl, places=6)
        clipped = low_variance_kl(torch.tensor([[-30.0, 30.0, 0.0]]), torch.zeros(1, 3))
        torch.testing.assert_close(clipped, torch.tensor([[10.0, 10.0, 0.0]]))

    def test_reference_has_no_gradient_and_mask_excludes_padding(self):
        current = torch.tensor([[0.1, 0.2, 100.0]], requires_grad=True)
        reference = torch.zeros(1, 3, requires_grad=True)
        weighted_mean(low_variance_kl(current, reference), torch.tensor([[1, 1, 0]])).backward()
        self.assertIsNone(reference.grad)
        self.assertEqual(current.grad[0, 2].item(), 0)

    def test_microbatch_and_uneven_distributed_gradients_equal_whole_batch(self):
        ids = torch.tensor([[0, 0, 1, -1, -1], [0, 1, 1, 1, -1], [0, 0, 0, 1, 1], [0, -1, -1, -1, -1]])
        mask = ids.ge(0)
        weights = action_loss_weights(ids, mask)
        feature = torch.tensor([[0.2, 0.3, -0.4, 10, 20], [-0.5, 0.3, 0.4, 0.2, 10],
                                [0.4, 0.4, 0.4, 0.6, 0.6], [-0.3, 20, 20, 20, 20]])
        advantage = torch.tensor([[1, 1, 0.5, 0, 0], [-0.4, 0.7, 0.7, 0.7, 0],
                                  [0.2, 0.2, 0.2, -0.3, -0.3], [-0.5, 0, 0, 0, 0]])
        for action_balanced in [False, True]:
            with self.subTest(action_balanced=action_balanced):
                actual_weights = weights if action_balanced else None
                denominator, _ = normalization_denominator(mask, actual_weights)

                def make_loss(parameter, rows, world_size=1, scale=1):
                    old = torch.full_like(feature[rows], -2)
                    current = old + parameter * feature[rows]
                    w = actual_weights[rows] if action_balanced else None
                    norm = dict(global_denominator=denominator, world_size=world_size, loss_scale_factor=scale)
                    pg, _ = clipped_policy_loss(old, current, advantage[rows], mask[rows], w, **norm)
                    kl = weighted_mean(low_variance_kl(current, old), mask[rows], w, **norm)
                    return pg + 0.001 * kl

                whole = torch.tensor(0.1, requires_grad=True)
                make_loss(whole, [0, 1, 2, 3]).backward()
                micro = torch.tensor(0.1, requires_grad=True)
                for rows in [[0], [1, 2], [3]]:
                    # Also verify cancellation of an existing trainer's scale.
                    (make_loss(micro, rows, scale=0.25) * 0.25).backward()
                torch.testing.assert_close(micro.grad, whole.grad, atol=1e-7, rtol=1e-6)
                gradients = []
                for rank_rows in [[0], [1, 2, 3]]:
                    parameter = torch.tensor(0.1, requires_grad=True)
                    for row in rank_rows:
                        make_loss(parameter, [row], world_size=2).backward()
                    gradients.append(parameter.grad)
                torch.testing.assert_close(sum(gradients) / 2, whole.grad, atol=1e-7, rtol=1e-6)

    def test_action_mean_differs_from_token_mean_even_with_outcome_advantage(self):
        ids = torch.tensor([[0, 0, 0, 1]])
        values = torch.tensor([[1.0, 1.0, 1.0, 3.0]])
        mask = ids.ge(0)
        self.assertEqual(weighted_mean(values, mask).item(), 1.5)
        self.assertEqual(weighted_mean(values, mask, action_loss_weights(ids, mask)).item(), 2.0)


if __name__ == "__main__":
    unittest.main()
