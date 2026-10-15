"""Cross-layer V1 contracts: patched dispatch, metadata and actual loss hooks."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from spa_next.training.patch_verl_v1 import PATCHERS, replace_once

HAS_TORCH = importlib.util.find_spec("torch") is not None


class PatchContracts(unittest.TestCase):
    def test_pinned_sources_compile_and_reject_drift(self):
        root = Path(__file__).resolve().parents[2] / ".spa-vendor/verl070/verl"
        if not root.is_dir():
            self.skipTest("Vendored reference wheel is not present")
        for relative, patcher in PATCHERS.items():
            original = (root / relative).read_text(encoding="utf-8")
            patched = patcher(original)
            ast.parse(patched)
            self.assertNotEqual(original, patched)
        with self.assertRaises(RuntimeError):
            replace_once("anchor anchor", "anchor", "changed")


@unittest.skipUnless(HAS_TORCH, "PyTorch runtime required")
class TensorIntegration(unittest.TestCase):
    def setUp(self):
        import torch
        from spa_next.training import verl_v1
        self.torch, self.integration = torch, verl_v1

    def test_full_batch_metadata_survives_reorder_and_padding(self):
        torch = self.torch
        with tempfile.TemporaryDirectory() as output:
            settings = SimpleNamespace(arm="B", audit_dir=output, lambda_spa=.2,
                gamma=1., epsilon=1e-6, std_threshold=1e-4, residual_limit=1.)
            mask = torch.tensor([[1, 1, 0], [1, 1, 1], [1, 1, 0], [1, 1, 1]])
            responses = torch.tensor([[20, 99, 0], [20, 21, 99], [30, 99, 0], [30, 31, 99]])
            traces = []
            for i in range(4):
                n = int(mask[i].sum())
                traces.append(dict(input_ids=[11, 12] + responses[i, :n].tolist(),
                    prompt_length=2, action_end_positions=[n + 1], task_id=5,
                    initial_state_id="fixed", policy_version={"step": 7},
                    termination_reason="terminated"))
            data = SimpleNamespace(batch={"response_mask": mask, "responses": responses,
                    "prompts": torch.tensor([[0, 11, 12]] * 4),
                    "attention_mask": torch.cat((torch.tensor([[0, 1, 1]] * 4), mask), 1),
                    "token_level_rewards": torch.tensor([[0., 1., 0.], [0., 0., 0.], [0., .5, 0.], [0., 0., 1.]])},
                non_tensor_batch={"uid": ["same"] * 4, "spa_v1_trace": traces,
                                  "action_ids": [[0] * int(row.sum()) for row in mask]}, meta_info={})
            self.integration.compute_advantage(data, SimpleNamespace(spa_v1=settings))
            self.assertTrue(torch.equal(data.batch["advantages"] != 0, mask.bool()))
            self.assertTrue(torch.allclose(data.batch["action_loss_weights"].sum(-1), torch.ones(4)))
            self.assertTrue((Path(output) / "step_000007.json").exists())
            self.assertTrue(torch.equal(data.batch["action_ids"], torch.tensor([[0, 0, -1], [0, 0, 0], [0, 0, -1], [0, 0, 0]])))
            # Frozen HF and original vLLM probabilities are audited separately;
            # irrelevant padded values do not affect discrepancy metrics.
            data.batch["old_log_probs"] = torch.zeros_like(mask, dtype=torch.float32, requires_grad=True)
            data.batch["rollout_log_probs"] = torch.where(mask.bool(), torch.full_like(mask, -.25, dtype=torch.float32), 100.)
            self.integration.compute_advantage(data, SimpleNamespace(spa_v1=settings))
            self.assertEqual(data.meta_info["spa_v1_metrics"]["spa/hf_vllm_logprob_abs_mean"], .25)
            self.assertEqual(data.meta_info["spa_v1_metrics"]["spa/hf_vllm_logprob_abs_max"], .25)
            self.assertIsNone(data.batch["old_log_probs"].grad)
            import json
            saved = json.loads((Path(output) / "step_000007.json").read_text())
            self.assertEqual(saved["frozen_hf_old_log_probs"][0], [0., 0., 0.])
            self.assertEqual(saved["original_vllm_rollout_log_probs"][0], [-.25, -.25, 100.])
            traces[2]["initial_state_id"] = "different"
            with self.assertRaisesRegex(ValueError, "Group mixes"):
                self.integration.compute_advantage(data, SimpleNamespace(spa_v1=settings))

    def test_actor_hooks_share_denominator_for_policy_and_kl(self):
        torch = self.torch
        from spa_next.algorithms.action_grpo import action_loss_weights
        for arm in ("A", "B", "C"):
            cfg = SimpleNamespace(policy_loss=SimpleNamespace(loss_mode="spa_v1_" + arm), clip_ratio=.2)
            mask = torch.tensor([[1, 1, 1], [1, 0, 0]])
            ids = torch.tensor([[0, 1, 1], [0, -1, -1]])
            weights = action_loss_weights(ids, mask)
            inputs = dict(response_mask=mask, action_loss_weights=weights,
                          advantages=torch.tensor([[1., -.3, -.3], [-1., 0., 0.]]),
                          ref_log_prob=torch.zeros(2, 3))
            logits = torch.tensor([[.1, -.1, .05], [-.1, 0., 0.]], requires_grad=True)
            normalizer = self.integration.minibatch_normalizer(SimpleNamespace(batch=inputs), cfg, "cpu")
            whole = self.integration.policy_loss(inputs, torch.zeros_like(logits), logits, cfg, normalizer)[0]
            whole = whole + .001 * self.integration.kl_loss(inputs, logits, cfg, normalizer)
            expected = torch.autograd.grad(whole, logits)[0]
            loss = 0
            for i in range(2):
                micro = {key: value[i:i+1] for key, value in inputs.items()}
                loss = loss + self.integration.policy_loss(micro, torch.zeros_like(logits[i:i+1]), logits[i:i+1], cfg, normalizer)[0]
                loss = loss + .001 * self.integration.kl_loss(micro, logits[i:i+1], cfg, normalizer)
            actual = torch.autograd.grad(loss, logits)[0]
            self.assertTrue(torch.allclose(expected, actual, atol=1e-8), arm)


if __name__ == "__main__":
    unittest.main()
