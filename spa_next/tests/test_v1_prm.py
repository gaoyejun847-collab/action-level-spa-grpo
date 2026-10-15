"""PRM data leakage, token boundaries, signed scoring, and deployment gates."""
import copy
import inspect
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from spa_next.prm.client import PRMClient, PRMServiceError
from spa_next.prm.data import JsonlDataset, eligible_task_families, extract_record, split_families

try:
    import torch
except ImportError:
    torch = None


def trace_fixture():
    # Two complete replies; arbitrary repeated Thought text never enters boundary
    # detection. The gap is an observation with no PRM target or policy mask.
    return dict(task_id=7, initial_state_id='fixed-state', episode_id='episode',
                policy_version=dict(step=4, validate=False), termination_reason='terminated',
                prompt_ids=[1, 2], response_ids=[3, 9, 4, 5, 6, 9],
                response_mask=[1, 1, 0, 0, 1, 1], action_ids=[0, 0, -1, -1, 1, 1],
                outcome_reward=1., steps=[
                    dict(complete=True, context_token_ids=[1, 2], generated_token_ids=[3, 9],
                         action_span=[0, 2], error_type=None),
                    dict(complete=True, context_token_ids=[1, 2, 3, 9, 4, 5], generated_token_ids=[6, 9],
                         action_span=[4, 6], error_type='invalid_action')])


class PRMDataTest(unittest.TestCase):
    def test_original_boundaries_and_reward_separate(self):
        row, reason = extract_record(trace_fixture(), {7: 'family'}, eos_token_id=9)
        self.assertIsNone(reason)
        self.assertEqual(row['action_end_positions'], [3, 7])
        self.assertEqual(row['input_ids'], [1, 2, 3, 9, 4, 5, 6, 9])
        changed = trace_fixture()
        changed['outcome_reward'] = 0.
        other, _ = extract_record(changed, {7: 'family'}, eos_token_id=9)
        self.assertEqual(row['trajectory_sha256'], other['trajectory_sha256'])
        self.assertEqual(row['input_ids'], other['input_ids'])

    def test_heldout_family_and_validation_rollouts_rejected(self):
        self.assertEqual(extract_record(trace_fixture(), {})[1], 'heldout_or_unmatched_task_family')
        trace = trace_fixture()
        trace['policy_version']['validate'] = True
        self.assertEqual(extract_record(trace, {7: 'f'})[1], 'not_training_rollout')

    def test_partial_context_and_infrastructure_rejected(self):
        for termination in ('incomplete_action', 'context_budget', 'infrastructure_error'):
            trace = trace_fixture()
            trace['termination_reason'] = termination
            self.assertIsNone(extract_record(trace, {7: 'f'})[0])
        trace = trace_fixture()
        trace['steps'][-1]['complete'] = False
        self.assertEqual(extract_record(trace, {7: 'f'})[1], 'incomplete_action')
        trace = trace_fixture()
        trace['steps'][0]['error_type'] = 'network_failure'
        self.assertEqual(extract_record(trace, {7: 'f'})[1], 'infrastructure_error')

    def test_bad_masks_or_retokenization_rejected(self):
        trace = trace_fixture()
        trace['response_mask'][2] = 1
        self.assertEqual(extract_record(trace, {7: 'f'})[1], 'invalid_action_mask')
        trace = trace_fixture()
        trace['steps'][1]['context_token_ids'][0] = 999
        self.assertEqual(extract_record(trace, {7: 'f'})[1], 'retokenized_or_mismatched_trace')
        self.assertEqual(extract_record(trace_fixture(), {7: 'f'}, max_length=7)[1], 'over_context_limit')
        self.assertEqual(extract_record(trace_fixture(), {7: 'f'}, eos_token_id=10)[1], 'missing_eos')

    def test_complete_ten_step_budget_kept(self):
        trace = trace_fixture()
        trace.update(termination_reason='max_steps', response_ids=[], response_mask=[], action_ids=[], steps=[])
        for index in range(10):
            start = len(trace['response_ids'])
            context = trace['prompt_ids'] + trace['response_ids']
            trace['response_ids'].extend([3, 9])
            trace['response_mask'].extend([1, 1])
            trace['action_ids'].extend([index, index])
            trace['steps'].append(dict(complete=True, context_token_ids=context, generated_token_ids=[3, 9],
                                       action_span=[start, start + 2], error_type=None))
        row, reason = extract_record(trace, {7: 'f'}, eos_token_id=9)
        self.assertIsNone(reason)
        self.assertEqual(len(row['action_end_positions']), 10)

    def test_family_split_deterministic_and_disjoint(self):
        families = ['family-' + str(i) for i in range(100)]
        train, val = split_families(families)
        self.assertEqual((len(train), len(val)), (90, 10))
        self.assertFalse(train & val)
        self.assertEqual((train, val), split_families(list(reversed(families))))

    def test_price_variant_of_heldout_task_is_excluded(self):
        def row(task, goal):
            return dict(extra_info=dict(task_id=task), prompt=[dict(content='Instruction: [SEP] ' + goal + ' [SEP]')])
        eligible, _ = eligible_task_families(dict(
            train=[row(1, 'red shoes less than 40 dollars'), row(2, 'blue shoes less than 50 dollars'),
                   row(3, 'red coat less than 100 dollars')],
            val=[row(101, 'red shoes less than 90 dollars')],
            test=[row(102, 'red coat less than 80 dollars')]))
        self.assertEqual(set(eligible), {2})

    def test_offset_dataset(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'data.jsonl'
            path.write_text('\n'.join(json.dumps(dict(input_ids=[i])) for i in range(3)) + '\n')
            data = JsonlDataset(path)
            self.assertEqual(len(data), 3)
            self.assertEqual(data[2]['input_ids'], [2])


class PRMClientTest(unittest.TestCase):
    def test_version_service_failure_stops(self):
        client = PRMClient('http://127.0.0.1:1', 'expected', timeout=.01)
        with self.assertRaises(PRMServiceError):
            client.health()

    def test_reward_cannot_be_sent(self):
        client = PRMClient('http://localhost', 'expected')
        with self.assertRaises(ValueError):
            client.score([dict(input_ids=[1], action_end_positions=[0], reward=1.)])

    def test_missing_nonfinite_score_is_explicit_invalid(self):
        client = PRMClient('http://localhost', 'expected')
        with patch.object(client, '_request', return_value={'results': [dict(valid=True, prm_scores=[float('nan')])]}):
            result = client.score([dict(input_ids=[1], action_end_positions=[0])])
        self.assertFalse(result[0]['valid'])
        self.assertEqual(result[0]['prm_scores'], [])


@unittest.skipIf(torch is None, 'PyTorch model checks run in the server training environment')
class PRMModelTest(unittest.TestCase):
    def make_model(self):
        from types import SimpleNamespace
        from spa_next.prm.model import QwenSPAPRM
        class CausalBackbone(torch.nn.Module):
            config = SimpleNamespace(vocab_size=4)
            def forward(self, input_ids, attention_mask=None, logits_to_keep=None, use_cache=False):
                prefixes = input_ids.float().cumsum(dim=-1)
                values = prefixes[:, logits_to_keep, None].expand(-1, -1, 4)
                return SimpleNamespace(logits=values)
        return QwenSPAPRM(CausalBackbone())

    def test_zero_head_signed_scores_and_no_reward_argument(self):
        model = self.make_model()
        ids = torch.tensor([[1, 2, 3, 4]])
        scores = model(ids, [1, 3])
        self.assertEqual(scores.tolist(), [0., 0.])
        model.head.weight.data.fill_(-1.)
        self.assertTrue(bool(torch.all(model(ids, [1, 3]) < 0)))
        self.assertNotIn('reward', inspect.signature(model.forward).parameters)
        self.assertEqual(scores.dtype, torch.float32)

    def test_endpoint_does_not_see_future_feedback(self):
        model = self.make_model()
        model.head.weight.data.fill_(.25)
        initial = model(torch.tensor([[1, 2, 3, 4]]), [1, 3])
        changed = model(torch.tensor([[1, 2, 99, 88]]), [1, 3])
        self.assertEqual(float(initial[0]), float(changed[0]))
        self.assertNotEqual(float(initial[1]), float(changed[1]))

    def test_sum_mse_trains_head(self):
        model = self.make_model()
        scores = model(torch.tensor([[1, 2, 3, 4]]), [1, 3])
        (scores.sum() - 1.).square().backward()
        self.assertTrue(bool(torch.all(model.head.weight.grad != 0)))

    def test_quantized_path_never_receives_future_tokens(self):
        from spa_next.prm.model import score_record
        model = self.make_model()
        model.strict_prefix_scoring = True
        calls = []
        original_forward = model.forward
        def record_forward(input_ids, action_end_positions, attention_mask=None):
            calls.append(input_ids[0].tolist())
            return original_forward(input_ids, action_end_positions, attention_mask)
        model.forward = record_forward
        score_record(model, dict(input_ids=[1, 2, 3, 0], action_end_positions=[1, 3]), 'cpu')
        self.assertEqual(calls, [[1, 2], [1, 2, 3, 0]])

    def test_quantization_gate(self):
        from spa_next.prm.deployment import compare_scores
        self.assertTrue(compare_scores([[.2, .3], [.1]], [[.201, .299], [.1]], [.5, .2])['gate_passed'])
        self.assertFalse(compare_scores([[.2, .3]], [[.3, .4]], [.5])['gate_passed'])
        self.assertFalse(compare_scores([[.2]], [[float('nan')]], [.2])['gate_passed'])

    def test_real_qwen_selected_logits_and_dynamic_int8_prefix(self):
        try:
            from transformers import Qwen3Config, Qwen3ForCausalLM
        except ImportError:
            self.skipTest('Pinned Transformers Qwen3 is required')
        from spa_next.prm.model import QwenSPAPRM, score_record
        torch.manual_seed(42)
        config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                             head_dim=8, max_position_embeddings=32)
        backbone = Qwen3ForCausalLM(config).eval()
        ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
        with torch.inference_mode():
            full = backbone(ids).logits
            selected = backbone(ids, logits_to_keep=torch.tensor([2, 5])).logits
        torch.testing.assert_close(selected, full[:, [2, 5]])
        if 'fbgemm' not in torch.backends.quantized.supported_engines:
            self.skipTest('fbgemm dynamic INT8 backend is required')
        torch.backends.quantized.engine = 'fbgemm'
        backbone = torch.ao.quantization.quantize_dynamic(backbone, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
        model = QwenSPAPRM(backbone)
        model.head.weight.data.normal_(std=.1)
        model.strict_prefix_scoring = True
        with torch.inference_mode():
            original = score_record(model, dict(input_ids=[1, 2, 3, 4, 5, 6], action_end_positions=[2, 5]), 'cpu')
            changed = score_record(model, dict(input_ids=[1, 2, 3, 62, 61, 60], action_end_positions=[2, 5]), 'cpu')
        self.assertEqual(float(original[0]), float(changed[0]))
        self.assertEqual(len(original), 2)
        self.assertTrue(bool(torch.all(torch.isfinite(original))))

    def test_partial_global_batch_accumulation_matches_mean_gradient(self):
        self._check_batch_gradient([1., 2., 5., 8., 9.], 2, 8)

    def test_three_gpu_batch16_gradient(self):
        self._check_batch_gradient([float(i) for i in range(1, 17)], 3, 18)
        self._check_batch_gradient([float(i) for i in range(1, 12)], 3, 18)

    def _check_batch_gradient(self, examples, world, slots):
        full = torch.tensor(.4, requires_grad=True)
        sum((full * x - 1.).square() for x in examples).div(len(examples)).backward()
        accumulated = []
        padded = examples + [examples[0]] * (slots - len(examples))
        for rank in range(world):
            parameter = torch.tensor(.4, requires_grad=True)
            for index in range(rank, slots, world):
                loss = (parameter * padded[index] - 1.).square()
                (loss * (world / len(examples) if index < len(examples) else 0.)).backward()
            accumulated.append(parameter.grad)
        torch.testing.assert_close(torch.stack(accumulated).mean(), full.grad)

    def test_precision_reference_keeps_selected_lora_unmerged(self):
        try:
            from peft import LoraConfig, get_peft_model
            from transformers import Qwen3Config, Qwen3ForCausalLM
        except ImportError:
            self.skipTest('Pinned Qwen3 and PEFT are required')
        from spa_next.prm.data import sha256_file
        from spa_next.prm.deployment import load_training_reference
        from spa_next.prm.model import QwenSPAPRM
        config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                             head_dim=8, max_position_embeddings=32)
        with tempfile.TemporaryDirectory() as directory:
            base, checkpoint = Path(directory) / 'sft-base', Path(directory) / 'best-prm'
            backbone = Qwen3ForCausalLM(config).to(dtype=torch.bfloat16).eval()
            backbone.save_pretrained(base)
            snapshot = dict(kind='merged_sft_base', verified=True,
                            files={p.name: sha256_file(p) for p in base.iterdir() if p.is_file()})
            (base / 'spa_snapshot.json').write_text(json.dumps(snapshot))
            # Match training's from_pretrained path. Calling .to(BF16) on a
            # freshly constructed model also casts nonpersistent rotary buffers,
            # unlike HF's BF16 checkpoint loader, and is a different reference.
            backbone = Qwen3ForCausalLM.from_pretrained(base, local_files_only=True,
                torch_dtype=torch.bfloat16, attn_implementation='sdpa', device_map={'': 'cpu'})
            backbone = get_peft_model(backbone, LoraConfig(r=32, lora_alpha=64,
                lora_dropout=0., target_modules='all-linear', task_type='CAUSAL_LM'))
            for name, parameter in backbone.named_parameters():
                if 'lora_B' in name:
                    parameter.data.fill_(.01)
            model = QwenSPAPRM(backbone).eval()
            model.head.weight.data.fill_(.01)
            model.head.bias.data.fill_(.2)
            backbone.save_pretrained(checkpoint / 'adapter')
            torch.save(model.head.state_dict(), checkpoint / 'head.pt')
            manifest = dict(base=str(base), base_manifest_sha256=sha256_file(base / 'spa_snapshot.json'),
                            training_reference=dict(kind='unmerged-bf16-sft-base-plus-prm-lora-fp32-head',
                                checkpoint=str(checkpoint), files={str(p.relative_to(checkpoint)): sha256_file(p)
                                                                  for p in checkpoint.rglob('*') if p.is_file()}))
            restored = load_training_reference(manifest, 'cpu')
            self.assertTrue(hasattr(restored.backbone, 'peft_config'))
            self.assertEqual(restored.head.weight.dtype, torch.float32)
            self.assertTrue(all(not parameter.requires_grad for parameter in restored.parameters()))
            restored_parameters = dict(restored.named_parameters())
            for name, parameter in model.named_parameters():
                torch.testing.assert_close(restored_parameters[name], parameter, rtol=0, atol=0, msg=name)
            self.assertEqual(model.backbone.config._attn_implementation, restored.backbone.config._attn_implementation)
            with torch.inference_mode():
                ids = torch.tensor([[1, 2, 3]])
                torch.testing.assert_close(restored(ids, [2]), model(ids, [2]), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
