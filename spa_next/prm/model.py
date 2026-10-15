"""Signed action-increment PRM: a linear head on endpoint vocabulary logits."""
import torch
from torch import nn


def validate_inputs(input_ids, action_end_positions, max_length=4096, vocab_size=None):
    if not isinstance(input_ids, list) or not input_ids or len(input_ids) > max_length:
        raise ValueError('input_ids must be a nonempty, untruncated trajectory within the context limit')
    if any(type(token) is not int or token < 0 or (vocab_size is not None and token >= vocab_size) for token in input_ids):
        raise ValueError('Invalid original token IDs')
    if not isinstance(action_end_positions, list) or not action_end_positions:
        raise ValueError('At least one complete assistant action is required')
    if any(type(pos) is not int or pos < 0 or pos >= len(input_ids) for pos in action_end_positions):
        raise ValueError('Invalid action endpoint')
    if action_end_positions != sorted(set(action_end_positions)):
        raise ValueError('Action endpoints must be unique and increasing')


class QwenSPAPRM(nn.Module):
    def __init__(self, backbone, head=None):
        super().__init__()
        self.backbone = backbone
        self.head = head if head is not None else nn.Linear(backbone.config.vocab_size, 1, dtype=torch.float32)
        if head is None:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(self, input_ids, action_end_positions, attention_mask=None):
        # There is deliberately no outcome, observation label or future-action
        # feature argument. A causal backbone ensures each endpoint sees its prefix.
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError('PRM scores one unpadded trajectory per forward call')
        positions = torch.as_tensor(action_end_positions, dtype=torch.long, device=input_ids.device)
        if positions.numel() < 1 or positions.ndim != 1:
            raise ValueError('Action endpoints must be a nonempty vector')
        if int(positions[0]) < 0 or int(positions[-1]) >= input_ids.shape[1] or not bool(torch.all(positions[1:] > positions[:-1])):
            raise ValueError('Invalid endpoint order or bounds')
        output = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                               logits_to_keep=positions, use_cache=False)
        if output.logits.shape[1] != len(positions):
            raise RuntimeError('Backbone did not honor selected endpoint logits')
        # Disable any outer BF16 autocast explicitly: head and regression remain FP32.
        with torch.autocast(device_type=input_ids.device.type, enabled=False):
            return self.head(output.logits[0].float()).squeeze(-1)


def load_trainable(model_path, device):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM
    backbone = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
                torch_dtype=torch.bfloat16, attn_implementation='sdpa', device_map={'': device})
    backbone.config.use_cache = False
    backbone = get_peft_model(backbone, LoraConfig(r=32, lora_alpha=64, lora_dropout=0.,
                             target_modules='all-linear', bias='none', task_type='CAUSAL_LM'))
    backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    backbone.enable_input_require_grads()
    model = QwenSPAPRM(backbone)
    model.head.to(device=device, dtype=torch.float32)
    return model


def score_record(model, row, device):
    validate_inputs(row['input_ids'], row['action_end_positions'], vocab_size=model.backbone.config.vocab_size)
    if getattr(model, 'strict_prefix_scoring', False):
        # Dynamic INT8 activation scales can depend on every token in a matrix.
        # Even causal attention then cannot guarantee prefix independence when a
        # whole trajectory is quantized together. Quantized deployment therefore
        # runs each action's exact prefix independently, with no later observation
        # or action anywhere in that forward's input tensor.
        scores = []
        for end in row['action_end_positions']:
            ids = torch.tensor([row['input_ids'][:end + 1]], device=device, dtype=torch.long)
            scores.append(model(input_ids=ids, action_end_positions=[end], attention_mask=torch.ones_like(ids))[0])
        return torch.stack(scores)
    ids = torch.tensor([row['input_ids']], device=device, dtype=torch.long)
    return model(input_ids=ids, action_end_positions=row['action_end_positions'], attention_mask=torch.ones_like(ids))
