"""Small V1 hooks called by the guarded verl 0.7 integration patch."""
import json
import math
import os
import shutil
from pathlib import Path

import torch

from spa_next.algorithms.action_grpo import (
    compute_action_advantages, pad_action_ids,
)
from spa_next.algorithms.losses import (
    clipped_policy_loss, low_variance_kl, normalization_denominator, weighted_mean,
)


def enabled(config):
    return config.algorithm.adv_estimator == "spa_v1"


def _plain(value):
    if isinstance(value, torch.Tensor):
        return _plain(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None  # invalid PRM values remain explicit in fallback audit reasons
    if hasattr(value, "item"):
        return _plain(value.item())
    return value


def compute_advantage(data, config):
    """Score and normalize the whole on-policy batch before rank splitting."""
    settings = config.spa_v1
    arm = settings.arm
    mask = data.batch["response_mask"]
    action_ids = pad_action_ids(data.non_tensor_batch["action_ids"], mask)
    traces = list(data.non_tensor_batch["spa_v1_trace"])
    if len(traces) != len(mask):
        raise ValueError("Missing V1 trajectory metadata")
    steps = {int(t["policy_version"]["step"]) for t in traces}
    if len(steps) != 1 or min(steps) < 1:
        raise ValueError("Mixed or absent behavior policy versions")
    for i, trace in enumerate(traces):
        prompt_len = int(trace["prompt_length"])
        raw_ids = trace["input_ids"]
        valid_len = int(data.batch["attention_mask"][i, -mask.shape[1]:].sum())
        if raw_ids[prompt_len:] != data.batch["responses"][i, :valid_len].tolist():
            raise ValueError("PRM input differs from original generated response tokens")
        prompt = data.batch["prompts"][i]
        if raw_ids[:prompt_len] != prompt[-prompt_len:].tolist():
            raise ValueError("PRM input differs from original prompt tokens")
        expected = []
        for action in sorted(set(action_ids[i][mask[i].bool()].tolist())):
            indices = (action_ids[i] == action).nonzero().flatten()
            expected.append(prompt_len + int(indices[-1]))
        if expected != trace["action_end_positions"]:
            raise ValueError("PRM action endpoints do not match the policy mask")
    groups = {}
    for uid, trace in zip(data.non_tensor_batch["uid"], traces):
        provenance = (trace["task_id"], trace["initial_state_id"], trace["policy_version"]["step"])
        if uid in groups and groups[uid] != provenance:
            raise ValueError("Group mixes tasks, initial states or policy versions")
        groups[uid] = provenance

    scores, score_validity = None, None
    prm_version, score_results = None, None
    if arm == "C":
        from spa_next.prm.client import PRMClient
        prm_version = str(settings.prm_version)
        client = PRMClient(str(settings.prm_url), prm_version,
                           timeout=float(settings.prm_timeout))
        health = client.health()
        if not (health.get("precision_gate_passed") or health.get("quantization_gate_passed")):
            raise RuntimeError("PRM numerical comparison gate has not passed")
        score_indices = [i for i, trace in enumerate(traces) if trace["action_end_positions"]]
        results = client.score([{"input_ids": traces[i]["input_ids"],
                                 "action_end_positions": traces[i]["action_end_positions"]} for i in score_indices]) if score_indices else []
        if len(results) != len(score_indices):
            raise RuntimeError("PRM returned an incomplete batch")
        score_results = dict(zip(score_indices, results))
        scores = [None] * len(traces)
        score_validity = [False] * len(traces)
        for i, result in zip(score_indices, results):
            scores[i] = result.get("prm_scores")
            score_validity[i] = bool(result.get("valid"))

    result = compute_action_advantages(
        data.batch["token_level_rewards"].sum(-1), action_ids, mask,
        list(data.non_tensor_batch["uid"]), scores,
        [t["termination_reason"] for t in traces], spa_enabled=arm == "C",
        lambda_spa=float(settings.lambda_spa), gamma=float(settings.gamma),
        epsilon=float(settings.epsilon), std_threshold=float(settings.std_threshold),
        residual_limit=float(settings.residual_limit), expected_group_size=4,
        prm_validity=score_validity)
    for name in ("advantages", "returns", "action_loss_weights"):
        data.batch[name] = result[name]
    data.batch["action_ids"] = action_ids
    probability_audit = {}
    for key in ("old_log_probs", "rollout_log_probs"):
        if key in data.batch:
            value = data.batch[key].detach()
            if value.shape != mask.shape or not torch.isfinite(value[mask.bool()]).all():
                raise ValueError(f"Invalid {key} on complete action tokens")
            probability_audit[key] = value
    if len(probability_audit) == 2:
        valid = mask.bool()
        drift = (probability_audit["old_log_probs"].float()
                 - probability_audit["rollout_log_probs"].float()).abs()[valid]
        result["metrics"].update({
            "spa/hf_vllm_logprob_abs_mean": float(drift.mean()) if drift.numel() else 0.0,
            "spa/hf_vllm_logprob_abs_max": float(drift.max()) if drift.numel() else 0.0,
            "spa/hf_vllm_logprob_compared_tokens": int(drift.numel()),
        })
    data.meta_info["spa_v1_metrics"] = result["metrics"]
    out = Path(str(settings.audit_dir))
    out.mkdir(parents=True, exist_ok=True)
    step = steps.pop()
    payload = dict(step=step, arm=arm, prm_version=prm_version,
                   sft_version=getattr(settings, "sft_version", None),
                   group_ids=list(data.non_tensor_batch["uid"]),
                   trajectories=traces, action_ids=action_ids,
                   response_mask=mask, raw_prm_scores=scores,
                   prm_responses=score_results,
                   frozen_hf_old_log_probs=probability_audit.get("old_log_probs"),
                   original_vllm_rollout_log_probs=probability_audit.get("rollout_log_probs"),
                   result=result)
    target = out / f"step_{step:06d}.json"
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(_plain(payload), allow_nan=False), encoding="utf-8")
    os.replace(tmp, target)
    return data


def actor_mode(config):
    return config.policy_loss.loss_mode in {"spa_v1_A", "spa_v1_B", "spa_v1_C"}


def minibatch_normalizer(mini_batch, config, device):
    mask = mini_batch.batch["response_mask"].to(device)
    weights = None if config.policy_loss.loss_mode == "spa_v1_A" else mini_batch.batch["action_loss_weights"].to(device)
    valid = mini_batch.batch.get('spa_row_valid')
    return normalization_denominator(mask, weights, trajectory_valid=None if valid is None else valid.to(device))


def distributed_call(trainer, method, data, training=False):
    """Pad only worker RPCs; reward groups and trajectory audits stay unpadded."""
    if not enabled(trainer.config):
        return method(data)
    from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
    size = trainer._get_dp_size(trainer.actor_rollout_wg, 'actor')
    original = len(data)
    padded, count = pad_dataproto_to_divisor(data, size)
    if training:
        padded.batch['spa_row_valid'] = torch.ones(len(padded), dtype=torch.float32,
                                                   device=padded.batch['response_mask'].device)
        if count:
            padded.batch['spa_row_valid'][original:] = 0
            for key in ('response_mask', 'advantages', 'returns', 'action_loss_weights'):
                if key in padded.batch:
                    padded.batch[key][original:] = 0
    result = method(padded)
    return result if training else unpad_dataproto(result, count)


def policy_loss(model_inputs, old_log_prob, log_prob, config, normalizer):
    weights = None if config.policy_loss.loss_mode == "spa_v1_A" else model_inputs["action_loss_weights"]
    denominator, world = normalizer
    return clipped_policy_loss(old_log_prob, log_prob, model_inputs["advantages"],
                               model_inputs["response_mask"], weights,
                               clip_ratio=config.clip_ratio,
                               global_denominator=denominator, world_size=world)


def kl_loss(model_inputs, log_prob, config, normalizer):
    weights = None if config.policy_loss.loss_mode == "spa_v1_A" else model_inputs["action_loss_weights"]
    denominator, world = normalizer
    values = low_variance_kl(log_prob, model_inputs["ref_log_prob"])
    return weighted_mean(values, model_inputs["response_mask"], weights,
                         global_denominator=denominator, world_size=world)


def before_validation(trainer):
    """Checkpoint first so lengthy validation never endangers completed updates."""
    if enabled(trainer.config) and not trainer.config.trainer.get("val_only", False):
        trainer._save_checkpoint()
        trainer._spa_v1_saved_step = trainer.global_steps


def record_validation(trainer, metrics):
    if not enabled(trainer.config) or trainer.config.trainer.get("val_only", False):
        return
    success = float(metrics["val-aux/webshop/success/mean@1"])
    reward = float(metrics["val-core/webshop/reward/mean@1"])
    if not math.isfinite(success) or not math.isfinite(reward):
        raise ValueError("Validation returned nonfinite metrics")
    root = Path(trainer.config.trainer.default_local_dir)
    path = root / "best_validation.json"
    previous = json.loads(path.read_text()) if path.exists() else None
    if previous is not None and (success, reward) <= (previous["success"], previous["reward"]):
        return
    source = root / f"global_step_{trainer.global_steps}"
    if not (source / "data.pt").is_file():
        raise RuntimeError("Validation checkpoint was not saved completely")
    dest = root / "preserved_best" / source.name
    if dest.exists():
        raise RuntimeError(f"Refusing to overwrite preserved checkpoint: {dest}")
    entry = dict(step=trainer.global_steps, success=success, reward=reward,
                 checkpoint=str(dest), metrics=metrics)
    # Validate serialization before copying a large checkpoint; verl metrics
    # contain NumPy scalars as well as Python numbers.
    serialized = json.dumps(_plain(entry), indent=2, allow_nan=False)
    # Independent copies remain safe if ordinary checkpoint rotation unlinks files.
    shutil.copytree(source, dest)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(serialized, encoding="utf-8")
    os.replace(tmp, path)
