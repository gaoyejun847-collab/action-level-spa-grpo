"""Terminal-calibrated, action-level GRPO advantages.

An action is a complete generation call, including EOS.  Action IDs originate
from the rollout token trace, never from parsing or counting Thought markers.
PRM service availability/version checks belong to the scoring client: a missing
service must raise before this function, whereas an invalid individual score
triggers the explicitly specified whole-group outcome fallback.
"""

from __future__ import annotations

import math
from collections import defaultdict
from numbers import Integral
from typing import Any, Hashable, Sequence

import torch


def _mask(response_mask: torch.Tensor) -> torch.Tensor:
    if response_mask.ndim != 2:
        raise ValueError("response_mask must have shape [batch, response_tokens]")
    if not torch.all((response_mask == 0) | (response_mask == 1)):
        raise ValueError("response_mask must contain only zero or one")
    return response_mask.bool()


def _action_counts(action_ids: torch.Tensor, response_mask: torch.Tensor) -> list[int]:
    mask = _mask(response_mask)
    if action_ids.shape != mask.shape or action_ids.device != mask.device:
        raise ValueError("action_ids must match response_mask shape and device")
    if action_ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("action_ids must be an integer tensor")
    if (action_ids < -1).any() or not torch.equal(action_ids >= 0, mask):
        raise ValueError("Only complete action tokens may have action IDs; padding/observations use -1")
    counts = []
    for row in action_ids.detach().cpu().tolist():
        runs, previous = [], -1
        for value in row:
            if value >= 0 and value != previous:
                runs.append(value)
            previous = value
        if runs != list(range(len(runs))):
            raise ValueError("Action IDs must be contiguous, ordered, zero-based generation calls")
        counts.append(len(runs))
    return counts


def pad_action_ids(raw_ids: Sequence[Sequence[int]], response_mask: torch.Tensor) -> torch.Tensor:
    """Right-pad rollout IDs to the response tensor; reject hidden mask drift."""
    mask = _mask(response_mask)
    if len(raw_ids) != mask.shape[0]:
        raise ValueError("One action-ID sequence is required per response")
    result = torch.full(mask.shape, -1, dtype=torch.long, device=mask.device)
    for index, row in enumerate(raw_ids):
        if isinstance(row, torch.Tensor):
            row = row.detach().cpu().tolist()
        if len(row) > mask.shape[1] or any(not isinstance(value, Integral) for value in row):
            raise ValueError("Invalid action-ID metadata length or type")
        result[index, :len(row)] = torch.as_tensor(row, dtype=torch.long, device=mask.device)
    _action_counts(result, response_mask)
    return result


@torch.no_grad()
def action_loss_weights(action_ids: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    """Return 1 / (number of actions * tokens in this action), zero elsewhere."""
    counts = _action_counts(action_ids, response_mask)
    weights = torch.zeros(action_ids.shape, device=action_ids.device, dtype=torch.float32)
    for row, count in enumerate(counts):
        for action in range(count):
            positions = action_ids[row] == action
            weights[row, positions] = 1.0 / (count * int(positions.sum().item()))
    return weights


def calibrate_returns(scores: Sequence[float], outcome: float, gamma: float = 1.0) -> tuple[list[float], list[float], float]:
    """Calibrate the last complete action without clipping and compute returns.

    With gamma=1 these are *remaining* returns, not causal contributions.  The
    raw last-action score cancels algebraically; computing its calibrated value
    from the prefix also avoids round-off dependence on that cancelled value.
    """
    if not scores or not all(math.isfinite(float(value)) for value in scores):
        raise ValueError("Calibration requires a nonempty sequence of finite scores")
    if not math.isfinite(outcome) or not 0 <= gamma <= 1:
        raise ValueError("Outcome must be finite and gamma must be in [0, 1]")
    raw = [float(value) for value in scores]
    residual = float(outcome) - math.fsum(raw)
    calibrated = raw[:-1] + [float(outcome) - math.fsum(raw[:-1])]
    if gamma == 1:
        returns = [float(outcome) - math.fsum(raw[:index]) for index in range(len(raw))]
    else:
        returns = [0.0] * len(raw)
        cumulative = 0.0
        for index in range(len(raw) - 1, -1, -1):
            cumulative = calibrated[index] + gamma * cumulative
            returns[index] = cumulative
    return calibrated, returns, residual


def _standardize(values: list[float], epsilon: float) -> tuple[list[float], float, float]:
    mean = math.fsum(values) / len(values)
    std = math.sqrt(math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1)) if len(values) > 1 else 0.0
    return [(value - mean) / (std + epsilon) for value in values], mean, std


@torch.no_grad()
def compute_action_advantages(
    outcome_rewards: torch.Tensor,
    action_ids: torch.Tensor,
    response_mask: torch.Tensor,
    group_ids: Sequence[Hashable],
    prm_scores: Sequence[Sequence[float] | None] | None = None,
    termination_reasons: Sequence[str] | None = None,
    *,
    prm_validity: Sequence[bool] | None = None,
    spa_enabled: bool = True,
    lambda_spa: float = 0.2,
    gamma: float = 1.0,
    epsilon: float = 1e-6,
    std_threshold: float = 1e-4,
    residual_limit: float = 1.0,
    expected_group_size: int | None = None,
) -> dict[str, Any]:
    """Compute complete-group advantages before any actor/minibatch sharding.

    ``group_ids`` must identify one task, initial state and sampling policy. The
    caller validates that identity from rollout metadata. Missing/nonfinite PRM
    outputs are data-quality fallbacks, not substitutes for a missing PRM model.
    Padding has no depth, no score, no group-statistic entry, and no loss.
    """
    counts = _action_counts(action_ids, response_mask)
    batch = len(counts)
    if not batch:
        raise ValueError("A policy update requires at least one trajectory")
    if outcome_rewards.shape != (batch,) or len(group_ids) != batch:
        raise ValueError("Outcome rewards and group IDs must match the trajectory batch")
    if not 0 <= lambda_spa <= 1 or not 0 <= gamma <= 1:
        raise ValueError("lambda_spa and gamma must be in [0, 1]")
    if epsilon <= 0 or std_threshold <= 0 or residual_limit < 0:
        raise ValueError("Invalid normalization or residual thresholds")
    rewards = [float(value) for value in outcome_rewards.detach().cpu().tolist()]
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in rewards):
        raise ValueError("Environment outcome rewards must be finite and in [0, 1]")
    reasons = list(termination_reasons) if termination_reasons is not None else ["terminated"] * batch
    scores = list(prm_scores) if prm_scores is not None else [None] * batch
    validity = list(prm_validity) if prm_validity is not None else [True] * batch
    if len(scores) != batch or len(reasons) != batch or len(validity) != batch:
        raise ValueError("Score, termination and validity metadata must match the batch")
    if any(reason not in {"terminated", "max_steps", "incomplete_action", "context_budget"} for reason in reasons):
        raise ValueError("Infrastructure/unknown termination is not a training trajectory")

    grouped: dict[Hashable, list[int]] = defaultdict(list)
    for index, group in enumerate(group_ids):
        grouped[group].append(index)
    if expected_group_size is not None and any(len(indices) != expected_group_size for indices in grouped.values()):
        raise ValueError("Incomplete rollout group: refusing statistics on a partial group")

    advantages = torch.zeros(action_ids.shape, dtype=torch.float32, device=action_ids.device)
    token_returns = torch.zeros_like(advantages)
    outcome_advantages = torch.zeros(batch, dtype=torch.float32, device=action_ids.device)
    weights = action_loss_weights(action_ids, response_mask)
    audit: list[dict[str, Any]] = []
    for index, count in enumerate(counts):
        row: dict[str, Any] = dict(
            trajectory_index=index, group_id=str(group_ids[index]), outcome_reward=rewards[index],
            termination_reason=reasons[index], num_actions=count, raw_scores=None,
            calibrated_scores=None, calibration_residual=None,
            action_returns=[rewards[index]] * count, outcome_advantage=0.0,
            spa_advantages=[0.0] * count, mixed_advantages=[0.0] * count,
            mixing_coefficients=[0.0] * count, depth_fallback_reasons=[None] * count,
            score_error=None, group_fallback_reasons=[],
        )
        if spa_enabled:
            if count == 0:
                row["score_error"] = "no_complete_actions"
            elif not validity[index]:
                row["score_error"] = "invalid_prm_scores"
            elif scores[index] is None:
                row["score_error"] = "missing_prm_scores"
            else:
                raw = scores[index]
                if isinstance(raw, torch.Tensor):
                    raw = raw.detach().cpu().tolist()
                try:
                    raw = [float(value) for value in raw]
                except (TypeError, ValueError):
                    row["score_error"] = "invalid_prm_scores"
                else:
                    if len(raw) != count:
                        row["score_error"] = "prm_score_count_mismatch"
                    elif not all(math.isfinite(value) for value in raw):
                        row["score_error"] = "nonfinite_prm_scores"
                    else:
                        row["raw_scores"] = raw
                        try:
                            calibrated, returns, residual = calibrate_returns(raw, rewards[index], gamma)
                        except (OverflowError, ValueError):
                            row["score_error"] = "nonfinite_prm_aggregate"
                        else:
                            if not all(math.isfinite(value) for value in calibrated + returns + [residual]):
                                row["score_error"] = "nonfinite_prm_aggregate"
                            else:
                                row.update(calibrated_scores=calibrated, action_returns=returns,
                                           calibration_residual=residual)
                                if abs(residual) > residual_limit:
                                    row["score_error"] = "calibration_residual_exceeded"
        audit.append(row)

    fallback_groups = 0
    fallback_depths, total_depths = 0, 0
    for indices in grouped.values():
        out_adv, mean_r, std_r = _standardize([rewards[index] for index in indices], epsilon)
        group_reasons = set()
        if std_r < std_threshold:
            out_adv = [0.0] * len(indices)
            group_reasons.add("outcome_zero_variance")
        if spa_enabled:
            for index in indices:
                if audit[index]["score_error"]:
                    group_reasons.add(audit[index]["score_error"])
                if reasons[index] in {"incomplete_action", "context_budget"}:
                    group_reasons.add(reasons[index])
        group_enabled = spa_enabled and not group_reasons
        if spa_enabled and group_reasons:
            fallback_groups += 1
        for local, index in enumerate(indices):
            outcome_advantages[index] = out_adv[local]
            audit[index].update(outcome_advantage=out_adv[local], outcome_group_mean=mean_r,
                                outcome_group_std=std_r, group_fallback_reasons=sorted(group_reasons))
        for depth in range(max(counts[index] for index in indices)):
            present = [index for index in indices if counts[index] > depth]
            if group_enabled:
                depth_adv, _, depth_std = _standardize([audit[index]["action_returns"][depth] for index in present], epsilon)
                depth_reason = "single_sample_depth" if len(present) < 2 else "depth_zero_variance" if depth_std < std_threshold else None
            else:
                # Do not normalize invalid/oversized process returns in a
                # group already assigned the complete outcome fallback.
                depth_adv = [0.0] * len(present)
                depth_reason = "single_sample_depth" if len(present) < 2 else None
            alpha = lambda_spa if group_enabled and depth_reason is None else 0.0
            total_depths += 1
            if spa_enabled and (not group_enabled or depth_reason is not None):
                fallback_depths += 1
            for local, index in enumerate(present):
                spa_adv = depth_adv[local] if group_enabled and depth_reason is None else 0.0
                outcome_adv = audit[index]["outcome_advantage"]
                mixed = (1.0 - alpha) * outcome_adv + alpha * spa_adv
                row = audit[index]
                row["spa_advantages"][depth] = spa_adv
                row["mixed_advantages"][depth] = mixed
                row["mixing_coefficients"][depth] = alpha
                row["depth_fallback_reasons"][depth] = depth_reason
                tokens = action_ids[index] == depth
                advantages[index, tokens] = mixed
                token_returns[index, tokens] = row["action_returns"][depth] if group_enabled else rewards[index]

    residuals = [abs(row["calibration_residual"]) for row in audit if row["calibration_residual"] is not None]
    metrics = {
        "spa/group_fallback_rate": fallback_groups / len(grouped),
        "spa/depth_fallback_rate": fallback_depths / total_depths if total_depths else 0.0,
        "spa/mean_abs_calibration_residual": math.fsum(residuals) / len(residuals) if residuals else 0.0,
        "spa/mean_actions": sum(counts) / batch,
        "spa/mean_mixing_coefficient": math.fsum(sum(row["mixing_coefficients"]) for row in audit) / sum(counts) if sum(counts) else 0.0,
    }
    return dict(advantages=advantages, returns=token_returns, action_loss_weights=weights,
                outcome_advantages=outcome_advantages, audit=audit, metrics=metrics)
