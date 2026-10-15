"""Audit saved online rollouts without loading the model or simulator."""
import argparse
import collections
import json
import math
from pathlib import Path


def audit(directory, group_size=4):
    groups = collections.defaultdict(list)
    versions = collections.Counter()
    reasons = collections.Counter()
    records = sorted(Path(directory).glob("*.json"))
    if not records:
        raise ValueError("No trajectory records")
    for path in records:
        row = json.loads(path.read_text())
        ids, mask, actions = row["response_ids"], row["response_mask"], row["action_ids"]
        assert len(ids) == len(mask) == len(actions), path
        expected_mask, expected_actions = [0] * len(ids), [-1] * len(ids)
        for turn in row["steps"]:
            start, end = turn["action_span"]
            assert 0 <= start <= end <= len(ids), path
            assert turn["context_token_ids"] == row["prompt_ids"] + ids[:start], path
            assert turn["generated_token_ids"] == ids[start:end], path
            if turn["complete"]:
                assert end > start and len(turn["behavior_logprobs"]) == end - start, path
                assert all(math.isfinite(x) for x in turn["behavior_logprobs"]), path
                expected_mask[start:end] = [1] * (end - start)
                expected_actions[start:end] = [turn["step_id"]] * (end - start)
        assert any(mask) and mask == expected_mask and actions == expected_actions, path
        assert row["prm_version"] is None, path
        assert 0 <= row["outcome_reward"] <= 1, path
        version = row["policy_version"]
        versions[f"step_{version['step']}_validate_{version['validate']}"] += 1
        reasons[row["termination_reason"]] += 1
        if not version["validate"]:
            # Keep only group metadata, not tens of thousands of token traces.
            compact = {key: row[key] for key in ('policy_version', 'task_id', 'initial_state_id',
                       'env_seed', 'episode_id', 'outcome_reward', 'success', 'termination_reason')}
            groups[(version["step"], row["group_id"])].append(compact)
    step_groups = collections.defaultdict(list)
    for key, rows in groups.items():
        assert len(rows) == group_size, (key, "incomplete/duplicated group", len(rows))
        assert {r["policy_version"]["rollout_n"] for r in rows} == set(range(group_size)), key
        assert len({(r["task_id"], r["initial_state_id"], r["env_seed"]) for r in rows}) == 1, key
        assert len({r["episode_id"] for r in rows}) == group_size, key
        step_groups[key[0]].append(rows)
    step_summary = {}
    for step, batches in sorted(step_groups.items()):
        samples = [row for batch in batches for row in batch]
        same_reward = sum(len({r["outcome_reward"] for r in batch}) == 1 for batch in batches)
        step_summary[step] = dict(
            groups=len(batches), zero_reward_variance_fraction=same_reward / len(batches),
            reward_mean=sum(r["outcome_reward"] for r in samples) / len(samples),
            success_rate=sum(r["success"] for r in samples) / len(samples),
            termination_reasons=dict(collections.Counter(r["termination_reason"] for r in samples)))
    return dict(trajectories=len(records), training_groups=len(groups),
                versions=dict(versions), termination_reasons=dict(reasons),
                training_steps=step_summary,
                checks="original tokens, context prefixes, action-only masks, finite behavior logprobs, isolated fixed-state groups")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", nargs="?", default="outputs/v0/trajectories")
    parser.add_argument("--group-size", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(audit(args.directory, args.group_size), indent=2))
