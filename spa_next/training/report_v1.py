"""Summarize V1 held-out outcomes and verify the equal A/B/C sampling budget."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def read_trajectories(directory):
    return [json.loads(p.read_text()) for p in sorted(Path(directory).glob('*.json'))]


def outcome_summary(records):
    unique = {}
    for row in records:
        task = int(row['task_id'])
        if task in unique:
            previous = unique[task]
            if (previous['outcome_reward'], previous['success']) != (row['outcome_reward'], row['success']):
                raise ValueError('Repeated greedy validation task has inconsistent outcome: ' + str(task))
        unique[task] = row
    if not unique:
        raise ValueError('No validation trajectories found')
    rows = list(unique.values())
    failures, terminations = Counter(), Counter()
    for row in rows:
        reason = row['termination_reason']
        terminations[reason] += 1
        if row['success']:
            continue
        if reason in ('context_budget', 'incomplete_action', 'max_steps'):
            failures[reason] += 1
        elif any(step.get('error_type') for step in row['steps']):
            failures['environment_or_action_error'] += 1
        elif reason == 'terminated':
            failures['purchase_not_fully_correct'] += 1
        else:
            failures[reason] += 1
    return dict(tasks=len(rows), padding_duplicates_removed=len(records)-len(rows),
        reward_mean=sum(float(row['outcome_reward']) for row in rows)/len(rows),
        success_mean=sum(bool(row['success']) for row in rows)/len(rows),
        steps_mean=sum(sum(bool(s.get('complete')) for s in row['steps']) for row in rows)/len(rows),
        termination_counts=dict(terminations), failure_counts=dict(failures))


def audit_budget(records, expected_tasks, *, arm=None, sft_version=None):
    grouped = defaultdict(list)
    versions = set()
    states = {}
    for row in records:
        version = row['policy_version']
        if version.get('validate'):
            continue
        if arm is not None and row.get('actor_version', {}).get('experiment') != arm:
            raise ValueError('Trajectory actor arm differs from the requested experiment')
        if sft_version is not None and row.get('sft_version') != sft_version:
            raise ValueError('Trajectory SFT base version differs from the common snapshot')
        grouped[(int(version['step']), int(row['task_id']))].append(row)
        versions.add(json.dumps(row.get('model'), sort_keys=True))
    actual_tasks = []
    per_step = defaultdict(list)
    for (step, task), rows in grouped.items():
        if len(rows) != 4 or len({r['episode_id'] for r in rows}) != 4:
            raise ValueError(f'Expected four fresh trajectories for step {step}, task {task}')
        if len({r['initial_state_id'] for r in rows}) != 1:
            raise ValueError('Initial state differs within a task group')
        states[str(task)] = rows[0]['initial_state_id']
        if {r['policy_version'].get('rollout_n') for r in rows} != {0, 1, 2, 3}:
            raise ValueError('Wrong rollout budget')
        actual_tasks.append(task)
        per_step[step].append(task)
    if Counter(actual_tasks) != Counter(expected_tasks):
        raise ValueError('Training tasks or number of visits differ from the frozen 512 tasks')
    if set(per_step) != set(range(1, 129)) or any(len(tasks) != 4 for tasks in per_step.values()):
        raise ValueError('Expected 128 updates of four complete task groups')
    batches = [sorted(per_step[step]) for step in sorted(per_step)]
    expected_batches = [sorted(expected_tasks[i:i+4]) for i in range(0, len(expected_tasks), 4)]
    if batches != expected_batches:
        raise ValueError('Training task order differs from the frozen common experiment order')
    return dict(updates=128, distinct_tasks=512, training_trajectories=len(grouped)*4,
                fresh_episode_ids=True, identical_initial_states=True, task_batches=batches,
                policy_models=sorted(versions), initial_states=states)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    dest = args.output
    manifest = json.loads((dest / 'data/manifest.json').read_text())
    baseline = outcome_summary(read_trajectories(dest / 'baseline/trajectories'))
    if baseline['tasks'] != manifest['full_validation_count']:
        raise ValueError('Unified baseline did not cover the full validation set')
    report = dict(historical_transformers_sft=dict(tasks=1209, reward_mean=0.6077586093715117,
                  success_mean=0.26054590570719605), unified_verl_sft=baseline, experiments={})
    status = json.loads((dest / 'status.json').read_text())
    common_states = None
    for variant in ('A', 'B', 'C'):
        run = dest / variant
        metrics = outcome_summary(read_trajectories(run / 'full-validation/trajectories'))
        if metrics['tasks'] != manifest['full_validation_count']:
            raise ValueError('Incomplete final validation for ' + variant)
        budget = audit_budget(read_trajectories(run / 'trajectories'), manifest['tasks']['train'],
                              arm=variant, sft_version=status['sft_version'])
        if budget['policy_models'] != [json.dumps(status['model'])]:
            raise ValueError('Policy model path differs from the common SFT snapshot')
        if common_states is not None and budget['initial_states'] != common_states:
            raise ValueError('Initial training states differ between experiment arms')
        common_states = budget['initial_states']
        summary = dict(final_validation=metrics, sampling_audit=budget,
            reward_delta=metrics['reward_mean']-baseline['reward_mean'],
            success_delta=metrics['success_mean']-baseline['success_mean'])
        audit_files = sorted((run / 'audit').glob('step_*.json'))
        if len(audit_files) != 128:
            raise ValueError('Expected one complete advantage audit per update for ' + variant)
        reasons, fallback, depth_fallback = Counter(), [], []
        versions = set()
        for path in audit_files:
            audit = json.loads(path.read_text())
            if audit['arm'] != variant:
                raise ValueError('Wrong algorithm arm in ' + str(path))
            versions.add(audit['prm_version'])
            fallback.append(audit['result']['metrics']['spa/group_fallback_rate'])
            depth_fallback.append(audit['result']['metrics']['spa/depth_fallback_rate'])
            for trajectory in audit['result']['audit']:
                reasons.update(trajectory['group_fallback_reasons'])
        summary.update(advantage_audit_records=len(audit_files),
            group_fallback_rate=sum(fallback)/len(fallback),
            depth_fallback_rate=sum(depth_fallback)/len(depth_fallback),
            fallback_reason_trajectory_counts=dict(reasons), prm_versions=list(versions))
        if variant == 'C' and (None in versions or len(versions) != 1):
            raise ValueError('Experiment C did not use one pinned frozen PRM')
        report['experiments'][variant] = summary
    stages = [json.loads(line) for line in (dest / 'stages.jsonl').read_text().splitlines()]
    report['resource_time_seconds'] = {stage['stage']: stage['elapsed_seconds'] for stage in stages}
    resource_rows = [json.loads(line) for line in (dest / 'resources.jsonl').read_text().splitlines()]
    peaks = defaultdict(int)
    for row in resource_rows:
        for line in row['gpu'].splitlines():
            index, used, utilization = (int(x.strip()) for x in line.split(','))
            peaks[str(index)] = max(peaks[str(index)], used)
    report['sampled_peak_gpu_memory_mib'] = dict(peaks)
    report['prm_training'] = json.loads((dest / 'prm/training/training_report.json').read_text())
    report['prm_quantization'] = json.loads((dest / 'prm/frozen/quantization_report.json').read_text())
    report['interpretation'] = 'Single small experiment; no statistical superiority claim; test set not evaluated.'
    (dest / 'comparison.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
