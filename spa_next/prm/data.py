"""Prepare leakage-free PRM supervision from original, token-exact V0 traces."""
import collections
import hashlib
import json
import math
import random
from pathlib import Path

from spa_next.training.sft import goal_family


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def eligible_task_families(rows_by_split):
    splits = {}
    for name in ('train', 'val', 'test'):
        rows = rows_by_split[name]
        splits[name] = {int(row['extra_info']['task_id']):
                        goal_family(row['prompt'][-1]['content']) for row in rows}
    heldout_ids = set(splits['val']) | set(splits['test'])
    heldout_families = set(splits['val'].values()) | set(splits['test'].values())
    eligible = {task: family for task, family in splits['train'].items()
                if task not in heldout_ids and family not in heldout_families}
    return eligible, splits


def frozen_tasks(split_dir):
    import pyarrow.parquet as pq
    return eligible_task_families({name: pq.read_table(Path(split_dir) / (name + '.parquet')).to_pylist()
                                  for name in ('train', 'val', 'test')})


def extract_record(trace, eligible, *, max_length=4096, eos_token_id=None):
    """Return supervision and rejection reason; outcome never enters model inputs."""
    task_id = int(trace.get('task_id', -1))
    if task_id not in eligible:
        return None, 'heldout_or_unmatched_task_family'
    if trace.get('policy_version', {}).get('validate') is not False:
        return None, 'not_training_rollout'
    if trace.get('termination_reason') not in ('terminated', 'max_steps'):
        return None, 'incomplete_or_infrastructure_termination'
    steps = trace.get('steps', [])
    if not steps or any(s.get('complete') is not True for s in steps):
        return None, 'incomplete_action'
    if trace['termination_reason'] == 'max_steps' and len(steps) != 10:
        return None, 'invalid_step_budget'
    if any(s.get('error_type') not in (None, 'invalid_format', 'invalid_action') for s in steps):
        return None, 'infrastructure_error'
    reward = trace.get('outcome_reward')
    if not isinstance(reward, (int, float)) or not math.isfinite(reward) or not 0 <= reward <= 1:
        return None, 'invalid_environment_reward'
    prompt, response = trace.get('prompt_ids', []), trace.get('response_ids', [])
    mask, action_ids = trace.get('response_mask', []), trace.get('action_ids', [])
    if not prompt or not response or len(mask) != len(response) or len(action_ids) != len(response):
        return None, 'invalid_token_alignment'
    ids = prompt + response
    if any(type(token) is not int or token < 0 for token in ids):
        return None, 'invalid_token_ids'
    if len(ids) > max_length:
        return None, 'over_context_limit'
    endpoints, seen_positions, previous_end = [], set(), 0
    for index, step in enumerate(steps):
        span = step.get('action_span', [])
        if len(span) != 2:
            return None, 'invalid_action_span'
        start, end = span
        if not (isinstance(start, int) and isinstance(end, int) and previous_end <= start < end <= len(response)):
            return None, 'invalid_action_span'
        generated = step.get('generated_token_ids', [])
        if generated != response[start:end] or step.get('context_token_ids') != ids[:len(prompt) + start]:
            return None, 'retokenized_or_mismatched_trace'
        if eos_token_id is not None and generated[-1] != eos_token_id:
            return None, 'missing_eos'
        if any(mask[t] != 1 or action_ids[t] != index for t in range(start, end)):
            return None, 'invalid_action_mask'
        seen_positions.update(range(start, end))
        endpoints.append(len(prompt) + end - 1)
        previous_end = end
    if any(bool(value) != (t in seen_positions) for t, value in enumerate(mask)):
        return None, 'invalid_action_mask'
    if any(action_ids[t] != -1 for t in range(len(response)) if t not in seen_positions):
        return None, 'invalid_observation_mask'
    # Exclude any suffix after the last assistant decision. Earlier observations
    # remain causal context; action endpoint logits cannot attend to future tokens.
    ids = ids[:endpoints[-1] + 1]
    identity = dict(task_id=task_id, initial_state_id=trace.get('initial_state_id'), input_ids=ids,
                    action_end_positions=endpoints)
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return dict(**identity, reward=float(reward), family=eligible[task_id],
                trajectory_sha256=fingerprint, episode_id=trace.get('episode_id'),
                policy_version=trace.get('policy_version'), termination_reason=trace['termination_reason']), None


def split_families(families, seed=42):
    ordered = sorted(set(families))
    if len(ordered) < 2:
        raise ValueError('PRM needs at least two independent task families')
    random.Random(seed).shuffle(ordered)
    count = max(1, min(len(ordered) - 1, round(len(ordered) * .1)))
    return set(ordered[count:]), set(ordered[:count])


def prepare(args):
    from transformers import AutoTokenizer
    output = Path(args.output)
    if not 1 <= args.max_length <= 4096:
        raise ValueError('PRM context must fit the reviewed 4096-token budget')
    if (output / 'data_report.json').is_file():
        report = json.loads((output / 'data_report.json').read_text())
        if (report['seed'] != args.seed or report['max_length'] != args.max_length or
                Path(report['model']).resolve() != Path(args.model).resolve() or
                report['source_roots'] != [str(Path(x).resolve()) for x in args.trajectories]):
            raise ValueError('Existing PRM dataset was prepared with different inputs or settings')
        for name in ('train', 'val', 'test'):
            if sha256_file(Path(args.split_dir) / (name + '.parquet')) != report['frozen_split_sha256'][name]:
                raise ValueError('Frozen policy task partition changed after PRM preparation')
        for name in ('train', 'val'):
            if sha256_file(output / (name + '.jsonl')) != report['dataset_sha256'][name]:
                raise ValueError('Frozen PRM supervision changed after preparation')
        print(json.dumps(dict(event='PRM_DATA_REUSED', output=str(output.resolve()), summary=report['summary'])), flush=True)
        return report
    output.mkdir(parents=True, exist_ok=False)
    eligible, splits = frozen_tasks(args.split_dir)
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tok.eos_token_id is None:
        raise ValueError('An explicit assistant EOS token is required')
    seen, families, counts = set(), set(), collections.Counter()
    spool = output / 'accepted.jsonl'
    sources = sorted({p.resolve() for root in args.trajectories for p in Path(root).rglob('*.json')})
    if not sources:
        raise ValueError('No original V0 trajectory JSON files found')
    with spool.open('w', encoding='utf-8') as accepted, (output / 'source_audit.jsonl').open('w', encoding='utf-8') as audit:
        for i, path in enumerate(sources):
            try:
                trace = json.loads(path.read_text(encoding='utf-8'))
                row, reason = extract_record(trace, eligible, max_length=args.max_length, eos_token_id=tok.eos_token_id)
            except (ValueError, TypeError, KeyError, IndexError) as exc:
                row, reason = None, 'malformed_trace:' + type(exc).__name__
            if row and row['trajectory_sha256'] in seen:
                row, reason = None, 'duplicate_trajectory'
            if row:
                seen.add(row['trajectory_sha256'])
                families.add(row['family'])
                accepted.write(json.dumps(row, separators=(',', ':')) + '\n')
            decision = reason or 'accepted'
            counts[decision] += 1
            audit.write(json.dumps(dict(source=str(path), decision=decision)) + '\n')
            if (i + 1) % 2000 == 0:
                print(json.dumps(dict(event='prepare_progress', files=i + 1, accepted=counts['accepted'])), flush=True)
    train_families, val_families = split_families(families, args.seed)
    summary = {name: dict(trajectories=0, tasks=set(), families=set(), reward_sum=0., successful=0,
                         tokens=0, actions=0, max_tokens=0) for name in ('train', 'val')}
    with (output / 'train.jsonl').open('w', encoding='utf-8') as train, (output / 'val.jsonl').open('w', encoding='utf-8') as val:
        with spool.open(encoding='utf-8') as records:
            for line in records:
                row = json.loads(line)
                split = 'val' if row['family'] in val_families else 'train'
                (val if split == 'val' else train).write(line)
                stats = summary[split]
                stats['trajectories'] += 1
                stats['tasks'].add(row['task_id'])
                stats['families'].add(row['family'])
                stats['reward_sum'] += row['reward']
                stats['successful'] += int(row['reward'] >= 1 - 1e-6)
                stats['tokens'] += len(row['input_ids'])
                stats['actions'] += len(row['action_end_positions'])
                stats['max_tokens'] = max(stats['max_tokens'], len(row['input_ids']))
    for stats in summary.values():
        stats['tasks'], stats['families'] = len(stats['tasks']), len(stats['families'])
        stats['reward_mean'] = stats['reward_sum'] / stats['trajectories']
    constant = summary['train']['reward_mean']
    with (output / 'val.jsonl').open() as records:
        constant_mse = sum((json.loads(line)['reward'] - constant) ** 2 for line in records) / summary['val']['trajectories']
    report = dict(schema_version=1, kind='qwen-spa-prm-supervision', seed=args.seed,
                  model=str(args.model), max_length=args.max_length, eos_token_id=tok.eos_token_id,
                  source_roots=[str(Path(x).resolve()) for x in args.trajectories],
                  filters=dict(counts), summary=summary, constant_train_mean=constant,
                  constant_validation_mse=constant_mse,
                  eligible_policy_train_task_ids=sorted(eligible),
                  train_families=sorted(train_families), val_families=sorted(val_families),
                  frozen_split_sha256={name: sha256_file(Path(args.split_dir) / (name + '.parquet')) for name in splits},
                  dataset_sha256={name: sha256_file(output / (name + '.jsonl')) for name in summary})
    (output / 'data_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    # Retain accepted.jsonl as an auditable, immutable source; it is never used for
    # policy updates. No SFT or old offline PRM predictions are supervision labels.
    print(json.dumps(dict(event='PRM_DATA_READY', output=str(output.resolve()), filters=dict(counts),
                          summary=summary, constant_train_mean=constant,
                          constant_validation_mse=constant_mse), indent=2), flush=True)
    return report


class JsonlDataset:
    """Offset-indexed records avoid replicating millions of Python token ints per GPU."""
    def __init__(self, path):
        self.path = Path(path)
        self.offsets = []
        with self.path.open('rb') as stream:
            while True:
                offset = stream.tell()
                if not stream.readline():
                    break
                self.offsets.append(offset)

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        with self.path.open('rb') as stream:
            stream.seek(self.offsets[index])
            return json.loads(stream.readline())
