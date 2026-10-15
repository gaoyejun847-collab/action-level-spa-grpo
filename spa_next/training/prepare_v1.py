"""Freeze the common A/B/C tasks without selecting on validation outcomes."""
import argparse
import hashlib
import json
import random
from pathlib import Path

from spa_next.training.sft import goal_family
from spa_next.training.sft_snapshot import sha256


def select_rows(splits, seed=42, train_count=512, val_count=128):
    heldout = {goal_family(row['prompt'][-1]['content'])
               for split in ('val', 'test') for row in splits[split]}
    eligible = [row for row in splits['train']
                if goal_family(row['prompt'][-1]['content']) not in heldout]
    # Freeze an explicit order once. The V1 loader must use shuffle=false.
    random.Random(seed).shuffle(eligible)
    validation = list(splits['val'])
    random.Random(seed).shuffle(validation)
    if len(eligible) < train_count or len(validation) < val_count:
        raise ValueError('Not enough isolated tasks for the reviewed experiment')
    selected = dict(train=eligible[:train_count], val=validation[:val_count],
                    val_full=list(splits['val']))
    for key, rows in selected.items():
        ids = [int(row['extra_info']['task_id']) for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicate tasks in ' + key)
    return selected, len(eligible)


def main():
    import pyarrow as pa
    import pyarrow.parquet as pq
    parser = argparse.ArgumentParser()
    parser.add_argument('--split-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source_hashes = {key: sha256(args.split_dir / (key + '.parquet'))
                     for key in ('train', 'val', 'test')}
    if args.output.exists():
        report = json.loads((args.output / 'manifest.json').read_text())
        if report['source_sha256'] != source_hashes:
            raise ValueError('Frozen source splits changed')
        for key, expected in report['parquet_sha256'].items():
            if sha256(args.output / (key + '.parquet')) != expected:
                raise ValueError('Common experiment tasks changed: ' + key)
        print(json.dumps(report, indent=2))
        return
    splits = {key: pq.read_table(args.split_dir / (key + '.parquet')).to_pylist()
              for key in source_hashes}
    selected, eligible = select_rows(splits)
    args.output.mkdir(parents=True, exist_ok=False)
    for key, rows in selected.items():
        pq.write_table(pa.Table.from_pylist(rows), args.output / (key + '.parquet'))
    task_ids = {key: [int(row['extra_info']['task_id']) for row in rows]
                for key, rows in selected.items()}
    report = dict(seed=42, shuffle_during_training=False, eligible_train_tasks=eligible,
        source_sha256=source_hashes, tasks=task_ids,
        parquet_sha256={key: sha256(args.output / (key + '.parquet')) for key in selected},
        task_order_sha256=hashlib.sha256(json.dumps(task_ids['train']).encode()).hexdigest(),
        policy_train_count=512, validation_count=128, full_validation_count=len(splits['val']),
        group_size=4, tasks_per_update=4, updates=128,
        test_used_for='identity exclusion only; never evaluated or used to select parameters')
    (args.output / 'manifest.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'tasks'}, indent=2))


if __name__ == '__main__':
    main()
