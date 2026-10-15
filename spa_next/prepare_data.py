"""Freeze WebShop reset observations and split identities into verl parquet rows."""
import argparse
import hashlib
import json
import random
from pathlib import Path
from urllib.request import Request, urlopen


def request(url, op, body):
    req = Request(url + "/" + op, data=json.dumps(body).encode(),
                  headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=120) as response:
        return json.load(response)


def select_splits(training, reserved_test, train_limit, val_limit):
    if len(set(training)) != len(training):
        raise ValueError("Duplicate train task IDs")
    reserved_test = set(reserved_test)
    usable = [task for task in training if task not in reserved_test]
    if train_limit < 4 or val_limit < 4 or train_limit + val_limit > len(usable):
        raise ValueError("Need disjoint train/validation tasks, at least four each")
    return {"train": usable[:train_limit], "val": usable[-val_limit:]}


def select_all_splits(task_count, reserved_test, seed=42, batch_size=4):
    """Cover every task once; keep existing official test tasks held out."""
    universe = set(range(task_count))
    reserved = set(reserved_test)
    if not reserved <= universe or task_count < 20 or batch_size < 1:
        raise ValueError("Invalid task universe or reserved test IDs")
    train_count = int(task_count * .8) // batch_size * batch_size
    val_count = round(task_count * .1)
    test_count = task_count - train_count - val_count
    if len(reserved) > test_count:
        raise ValueError("Reserved test list exceeds requested test partition")
    rng = random.Random(seed)
    remaining = sorted(universe - reserved)
    rng.shuffle(remaining)
    additional_test = test_count - len(reserved)
    test = sorted(reserved) + remaining[:additional_test]
    remaining = remaining[additional_test:]
    splits = dict(train=remaining[:train_count], val=remaining[train_count:], test=test)
    assert len(splits['val']) == val_count
    assert sum(map(len, splits.values())) == task_count
    return splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--out", default="outputs/v0/data")
    parser.add_argument("--train-limit", type=int, default=16)
    parser.add_argument("--val-limit", type=int, default=4)
    parser.add_argument("--all-tasks", action="store_true")
    parser.add_argument("--split-seed", type=int, default=42)
    args = parser.parse_args()
    import pyarrow as pa
    import pyarrow.parquet as pq
    root = Path(__file__).resolve().parents[1]
    training = json.loads((root / "eval_agent/data/webshop/train_indices.json").read_text())
    # Hold out validation FROM THE TRAIN SPLIT; never tune on the official test set.
    reserved_test = set()
    for filename in ("test_indices.json", "test_indices_500.json"):
        reserved_test.update(json.loads((root / "eval_agent/data/webshop" / filename).read_text()))
    if args.all_tasks:
        with urlopen(args.url + '/health', timeout=10) as response:
            task_count = int(json.load(response)['task_count'])
        splits = select_all_splits(task_count, reserved_test, args.split_seed)
    else:
        splits = select_splits(training, reserved_test, args.train_limit, args.val_limit)
    instruction = (root / "eval_agent/prompt/instructions/webshop_inst.txt").read_text()
    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'manifest.json').exists():
        raise FileExistsError('Dataset is already frozen; choose a new output directory')
    manifest = {"env": "webshop-human-goals", "env_seed": 0, "splits": splits,
                "excluded_test_overlap": sorted(set(training) & reserved_test),
                "instruction_sha256": hashlib.sha256(instruction.encode()).hexdigest()}
    manifest.update(mode='all-tasks' if args.all_tasks else 'subset', split_seed=args.split_seed,
                    task_count=sum(map(len, splits.values())), reserved_test_ids=sorted(reserved_test))
    for split, tasks in splits.items():
        rows = []
        for index, task_id in enumerate(tasks):
            episode = request(args.url, "reset", {"task_id": int(task_id), "env_seed": 0})
            try:
                rows.append(dict(data_source="webshop", agent_name="spa_webshop", ability="agent",
                                 prompt=[dict(role="system", content=instruction),
                                         dict(role="user", content=episode["observation"])],
                                 reward_model=dict(style="rule", ground_truth="environment"),
                                 extra_info=dict(task_id=int(task_id), index=int(task_id), env_seed=0,
                                                 split=split, initial_state_id=episode["initial_state_id"])))
            finally:
                request(args.url, "close", {"episode_id": episode["episode_id"]})
            if (index + 1) % 250 == 0:
                print(f'{split}: {index + 1}/{len(tasks)}', flush=True)
        pq.write_table(pa.Table.from_pylist(rows), dest / (split + ".parquet"))
        print(f'{split}: {len(rows)} rows saved', flush=True)
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
