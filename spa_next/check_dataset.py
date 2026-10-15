"""Check full coverage and actual tokenizer lengths before a long training job."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    manifest = json.loads((args.directory / 'manifest.json').read_text())
    sets, result = {}, {}
    for split, tasks in manifest['splits'].items():
        path = args.directory / (split + '.parquet')
        rows = pq.read_table(path).to_pylist()
        actual = [int(row['extra_info']['task_id']) for row in rows]
        assert actual == tasks and len(set(actual)) == len(actual), split
        sets[split] = set(actual)
        lengths = [len(tokenizer.apply_chat_template(row['prompt'], tokenize=True, add_generation_prompt=True))
                   for row in rows]
        assert max(lengths) <= 1024, (split, 'prompt exceeds configured limit', max(lengths))
        result[split] = dict(tasks=len(rows), max_prompt_tokens=max(lengths),
                             parquet_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    assert not (sets['train'] & sets['val'] or sets['train'] & sets['test'] or sets['val'] & sets['test'])
    assert set.union(*sets.values()) == set(range(manifest['task_count']))
    assert set(manifest['reserved_test_ids']) <= sets['test']
    assert len(sets['train']) % 4 == 0
    print(json.dumps(dict(check='passed', splits=result), indent=2))


if __name__ == '__main__':
    main()
