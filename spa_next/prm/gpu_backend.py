"""Qualify original, unmerged BF16 PRM on a dedicated GPU against saved reference scores."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def gpu_version(manifest):
    return 'qwen-spa-prm-original-gpu-' + hashlib.sha256(
        json.dumps(manifest['training_reference'], sort_keys=True).encode()).hexdigest()[:24]


def main():
    import torch
    from spa_next.prm.data import JsonlDataset, sha256_file
    from spa_next.prm.deployment import validate_artifact, load_training_reference, compare_scores
    from spa_next.prm.model import score_record
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--device', default='cuda:3')
    args = parser.parse_args()
    manifest = validate_artifact(args.model)
    old_report = json.loads((args.model / 'quantization_report.json').read_text())
    if old_report['validation_sha256'] != sha256_file(args.data / 'val.jsonl'):
        raise RuntimeError('Validation data changed')
    if old_report['artifact_manifest_sha256'] != sha256_file(args.model / 'prm_manifest.json'):
        raise RuntimeError('Reference artifact changed')
    audits = [json.loads(line) for line in (args.model / 'quantization_predictions.jsonl').read_text().splitlines()]
    if len(audits) != 256 or [a['trajectory_sha256'] for a in audits] != old_report['audited_trajectory_sha256']:
        raise RuntimeError('Incomplete reference audit')
    wanted = {a['trajectory_sha256'] for a in audits}
    ds, rows = JsonlDataset(args.data / 'val.jsonl'), {}
    for i in range(len(ds)):
        row = ds[i]
        if row['trajectory_sha256'] in wanted:
            rows[row['trajectory_sha256']] = row
    model = load_training_reference(manifest, args.device)
    predicted, elapsed = [], []
    with torch.inference_mode():
        for i, audit in enumerate(audits):
            row = rows[audit['trajectory_sha256']]
            started = time.monotonic()
            predicted.append(score_record(model, row, args.device).float().cpu().tolist())
            elapsed.append(time.monotonic()-started)
            if (i+1) % 32 == 0:
                print(json.dumps(dict(completed=i+1, total=256)), flush=True)
    report = compare_scores([a['original_scores'] for a in audits], predicted, [a['reward'] for a in audits])
    report.update(backend='original-gpu', model_version=gpu_version(manifest),
                  artifact_manifest_sha256=sha256_file(args.model / 'prm_manifest.json'),
                  validation_sha256=old_report['validation_sha256'], seconds_per_trajectory=sum(elapsed)/len(elapsed),
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device),
                  note='Original unmerged model, no INT8 and no PRM LoRA merge')
    (args.model / 'gpu_reference_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    if not report['gate_passed']:
        raise RuntimeError('Original GPU precision qualification failed')


if __name__ == '__main__':
    main()
