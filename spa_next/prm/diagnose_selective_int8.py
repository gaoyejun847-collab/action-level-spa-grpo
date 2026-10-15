"""Small diagnostic only: exclude vocabulary projection from dynamic INT8.

Never produces a deployment gate or changes the frozen artifact.
"""
import json
from pathlib import Path
import time


def main():
    import torch
    from transformers import AutoModelForCausalLM
    from spa_next.prm.data import JsonlDataset
    from spa_next.prm.model import QwenSPAPRM, score_record
    from spa_next.prm.deployment import compare_scores
    root = Path('outputs/v1-spa-abc-20260920/prm')
    frozen = root / 'frozen'
    audits = [json.loads(line) for line in (frozen / 'quantization_predictions.jsonl').read_text().splitlines()]
    # Two pinned examples from the original audit, not used to certify a gate.
    selected = [audits[6], audits[8]]
    wanted = {row['trajectory_sha256'] for row in selected}
    dataset = JsonlDataset(root / 'data/val.jsonl')
    rows = {}
    for index in range(len(dataset)):
        row = dataset[index]
        if row['trajectory_sha256'] in wanted:
            rows[row['trajectory_sha256']] = row
    torch.set_num_threads(8)
    torch.backends.quantized.engine = 'fbgemm'
    backbone = AutoModelForCausalLM.from_pretrained(frozen, local_files_only=True,
        torch_dtype=torch.float32, attn_implementation='sdpa', device_map={'': 'cpu'})
    output_layer = backbone.get_output_embeddings()
    targets = {name for name, module in backbone.named_modules()
               if isinstance(module, torch.nn.Linear) and module is not output_layer}
    backbone = torch.ao.quantization.quantize_dynamic(backbone, targets, dtype=torch.qint8, inplace=True)
    assert isinstance(backbone.get_output_embeddings(), torch.nn.Linear)
    assert backbone.get_output_embeddings().weight.dtype == torch.float32
    model = QwenSPAPRM(backbone)
    model.head.load_state_dict(torch.load(frozen / 'head.pt', map_location='cpu', weights_only=True))
    model.strict_prefix_scoring = True
    model.eval().requires_grad_(False)
    results = []
    with torch.inference_mode():
        for audit in selected:
            row = rows[audit['trajectory_sha256']]
            started = time.monotonic()
            scores = score_record(model, row, 'cpu').tolist()
            result = dict(trajectory_sha256=row['trajectory_sha256'], task_id=row['task_id'],
                          reward=row['reward'], original=audit['original_scores'],
                          full_int8=audit['quantized_scores'], selective_int8=scores,
                          seconds=time.monotonic()-started)
            results.append(result)
            print(json.dumps(result), flush=True)
    comparison = compare_scores([r['original'] for r in results],
                                [r['selective_int8'] for r in results], [r['reward'] for r in results])
    report = dict(diagnostic_only=True, deployment_authorized=False,
                  excluded_layers=['lm_head', 'scalar_head'], quantized_layers=len(targets),
                  rows=results, comparison=comparison)
    (root / 'selective_int8_diagnosis.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
