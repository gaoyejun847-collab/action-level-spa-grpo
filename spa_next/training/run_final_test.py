"""Test the validation-selected final C model and aggregate the three models."""
import argparse
import csv
import json
from pathlib import Path
import re
import signal
from types import SimpleNamespace

from spa_next.training.run_v1 import Pipeline, atomic_json
from spa_next.training.sft_snapshot import sha256


def metrics(log):
    text = Path(log).read_text(errors='replace')
    result = {}
    for name, key in [('success_rate', 'val-aux/webshop/success/mean@1'),
                      ('average_reward', 'val-core/webshop/reward/mean@1')]:
        matches = re.findall(re.escape(key) + r':([0-9.eE+-]+)', text)
        if not matches:
            raise RuntimeError('Missing completed evaluation metric: ' + key)
        result[name] = float(matches[-1])
        if not 0 <= result[name] <= 1:
            raise RuntimeError('Invalid evaluation metric')
    return result


def main():
    import yaml
    import pyarrow.parquet as pq
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.resolve()
    state = json.loads((root / 'status.json').read_text())
    assert state['stage'] == 'completed'
    best = json.loads((root / 'checkpoints/best_validation.json').read_text())
    checkpoint = Path(best['checkpoint'])
    assert (checkpoint / 'data.pt').is_file()
    baseline = json.loads((root / 'baseline_test_plan.json').read_text())
    test = Path(baseline['test_path'])
    assert sha256(test) == baseline['test_sha256']
    count = pq.read_metadata(test).num_rows
    dest = root / ('test-final-step-' + str(best['step']))
    pipeline = Pipeline(SimpleNamespace(output=dest, model=Path(state['model']),
        v0=Path(state['source_v0']), prm_resume=None, prm_gpus='0,1,2', use_trained_prm=True))
    pipeline.state.update(plan='Three-model frozen test comparison; no training',
        policy_updates_per_experiment=0, test_evaluation=True, test_tasks=count,
        checkpoint=str(checkpoint), selected_step=best['step'])
    rows = []
    # Verify the historical tests used the same task file and inference settings.
    reference = yaml.safe_load((root / 'full-validation-best/checkpoints/resolved_config.yaml').read_text())
    paths = ['data.max_prompt_length', 'data.max_response_length', 'data.val_batch_size',
             'actor_rollout_ref.rollout.val_kwargs', 'actor_rollout_ref.rollout.tensor_model_parallel_size',
             'actor_rollout_ref.rollout.agent', 'trainer.n_gpus_per_node']
    def value(config, path):
        for part in path.split('.'):
            config = config[part]
        return config
    for name, folder, step in [('SFT', 'test-SFT', 0), ('V0_GRPO', 'test-V0_GRPO', 2417)]:
        receipt = json.loads((root / 'receipts' / ('test_' + name + '.json')).read_text())
        assert receipt['returncode'] == 0
        config = yaml.safe_load((root / folder / 'checkpoints/resolved_config.yaml').read_text())
        assert config['data']['val_files'] == str(test)
        for path in paths:
            if value(config, path) != value(reference, path):
                raise RuntimeError('Baseline configuration mismatch: ' + name + ' ' + path)
        rows.append(dict(model=name, step=step, tasks=count, reused=True,
                         log=receipt['log'], **metrics(receipt['log'])))
    assert sha256(Path(baseline['grpo_adapter']) / 'adapter_model.safetensors') == baseline['grpo_adapter_sha256']
    atomic_json(dest / 'comparison_plan.json', dict(test_path=str(test), test_sha256=baseline['test_sha256'],
        tasks=count, selected_checkpoint=str(checkpoint), baseline_results=rows,
        matched_config_fields=paths, selection='checkpoint fixed using validation, never test scores'))
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Final test interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        pipeline.start_service('environment', ['bash', 'spa_next/scripts/start_webshop.sh'], 8765, 600)
        pipeline.run('test_final_C', [pipeline.python, '-m', 'spa_next.training.verl_v1_launch',
            '--experiment', 'C', '--model', str(pipeline.model), '--train', str(root / 'data/train.parquet'),
            '--val', str(test), '--output', str(dest / 'checkpoints'), '--policy-gpus', '3',
            '--eval-only', '--resume-path', str(checkpoint)], dest / 'trajectories')
        rows.append(dict(model='SPA_GRPO', step=best['step'], tasks=count, reused=False,
            log=str(dest / 'logs/test_final_C.log'), **metrics(dest / 'logs/test_final_C.log')))
        atomic_json(dest / 'three_model_results.json', dict(test_sha256=baseline['test_sha256'], results=rows))
        with (dest / 'three_model_results.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        pipeline.status('completed', child_pid=None, active_log=None, results=rows)
    except BaseException as exc:
        pipeline.status('failed', failed_stage=pipeline.state.get('stage'), error=str(exc))
        raise
    finally:
        pipeline.stop_service(pipeline.child)
        pipeline.stop_service(pipeline.env_service)


if __name__ == '__main__':
    main()
