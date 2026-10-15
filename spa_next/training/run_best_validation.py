"""Evaluate the previously selected C checkpoint on the frozen full validation set."""
import argparse
import json
from pathlib import Path
import signal
from types import SimpleNamespace
from spa_next.training.run_v1 import Pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-root', type=Path, required=True)
    args = parser.parse_args()
    root = args.experiment_root.resolve()
    completed = json.loads((root / 'status.json').read_text())
    if completed['stage'] != 'C_completed':
        raise RuntimeError('C training and final validation must be complete first')
    best = json.loads((root / 'C-jsonfix/checkpoints/best_validation.json').read_text())
    checkpoint = Path(best['checkpoint']).resolve()
    checkpoint.relative_to(root / 'C-jsonfix/checkpoints/preserved_best')
    if not (checkpoint / 'data.pt').is_file():
        raise RuntimeError('Selected checkpoint is incomplete')
    dest = root / 'C-jsonfix' / ('full-validation-best-step-' + str(best['step']))
    pipeline = Pipeline(SimpleNamespace(output=dest, model=Path(completed['model']),
        v0=Path(completed['source_v0']), prm_resume=None, prm_gpus='0,1,2', use_trained_prm=True))
    pipeline.state.update(plan='Evaluate validation-selected C checkpoint; no policy updates',
        policy_updates_per_experiment=0, selected_step=best['step'], checkpoint=str(checkpoint),
        sft_version=completed['sft_version'], final_validation_tasks=1209)
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Evaluation interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        pipeline.start_service('environment', ['bash', 'spa_next/scripts/start_webshop.sh'], 8765, 600)
        command = [pipeline.python, '-m', 'spa_next.training.verl_v1_launch',
            '--experiment', 'C', '--model', completed['model'],
            '--train', str(root / 'data/train.parquet'), '--val', str(root / 'data/val_full.parquet'),
            '--output', str(dest / 'checkpoints'), '--policy-gpus', '3',
            '--eval-only', '--resume-path', str(checkpoint)]
        pipeline.run('full_validation_best', command, dest / 'trajectories')
        pipeline.status('best_validation_completed', child_pid=None, active_log=None)
    except BaseException as exc:
        pipeline.status('failed', failed_stage=pipeline.state.get('stage'), error=str(exc))
        raise
    finally:
        pipeline.stop_service(pipeline.child)
        pipeline.stop_service(pipeline.env_service)


if __name__ == '__main__':
    main()
