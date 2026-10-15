"""Run one bounded, independent V0 experiment and release its environment afterward."""
import json
import argparse
import os
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true', help='Partition all tasks and train one full epoch')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    name = ('v0-grpo-full-' if args.full else 'v0-grpo-trial16-') + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    dest = root / 'outputs' / name
    dest.mkdir(parents=True, exist_ok=False)
    state = dict(experiment=name, directory=str(dest), pid=os.getpid(),
                 model='Qwen3-4B-Instruct-2507', lora_rank=32, prm=False,
                 train_tasks=None if args.full else 64, validation_tasks=None if args.full else 16,
                 updates=None if args.full else 16, group_size=4, full_dataset=args.full,
                 initialization='base model, fresh LoRA', started_utc=datetime.now(timezone.utc).isoformat())
    def status(stage, **kwargs):
        state.update(stage=stage, **kwargs)
        temp = dest / 'status.tmp'
        temp.write_text(json.dumps(state, indent=2))
        temp.replace(dest / 'status.json')
        print(json.dumps(state), flush=True)
    service = None
    try:
        try:
            urlopen('http://127.0.0.1:8765/health', timeout=3).close()
        except OSError:
            pass
        else:
            raise RuntimeError('Port 8765 already serves an environment; refusing to interfere')
        status('starting_environment')
        with (dest / 'environment.log').open('w') as log:
            service = subprocess.Popen(['bash', 'spa_next/scripts/start_webshop.sh'], cwd=root,
                                       stdout=log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 600
        while True:
            if service.poll() is not None:
                raise RuntimeError('Environment startup failed; see environment.log')
            try:
                urlopen('http://127.0.0.1:8765/health', timeout=3).close()
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise TimeoutError('Environment did not become ready')
                time.sleep(5)
        status('preparing_data', environment_pid=service.pid)
        python = str(root / '.venv/bin/python')
        with (dest / 'prepare.log').open('w') as log:
            subprocess.run([python, '-m', 'spa_next.prepare_data', '--out', str(dest / 'data'),
                            *(['--all-tasks'] if args.full else ['--train-limit', '64', '--val-limit', '16'])], cwd=root,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        manifest = json.loads((dest / 'data/manifest.json').read_text())
        train_count, val_count = len(manifest['splits']['train']), len(manifest['splits']['val'])
        updates = train_count // 4
        # Verify every prompt fits instead of letting verl silently filter tasks.
        if args.full:
            status('checking_data', train_tasks=train_count, validation_tasks=val_count,
                   test_tasks=len(manifest['splits']['test']), updates=updates)
            with (dest / 'data_check.json').open('w') as log:
                subprocess.run([python, '-m', 'spa_next.check_dataset', str(dest / 'data'),
                                '--model', str(root / 'models/Qwen3-4B-Instruct-2507')], cwd=root,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        env = dict(os.environ, SPA_TRAJECTORY_DIR=str(dest / 'trajectories'))
        command = ['bash', 'spa_next/scripts/train_v0.sh',
                   f'data.train_files={dest}/data/train.parquet',
                   f'data.val_files={dest}/data/val.parquet',
                   f'trainer.default_local_dir={dest}/checkpoints',
                   f'trainer.experiment_name={name}',
                   f'trainer.total_training_steps={updates}', 'trainer.total_epochs=1',
                   f'trainer.save_freq={20 if args.full else 4}',
                   f'trainer.test_freq={1000 if args.full else 8}',
                   'trainer.resume_mode=disable', 'trainer.val_before_train=true']
        if args.full:
            command += ['data.filter_overlong_prompts=false', 'trainer.max_actor_ckpt_to_keep=3']
        (dest / 'command.json').write_text(json.dumps(command, indent=2))
        status('training')
        with (dest / 'train.log').open('w') as log:
            result = subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'Training exited {result.returncode}; see train.log')
        with (dest / 'trajectory_audit.json').open('w') as log:
            subprocess.run([python, '-m', 'spa_next.audit_trajectories', str(dest / 'trajectories')],
                           cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
        if args.full:
            # Test only the final checkpoint, once; test scores never drive updates.
            status('final_test')
            test_command = command + [f'data.val_files={dest}/data/test.parquet',
                f'trainer.default_local_dir={dest}/test-evaluation',
                f'trainer.experiment_name={name}-test', 'trainer.resume_mode=resume_path',
                f'trainer.resume_from_path={dest}/checkpoints/global_step_{updates}',
                'trainer.val_only=true', 'trainer.val_before_train=true']
            test_env = dict(env, SPA_TRAJECTORY_DIR=str(dest / 'test-trajectories'))
            (dest / 'test_command.json').write_text(json.dumps(test_command, indent=2))
            with (dest / 'test.log').open('w') as log:
                subprocess.run(test_command, cwd=root, env=test_env,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            with (dest / 'test_audit.json').open('w') as log:
                subprocess.run([python, '-m', 'spa_next.audit_trajectories', str(dest / 'test-trajectories')],
                               cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
        status('completed', finished_utc=datetime.now(timezone.utc).isoformat())
    except Exception as exc:
        status('failed', error=str(exc), finished_utc=datetime.now(timezone.utc).isoformat())
        traceback.print_exc()
        raise
    finally:
        if service is not None and service.poll() is None:
            service.terminate()
            service.wait(timeout=30)


if __name__ == '__main__':
    main()
