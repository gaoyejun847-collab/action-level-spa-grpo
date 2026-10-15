"""Resume an existing full run, verify two checkpoint cycles, then continue."""
import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    dest = args.directory.resolve()
    assert root / 'outputs' in dest.parents
    lock = (dest / 'resume.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = json.loads((dest / 'status.json').read_text())
    checkpoint = int((dest / 'checkpoints/latest_checkpointed_iteration.txt').read_text())
    target = state['updates']
    assert 0 < checkpoint < target
    attempt = dest / ('resume-' + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S'))
    attempt.mkdir()
    (attempt / 'previous_status.json').write_text(json.dumps(state, indent=2))
    state.update(pid=os.getpid(), resume_from=checkpoint, resume_directory=str(attempt),
                 resumed_utc=datetime.now(timezone.utc).isoformat())
    for key in ('error', 'finished_utc'):
        state.pop(key, None)
    def status(stage, **kwargs):
        state.update(stage=stage, **kwargs)
        temp = dest / 'status.tmp'
        temp.write_text(json.dumps(state, indent=2))
        temp.replace(dest / 'status.json')
        print(json.dumps(state), flush=True)
    service = None
    try:
        status('checking_resume')
        report = json.loads((dest / 'data_check.json').read_text())
        for split, info in report['splits'].items():
            assert hashlib.sha256((dest / 'data' / (split + '.parquet')).read_bytes()).hexdigest() == info['parquet_sha256']
        actor = dest / 'checkpoints' / f'global_step_{checkpoint}' / 'actor'
        for rank in range(4):
            for kind in ('model', 'optim', 'extra_state'):
                assert (actor / f'{kind}_world_size_4_rank_{rank}.pt').stat().st_size > 0
        # Completed updates must not be mixed with partial rollouts after a crash.
        archived = 0
        for path in (dest / 'trajectories').glob('*.json'):
            row = json.loads(path.read_text())
            if row['policy_version']['step'] > checkpoint:
                folder = attempt / 'uncommitted-trajectories'
                folder.mkdir(exist_ok=True)
                path.rename(folder / path.name)
                archived += 1
        env = dict(os.environ, MALLOC_ARENA_MAX='2', MALLOC_TRIM_THRESHOLD_='131072',
                   SPA_TRAJECTORY_DIR=str(dest / 'trajectories'))
        try:
            urlopen('http://127.0.0.1:8765/health', timeout=3).close()
        except OSError:
            pass
        else:
            raise RuntimeError('An environment already uses port 8765')
        status('starting_environment', archived_uncommitted=archived)
        with (attempt / 'environment.log').open('w') as log:
            service = subprocess.Popen(['bash', 'spa_next/scripts/start_webshop.sh'], cwd=root, env=env,
                                       stdout=log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 600
        while True:
            if service.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError('Environment startup failed')
            try:
                urlopen('http://127.0.0.1:8765/health', timeout=3).close()
                break
            except OSError:
                time.sleep(5)
        command = json.loads((dest / 'command.json').read_text())
        def run(command, filename, stage, trajectory=None):
            status(stage, environment_pid=service.pid, active_log=str(attempt / filename))
            (attempt / (filename + '.command.json')).write_text(json.dumps(command, indent=2))
            run_env = dict(env)
            if trajectory:
                run_env['SPA_TRAJECTORY_DIR'] = str(trajectory)
            with (attempt / filename).open('w') as log:
                subprocess.run(command, cwd=root, env=run_env, stdout=log, stderr=subprocess.STDOUT, check=True)
        probe_target = min(checkpoint + 2, target)
        run(command + ['trainer.resume_mode=auto', 'trainer.val_before_train=false',
                       f'trainer.total_training_steps={probe_target}', 'trainer.save_freq=1',
                       'trainer.test_freq=-1'], 'checkpoint-probe.log', 'resume_checkpoint_probe')
        assert int((dest / 'checkpoints/latest_checkpointed_iteration.txt').read_text()) == probe_target
        # The long run keeps the original validation/checkpoint schedule.
        if probe_target < target:
            run(command + ['trainer.resume_mode=auto', 'trainer.val_before_train=false'],
                'train.log', 'training')
        python = str(root / '.venv/bin/python')
        run([python, '-m', 'spa_next.audit_trajectories', str(dest / 'trajectories')],
            'trajectory_audit.json', 'auditing_training')
        run(command + [f'data.val_files={dest}/data/test.parquet',
            f'trainer.default_local_dir={dest}/test-evaluation',
            f'trainer.experiment_name={state["experiment"]}-test', 'trainer.resume_mode=resume_path',
            f'trainer.resume_from_path={dest}/checkpoints/global_step_{target}',
            'trainer.val_only=true', 'trainer.val_before_train=true'],
            'test.log', 'final_test', dest / 'test-trajectories')
        run([python, '-m', 'spa_next.audit_trajectories', str(dest / 'test-trajectories')],
            'test_audit.json', 'auditing_test')
        status('completed', finished_utc=datetime.now(timezone.utc).isoformat())
    except Exception as exc:
        status('failed', error=str(exc), finished_utc=datetime.now(timezone.utc).isoformat())
        raise
    finally:
        if service is not None and service.poll() is None:
            service.terminate()
            service.wait(timeout=30)


if __name__ == '__main__':
    main()
