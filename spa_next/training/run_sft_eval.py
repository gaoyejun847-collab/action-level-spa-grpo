"""Own the WebShop service, run a smoke evaluation, then full validation."""
import argparse
import fcntl
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--adapter', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    args.output.mkdir(parents=True, exist_ok=False)
    lock = (args.output / 'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {'pid': os.getpid(), 'adapter': str(args.adapter.resolve()),
             'data': str(args.data.resolve()), 'tasks': 1209,
             'started_utc': datetime.now(timezone.utc).isoformat()}
    def status(stage, **extra):
        state.update(stage=stage, **extra)
        temp = args.output / 'status.tmp'
        temp.write_text(json.dumps(state, indent=2)); temp.replace(args.output / 'status.json')
        print(json.dumps(state), flush=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    service = None
    try:
        try:
            urlopen('http://127.0.0.1:8765/health', timeout=3)
        except OSError:
            pass
        else:
            raise RuntimeError('Environment port 8765 is already occupied')
        status('starting_environment')
        with (args.output / 'environment.log').open('w') as log:
            service = subprocess.Popen(['bash', 'spa_next/scripts/start_webshop.sh'], cwd=root,
                                       env=env, stdout=log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 600
        while True:
            if service.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError('Environment failed to start')
            try:
                urlopen('http://127.0.0.1:8765/health', timeout=3).close(); break
            except OSError:
                time.sleep(5)
        base = [str(root/'.venv/bin/python'), '-m', 'torch.distributed.run', '--standalone',
                '--nproc_per_node=4', '-m', 'spa_next.eval_sft', '--model',
                str(root/'models/Qwen3-4B-Instruct-2507'), '--adapter', str(args.adapter.resolve()),
                '--data', str(args.data.resolve())]
        for stage, folder, extra in [('smoke', args.output/'smoke-trajectories', ['--limit','8']),
                                     ('evaluating', args.output/'trajectories', [])]:
            log_path = args.output / (stage + '.log')
            status(stage, active_log=str(log_path))
            with log_path.open('w') as log:
                subprocess.run(base + ['--output', str(folder)] + extra, cwd=root, env=env,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            files = list(folder.glob('rank-*.jsonl'))
            count = sum(len(f.read_text().splitlines()) for f in files)
            expected = 8 if stage == 'smoke' else 1209
            if count != expected: raise RuntimeError(f'{stage}: expected {expected}, got {count}')
        records = [json.loads(line) for f in (args.output/'trajectories').glob('rank-*.jsonl')
                   for line in f.read_text().splitlines()]
        metrics = {'tasks': len(records),
                   'reward_mean': sum(x['outcome_reward'] for x in records)/len(records),
                   'success_mean': sum(x['success'] for x in records)/len(records),
                   'steps_mean': sum(x['steps'] for x in records)/len(records),
                   'budget_exhausted_mean': sum(x['budget_exhausted'] for x in records)/len(records)}
        (args.output/'metrics.json').write_text(json.dumps(metrics, indent=2))
        status('completed', metrics=metrics, finished_utc=datetime.now(timezone.utc).isoformat())
    except Exception as exc:
        status('failed', error=str(exc)); raise
    finally:
        if service is not None and service.poll() is None:
            service.terminate(); service.wait(timeout=30)


if __name__ == '__main__': main()
