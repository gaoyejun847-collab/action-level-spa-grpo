"""Start independent SFT only after the specified GRPO run has completed."""
import argparse
import fcntl
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--after', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    lock = (output / 'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (output / 'adapter').exists(), 'Never overwrite an existing trained adapter'
    state = dict(pid=os.getpid(), initialization='original Qwen3-4B-Instruct-2507; fresh LoRA',
                 model=str(root / 'models/Qwen3-4B-Instruct-2507'), lora_rank=32, lora_alpha=64,
                 epochs=1, learning_rate=1e-4, global_batch_size=16, seed=42, prm=False,
                 after=str(args.after.resolve()), started_utc=datetime.now(timezone.utc).isoformat())
    def status(stage, **kwargs):
        state.update(stage=stage, **kwargs)
        temp = output / 'status.tmp'
        temp.write_text(json.dumps(state, indent=2))
        temp.replace(output / 'status.json')
        print(json.dumps(state), flush=True)
    try:
        report = json.loads((output / 'data_report.json').read_text())
        status('waiting_for_grpo_completion', examples=report['examples'])
        while True:
            previous = json.loads((args.after / 'status.json').read_text())
            if previous['stage'] == 'completed': break
            if previous['stage'] in ('failed', 'stopped_by_user'):
                raise RuntimeError('GRPO did not complete successfully; SFT was not started')
            proc = Path('/proc') / str(previous['pid'])
            if not proc.exists(): raise RuntimeError('GRPO supervisor exited without completed status')
            time.sleep(30)
        status('waiting_for_gpu_release')
        deadline = time.monotonic() + 600
        while True:
            memory = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used',
                                              '--format=csv,noheader,nounits'], text=True)
            values = [int(x.strip()) for x in memory.splitlines()]
            if len(values) == 4 and max(values) < 500: break
            if time.monotonic() > deadline: raise RuntimeError('GPUs still occupied; no workloads were killed')
            time.sleep(10)
        env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   MALLOC_ARENA_MAX='2', MALLOC_TRIM_THRESHOLD_='131072', TOKENIZERS_PARALLELISM='false',
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUDA_VISIBLE_DEVICES='0,1,2,3')
        command = [str(root / '.venv/bin/python'), '-m', 'torch.distributed.run', '--standalone',
                   '--nproc_per_node=4', '-m', 'spa_next.training.sft', 'train', '--model', state['model'],
                   '--output', str(output)]
        for stage, extra in [('memory_probe', ['--probe']), ('training', [])]:
            logfile = output / (stage + '.log')
            status(stage, active_log=str(logfile))
            (output / (stage + '.command.json')).write_text(json.dumps(command + extra, indent=2))
            with logfile.open('w') as log:
                subprocess.run(command + extra, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        assert (output / 'adapter/adapter_model.safetensors').stat().st_size > 0
        metrics = json.loads((output / 'trainer/train_results.json').read_text())
        status('completed', metrics=metrics, finished_utc=datetime.now(timezone.utc).isoformat())
    except Exception as exc:
        status('failed', error=str(exc))
        raise


if __name__ == '__main__': main()
