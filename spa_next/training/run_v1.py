"""Supervise the reviewed SFT -> PRM gates -> three bounded V1 experiments.

Every stage has an exact command, log and completion receipt. Failed gates halt
the pipeline. A receipt is reusable only with the identical command; partially
completed training is never silently restarted or treated as completed.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from datetime import datetime, timezone
from urllib.request import urlopen


def utc():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


class Pipeline:
    def __init__(self, args):
        self.root = Path(__file__).resolve().parents[2]
        self.dest = args.output.resolve()
        self.dest.mkdir(parents=True, exist_ok=True)
        self.lock = (self.dest / 'run.lock').open('a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.python = str(self.root / '.venv/bin/python')
        self.model = args.model.resolve()
        self.source = args.v0.resolve()
        self.prm = self.dest / 'prm'
        self.prm_resume = args.prm_resume
        self.prm_gpus = args.prm_gpus
        self.use_trained_prm = args.use_trained_prm
        self.env_service = self.prm_service = None
        self.child = None
        self.env = dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1',
            MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MALLOC_ARENA_MAX='2',
            MALLOC_TRIM_THRESHOLD_='131072', TOKENIZERS_PARALLELISM='false',
            HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUDA_VISIBLE_DEVICES=args.prm_gpus,
            SPA_ENV_URL='http://127.0.0.1:8765', SPA_MAX_STEPS='10', SPA_MAX_ACTION_TOKENS='256')
        self.state = dict(pid=os.getpid(), started_utc=utc(), directory=str(self.dest),
            model=str(self.model), source_v0=str(self.source), plan='SFT776 -> frozen Qwen SPA PRM -> A/B/C',
            policy_updates_per_experiment=128, train_tasks=512, group_size=4,
            final_validation_tasks=1209, test_evaluation=False)

    def status(self, stage, **values):
        self.state.update(stage=stage, updated_utc=utc(), **values)
        atomic_json(self.dest / 'status.json', self.state)
        print(json.dumps(self.state), flush=True)

    def resources(self, stage):
        memory = {line.split(':')[0]: line.split(':')[1].strip()
                  for line in Path('/proc/meminfo').read_text().splitlines()
                  if line.startswith(('MemAvailable:', 'MemTotal:'))}
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=15)
        with (self.dest / 'resources.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(time=utc(), stage=stage, memory=memory,
                                         gpu=gpu.stdout.strip())) + '\n')

    def run(self, stage, command, trajectory=None):
        receipts = self.dest / 'receipts'
        receipts.mkdir(exist_ok=True)
        receipt = receipts / (stage + '.json')
        digest = hashlib.sha256(json.dumps(command).encode()).hexdigest()
        if receipt.exists():
            previous = json.loads(receipt.read_text())
            if previous['command_sha256'] != digest or previous['returncode'] != 0:
                raise ValueError('Stage receipt cannot be reused: ' + stage)
            self.status(stage + '_already_completed')
            return
        if stage.startswith(('policy_smoke_', 'training_')) and trajectory is not None:
            checkpoint_root = trajectory.parent / 'checkpoints'
            if any(trajectory.glob('*.json')) or any(checkpoint_root.glob('global_step_*')):
                raise RuntimeError('Incomplete existing policy run requires explicit checkpoint recovery; '
                                   'refusing a fresh restart into ' + str(trajectory.parent))
        if stage in ('prm_training', 'prm_memory_probe'):
            training_output = Path(command[command.index('--output') + 1])
            if (training_output / 'training_config.json').exists() and '--resume' not in command:
                raise RuntimeError('Incomplete existing PRM run requires explicit recovery; '
                                   'preserved outputs: ' + str(training_output))
        logs = self.dest / 'logs'
        logs.mkdir(exist_ok=True)
        logpath = logs / (stage + '.log')
        if logpath.exists():
            # Keep evidence of prior failed attempts; do not discard training data.
            logpath = logs / (stage + '-' + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S') + '.log')
        atomic_json(logpath.with_suffix('.command.json'), command)
        env = dict(self.env)
        if stage == 'prm_training':
            env['CUDA_VISIBLE_DEVICES'] = self.prm_gpus
        if trajectory:
            env['SPA_TRAJECTORY_DIR'] = str(trajectory)
        started = time.monotonic()
        self.status(stage, active_log=str(logpath))
        with logpath.open('w') as log:
            process = subprocess.Popen(command, cwd=self.root, env=env,
                stdout=log, stderr=subprocess.STDOUT)
            self.child = process
            self.status(stage, child_pid=process.pid)
            while process.poll() is None:
                self.resources(stage)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    continue
        event = dict(stage=stage, command_sha256=digest, returncode=process.returncode,
                     elapsed_seconds=time.monotonic()-started, finished_utc=utc(), log=str(logpath))
        with (self.dest / 'stages.jsonl').open('a') as stream:
            stream.write(json.dumps(event) + '\n')
        if process.returncode:
            raise RuntimeError(f'{stage} exited {process.returncode}; see {logpath}')
        self.child = None
        atomic_json(receipt, event)

    def start_service(self, name, command, port, timeout):
        url = f'http://127.0.0.1:{port}/health'
        try:
            urlopen(url, timeout=3).close()
        except OSError:
            pass
        else:
            raise RuntimeError(f'Port {port} is already occupied; no other service was modified')
        logpath = self.dest / (name + '.log')
        self.status('starting_' + name, active_log=str(logpath))
        with logpath.open('a') as log:
            process = subprocess.Popen(command, cwd=self.root, env=self.env,
                stdout=log, stderr=subprocess.STDOUT)
        # Register ownership before waiting, so failure cleanup also stops it.
        if name == 'environment':
            self.env_service = process
        else:
            self.prm_service = process
        deadline = time.monotonic() + timeout
        while process.poll() is None:
            try:
                with urlopen(url, timeout=3) as response:
                    return json.load(response)
            except OSError:
                if time.monotonic() > deadline:
                    break
                time.sleep(5)
        raise RuntimeError(name + ' did not become ready; see ' + str(logpath))

    @staticmethod
    def stop_service(process):
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)

    def policy_command(self, variant, directory, validation='val', eval_only=False, resume=None, extra=()):
        cmd = [self.python, '-m', 'spa_next.training.verl_v1_launch', '--experiment', variant,
            '--model', str(self.model), '--train', str(self.dest / 'data/train.parquet'),
            '--val', str(self.dest / ('data/' + validation + '.parquet')),
            '--output', str(directory / 'checkpoints')]
        if variant == 'C':
            quant = json.loads((self.prm / 'frozen/quantization_report.json').read_text())
            if not quant['gate_passed']:
                raise ValueError('PRM INT8 gate failed')
            cmd += ['--prm-url', 'http://127.0.0.1:8766', '--prm-version', quant['model_version']]
        if eval_only:
            cmd += ['--eval-only']
        if resume:
            cmd += ['--resume-path', str(resume)]
        return cmd + list(extra)

    def execute(self):
        from spa_next.training.sft_snapshot import verify_snapshot
        try:
            self.status('verifying_sft_snapshot')
            snapshot = verify_snapshot(self.model)
            self.state['sft_version'] = snapshot['version']
            self.run('prepare_policy_data', [self.python, '-m', 'spa_next.training.prepare_v1',
                '--split-dir', str(self.source / 'data'), '--output', str(self.dest / 'data')])
            self.run('unit_tests', [self.python, '-m', 'unittest', 'discover', '-s', 'spa_next/tests'])
            self.run('verl_patch', [self.python, '-m', 'spa_next.training.patch_verl_v1'])
            if not all((self.dest / 'receipts' / (name + '.json')).exists()
                       for name in ('policy_smoke_A', 'unified_sft_baseline')):
                self.start_service('environment', ['bash', 'spa_next/scripts/start_webshop.sh'], 8765, 600)
            smoke = self.dest / 'smoke-A'
            self.run('policy_smoke_A', self.policy_command('A', smoke, extra=[
                'trainer.total_training_steps=2', 'trainer.val_before_train=false',
                'trainer.test_freq=-1', 'trainer.save_freq=1']), smoke / 'trajectories')
            baseline = self.dest / 'baseline'
            self.run('unified_sft_baseline', self.policy_command('A', baseline,
                validation='val_full', eval_only=True), baseline / 'trajectories')
            self.stop_service(self.env_service)
            self.env_service = None
            self.run('prm_prepare', [self.python, '-m', 'spa_next.prm', 'prepare', '--model', str(self.model),
                '--split-dir', str(self.source / 'data'), '--trajectories', str(self.source / 'trajectories'),
                '--output', str(self.prm / 'data')])
            training = [self.python, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
                '-m', 'spa_next.prm', 'train', '--model', str(self.model), '--data', str(self.prm / 'data')]
            self.run('prm_memory_probe', training + ['--output', str(self.prm / 'probe'), '--probe'])
            resumed_training = list(training)
            resumed_training[resumed_training.index('--nproc_per_node=4')] = '--nproc_per_node=' + str(len(self.prm_gpus.split(',')))
            resumed_training += ['--output', str(self.prm / 'training'), '--epochs', '3']
            if self.prm_resume:
                resumed_training += ['--resume', str(self.prm_resume.resolve())]
            if self.use_trained_prm:
                completed = json.loads((self.prm / 'training/training_report.json').read_text())
                checkpoint = Path(completed['best_checkpoint'])
                if (completed.get('status') != 'completed' or not completed.get('gate_passed')
                        or completed.get('smoke_only') or completed.get('updates') != 6090
                        or Path(completed['config']['base']).resolve() != self.model
                        or not (checkpoint / 'adapter/adapter_model.safetensors').is_file()
                        or not (checkpoint / 'head.pt').is_file()):
                    raise RuntimeError('Completed PRM report/checkpoint is not qualified for reuse')
                self.status('prm_training_already_completed', prm_checkpoint=str(checkpoint))
            else:
                self.run('prm_training', resumed_training)
            report = json.loads((self.prm / 'training/training_report.json').read_text())
            if not report['gate_passed']:
                raise RuntimeError('PRM held-out MSE did not beat the constant baseline; policy training stopped')
            self.run('prm_export', [self.python, '-m', 'spa_next.prm', 'export', '--model', str(self.model),
                '--training', str(self.prm / 'training'), '--output', str(self.prm / 'frozen')])
            if (self.prm / 'frozen/quantization_report.json').exists():
                from spa_next.prm.deployment import validate_artifact
                validate_artifact(self.prm / 'frozen')
                self.status('prm_quantization_report_reused')
            else:
                self.run('prm_quantization_gate', [self.python, '-m', 'spa_next.prm', 'verify',
                    '--model', str(self.prm / 'frozen'), '--data', str(self.prm / 'data'),
                    '--max-examples', '256', '--cpu-threads', '8'])
            quant = json.loads((self.prm / 'frozen/quantization_report.json').read_text())
            if not quant['gate_passed']:
                raise RuntimeError('PRM INT8 quantization gates failed; policy training stopped')
            self.state['prm_version'] = quant['model_version']
            if len(self.prm_gpus.split(',')) != 4:
                raise RuntimeError('PRM finished; policy GPU/batch adaptation must be validated before launching A/B/C. GPU 3 remains reserved.')
            self.start_service('environment', ['bash', 'spa_next/scripts/start_webshop.sh'], 8765, 600)
            self.start_service('prm_service', [self.python, '-m', 'spa_next.prm', 'serve',
                '--model', str(self.prm / 'frozen'), '--host', '127.0.0.1', '--port', '8766',
                '--cpu-threads', '8'], 8766, 1200)
            for variant in ('C', 'A', 'B'):
                smoke = self.dest / ('smoke-' + variant)
                self.run('policy_smoke_' + variant, self.policy_command(variant, smoke, extra=[
                    'trainer.total_training_steps=2', 'trainer.val_before_train=false',
                    'trainer.test_freq=-1', 'trainer.save_freq=1']), smoke / 'trajectories')
                run = self.dest / variant
                self.run('training_' + variant, self.policy_command(variant, run), run / 'trajectories')
                checkpoint = run / 'checkpoints/global_step_128'
                if not checkpoint.is_dir():
                    raise RuntimeError('Final checkpoint missing for ' + variant)
                validation = run / 'full-validation'
                self.run('full_validation_' + variant, self.policy_command(variant, validation,
                    validation='val_full', eval_only=True, resume=checkpoint), validation / 'trajectories')
            self.run('comparison', [self.python, '-m', 'spa_next.training.report_v1', '--output', str(self.dest)])
            self.status('completed', finished_utc=utc(), active_log=None, child_pid=None)
        except BaseException as exc:
            self.status('failed', failed_stage=self.state.get('stage'), error=str(exc), finished_utc=utc())
            raise
        finally:
            self.stop_service(self.child)
            self.stop_service(self.prm_service)
            self.stop_service(self.env_service)


def main():
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Supervisor interrupted by signal ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--v0', type=Path, required=True)
    parser.add_argument('--prm-gpus', default='0,1,2,3')
    parser.add_argument('--prm-resume', type=Path)
    parser.add_argument('--use-trained-prm', action='store_true')
    Pipeline(parser.parse_args()).execute()


if __name__ == '__main__':
    main()
