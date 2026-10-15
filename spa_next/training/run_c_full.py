"""Continue C on all isolated tasks after the already-running validation."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import time
from types import SimpleNamespace

from spa_next.training.prepare_v1 import select_rows
from spa_next.training.run_v1 import Pipeline, atomic_json
from spa_next.training.sft_snapshot import sha256


def prepare(root, dest):
    import pyarrow as pa
    import pyarrow.parquet as pq
    previous = json.loads((root / 'status.json').read_text())
    manifest = json.loads((root / 'data/manifest.json').read_text())
    source = Path(previous['source_v0']) / 'data'
    for name, digest in manifest['source_sha256'].items():
        if sha256(source / (name + '.parquet')) != digest:
            raise RuntimeError('Source split changed: ' + name)
    splits = {name: pq.read_table(source / (name + '.parquet')).to_pylist()
              for name in ('train', 'val', 'test')}
    selected, count = select_rows(splits, train_count=manifest['eligible_train_tasks'])
    ids = [int(row['extra_info']['task_id']) for row in selected['train']]
    assert ids[:512] == manifest['tasks']['train']
    # Fill the final task batch explicitly; no isolated task is dropped.
    repeat = (-count) % 4
    selected['train'].extend(selected['train'][:repeat])
    data = dest / 'data'
    data.mkdir(parents=True, exist_ok=False)
    for name, rows in selected.items():
        pq.write_table(pa.Table.from_pylist(rows), data / (name + '.parquet'))
    report = dict(unique_train_tasks=count, train_rows=count + repeat,
        repeated_task_ids=ids[:repeat], updates=(count + repeat) // 4,
        start_step=128, final_step=128 + (count + repeat) // 4,
        task_ids=ids, source_sha256=manifest['source_sha256'], seed=42,
        target_validation_success=0.70, target_is_guarantee=False,
        parquet_sha256={name: sha256(data / (name + '.parquet')) for name in selected},
        resume_data_state='reset: expanded dataset, preserve policy/optimizer/RNG',
        test_used_for='identity exclusion only')
    # Copy optimizer/RNG/model, deliberately omit old 512-task loader state.
    # Copies keep the original checkpoint immutable under checkpoint loading.
    resume = dest / 'resume/global_step_128'
    shutil.copytree(root / 'C-jsonfix/checkpoints/global_step_128/actor', resume / 'actor')
    atomic_json(data / 'manifest.json', report)
    return previous, report


def wait_for_validation(pipeline, path):
    pipeline.status('queued_after_best_validation', waiting_for=str(path))
    deadline = time.monotonic() + 24 * 3600
    while time.monotonic() < deadline:
        state = json.loads(path.read_text())
        if state['stage'] == 'failed':
            raise RuntimeError('Preceding validation failed; inspect before training')
        if state['stage'] == 'best_validation_completed':
            # Completion is written before finally closes the environment.
            with socket.socket() as sock:
                if sock.connect_ex(('127.0.0.1', 8765)) != 0:
                    return
        else:
            os.kill(state['pid'], 0)
        time.sleep(30)
    raise TimeoutError('Preceding validation did not release resources in 24h')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--test-before-training', action='store_true')
    parser.add_argument('--test-baselines', action='store_true')
    parser.add_argument('--resume-step', type=int)
    args = parser.parse_args()
    if args.test_baselines and not args.test_before_training:
        parser.error('--test-baselines requires --test-before-training')
    root, dest = args.experiment_root.resolve(), args.output.resolve()
    previous = json.loads((root / 'status.json').read_text())
    pipeline = Pipeline(SimpleNamespace(output=dest, model=Path(previous['model']),
        v0=Path(previous['source_v0']), prm_resume=None, prm_gpus='0,1,2', use_trained_prm=True))
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Full C continuation interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        manifest = dest / 'data/manifest.json'
        if not manifest.exists():
            previous, report = prepare(root, dest)
        else:
            report = json.loads(manifest.read_text())
            for name, digest in report['parquet_sha256'].items():
                assert sha256(dest / ('data/' + name + '.parquet')) == digest
        pipeline.state.update(plan='Continue C on all isolated training tasks, one full pass',
            train_tasks=report['unique_train_tasks'], policy_updates_per_experiment=report['updates'],
            start_step=128, final_step=report['final_step'], target_validation_success=0.70,
            sft_version=previous['sft_version'])
        resume_checkpoint = dest / 'resume/global_step_128'
        training_stage = 'training_C_full'
        if args.resume_step is not None:
            if not 128 < args.resume_step < report['final_step']:
                raise ValueError('Resume step outside continuation budget')
            resume_checkpoint = dest / 'checkpoints' / ('global_step_' + str(args.resume_step))
            required = [resume_checkpoint / 'data.pt', resume_checkpoint / 'actor/fsdp_config.json']
            for rank in range(3):
                required.extend(resume_checkpoint / 'actor' / (kind + '_world_size_3_rank_' + str(rank) + '.pt')
                                for kind in ('model', 'optim', 'extra_state'))
            if not all(p.is_file() and p.stat().st_size > 0 for p in required):
                raise RuntimeError('Resume checkpoint incomplete')
            pipeline.state.update(resumed_from_step=args.resume_step,
                remaining_updates=report['final_step'] - args.resume_step)
            training_stage = 'resume_C_full_' + str(args.resume_step)
        if args.prepare_only:
            pipeline.status('prepared')
            return
        test_plan = None
        if args.test_before_training:
            plan_path = dest / 'test_plan.json'
            if plan_path.exists():
                test_plan = json.loads(plan_path.read_text())
            else:
                best = json.loads((root / 'C-jsonfix/checkpoints/best_validation.json').read_text())
                test_path = pipeline.source / 'data/test.parquet'
                import pyarrow.parquet as pq
                test_plan = dict(checkpoint=best['checkpoint'], selected_step=best['step'],
                    selection='previously selected on validation; fixed before test evaluation',
                    test_path=str(test_path), test_sha256=sha256(test_path),
                    task_count=pq.read_metadata(test_path).num_rows,
                    purpose='one-time report only; no selection or tuning on test metrics')
                atomic_json(plan_path, test_plan)
            assert sha256(Path(test_plan['test_path'])) == report['source_sha256']['test']
            assert test_plan['test_sha256'] == report['source_sha256']['test']
            assert (Path(test_plan['checkpoint']) / 'data.pt').is_file()
            pipeline.state['before_training_test'] = test_plan
            if args.test_baselines:
                baseline_plan = dict(test_path=test_plan['test_path'], test_sha256=test_plan['test_sha256'],
                    sft_model=str(pipeline.model),
                    grpo_base=str(pipeline.root / 'models/Qwen3-4B-Instruct-2507'),
                    grpo_adapter=str(pipeline.source / 'checkpoints/global_step_2417/actor/lora_adapter'),
                    grpo_step=2417, selection='final V0 checkpoint; no test-based selection')
                baseline_plan['grpo_adapter_sha256'] = sha256(Path(baseline_plan['grpo_adapter']) / 'adapter_model.safetensors')
                plan_file = dest / 'baseline_test_plan.json'
                if plan_file.exists():
                    assert json.loads(plan_file.read_text()) == baseline_plan
                else:
                    atomic_json(plan_file, baseline_plan)
                pipeline.state['baseline_tests'] = baseline_plan
        wait_for_validation(pipeline, root / 'C-jsonfix/full-validation-best-step-64/status.json')
        pipeline.start_service('environment', ['bash', 'spa_next/scripts/start_webshop.sh'], 8765, 600)
        if test_plan:
            if args.test_baselines:
                for name, model, extra in [
                    ('SFT', baseline_plan['sft_model'], []),
                    ('V0_GRPO', baseline_plan['grpo_base'],
                     ['--evaluation-adapter', baseline_plan['grpo_adapter']])]:
                    evaluation = dest / ('test-' + name)
                    pipeline.run('test_' + name, [pipeline.python, '-m', 'spa_next.training.verl_v1_launch',
                        '--experiment', 'C', '--model', model, '--train', str(dest / 'data/train.parquet'),
                        '--val', test_plan['test_path'], '--output', str(evaluation / 'checkpoints'),
                        '--policy-gpus', '3', '--eval-only', 'trainer.resume_mode=disable'] + extra,
                        evaluation / 'trajectories')
            evaluation = dest / ('test-best-step-' + str(test_plan['selected_step']))
            pipeline.run('test_existing_C', [pipeline.python, '-m', 'spa_next.training.verl_v1_launch',
                '--experiment', 'C', '--model', str(pipeline.model),
                '--train', str(dest / 'data/train.parquet'), '--val', test_plan['test_path'],
                '--output', str(evaluation / 'checkpoints'), '--policy-gpus', '3',
                '--eval-only', '--resume-path', test_plan['checkpoint']], evaluation / 'trajectories')
        prm = root / 'prm/frozen'
        gate = json.loads((prm / 'gpu_reference_report.json').read_text())
        if not gate['gate_passed']:
            raise RuntimeError('PRM qualification missing')
        pipeline.state['prm_version'] = gate['model_version']
        pipeline.env['CUDA_VISIBLE_DEVICES'] = '3'
        pipeline.start_service('prm_service', [pipeline.python, '-m', 'spa_next.prm', 'serve',
            '--model', str(prm), '--host', '127.0.0.1', '--port', '8766',
            '--backend', 'original-gpu', '--device', 'cuda:0'], 8766, 600)
        pipeline.env['CUDA_VISIBLE_DEVICES'] = '0,1,2'
        def command(output, val, extra):
            return [pipeline.python, '-m', 'spa_next.training.verl_v1_launch', '--experiment', 'C',
                '--model', str(pipeline.model), '--train', str(dest / 'data/train.parquet'),
                '--val', str(dest / ('data/' + val + '.parquet')), '--output', str(output),
                '--policy-gpus', '3', '--prm-url', 'http://127.0.0.1:8766',
                '--prm-version', gate['model_version']] + extra
        pipeline.run(training_stage, command(dest / 'checkpoints', 'val', [
            '--resume-path', str(resume_checkpoint),
            'trainer.total_epochs=1', 'trainer.total_training_steps=' + str(report['final_step']),
            'trainer.test_freq=128', 'trainer.save_freq=20',
            'trainer.val_before_train=' + ('false' if args.resume_step is not None else 'true')]),
            dest / 'trajectories')
        best = json.loads((dest / 'checkpoints/best_validation.json').read_text())
        evaluation = dest / 'full-validation-best'
        pipeline.run('full_validation_best', command(evaluation / 'checkpoints', 'val_full', [
            '--eval-only', '--resume-path', best['checkpoint']]), evaluation / 'trajectories')
        pipeline.status('completed', child_pid=None, active_log=None)
    except BaseException as exc:
        pipeline.status('failed', failed_stage=pipeline.state.get('stage'), error=str(exc))
        raise
    finally:
        pipeline.stop_service(pipeline.child)
        pipeline.stop_service(pipeline.prm_service)
        pipeline.stop_service(pipeline.env_service)


if __name__ == '__main__':
    main()
