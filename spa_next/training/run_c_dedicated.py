"""Run C with three policy GPUs and one original-precision frozen PRM GPU."""
import argparse
import json
from pathlib import Path
import signal
from spa_next.training.run_v1 import Pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--v0', type=Path, required=True)
    args = parser.parse_args()
    args.prm_gpus, args.prm_resume, args.use_trained_prm = '0,1,2', None, True
    pipeline = Pipeline(args)
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Dedicated pipeline interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        from spa_next.training.sft_snapshot import verify_snapshot
        pipeline.status('verifying_sft_snapshot')
        pipeline.state.update(sft_version=verify_snapshot(pipeline.model)['version'],
                              plan='C first: three policy GPUs + original BF16 PRM GPU 3')
        pipeline.run('three_gpu_tests', [pipeline.python, '-m', 'unittest', 'discover', '-s', 'spa_next/tests'])
        pipeline.run('three_gpu_patch', [pipeline.python, '-m', 'spa_next.training.patch_verl_three_gpu'])
        pipeline.run('memory_safe_logprob', [pipeline.python, '-m', 'spa_next.training.patch_logprob_chunking'])
        pipeline.run('memory_safe_entropy', [pipeline.python, '-m', 'spa_next.training.patch_entropy_chunking'])
        pipeline.run('memory_safe_backward', [pipeline.python, '-m', 'spa_next.training.patch_logprob_backward'])
        pipeline.run('fused_frozen_head_patch', [pipeline.python, '-m', 'spa_next.training.patch_fused_frozen_head'])
        pipeline.run('rollout_corr_padding_patch', [pipeline.python, '-m', 'spa_next.training.patch_rollout_corr_padding'])
        pipeline.env['CUDA_VISIBLE_DEVICES'] = '3'
        pipeline.run('prm_original_gpu_gate', [pipeline.python, '-m', 'spa_next.prm.gpu_backend',
            '--model', str(pipeline.prm / 'frozen'), '--data', str(pipeline.prm / 'data'), '--device', 'cuda:0'])
        report = json.loads((pipeline.prm / 'frozen/gpu_reference_report.json').read_text())
        if not report['gate_passed']:
            raise RuntimeError('Original GPU PRM qualification failed')
        pipeline.state['prm_version'] = report['model_version']
        pipeline.start_service('prm_service', [pipeline.python, '-m', 'spa_next.prm', 'serve',
            '--model', str(pipeline.prm / 'frozen'), '--host', '127.0.0.1', '--port', '8766',
            '--backend', 'original-gpu', '--device', 'cuda:0'], 8766, 600)
        pipeline.env['CUDA_VISIBLE_DEVICES'] = '0,1,2'
        pipeline.start_service('environment', ['bash', 'spa_next/scripts/start_webshop.sh'], 8765, 600)
        def command(directory, validation='val', extra=()):
            return [pipeline.python, '-m', 'spa_next.training.verl_v1_launch', '--experiment', 'C',
                '--model', str(pipeline.model), '--train', str(pipeline.dest / 'data/train.parquet'),
                '--val', str(pipeline.dest / ('data/' + validation + '.parquet')),
                '--output', str(directory / 'checkpoints'), '--prm-url', 'http://127.0.0.1:8766',
                '--prm-version', report['model_version'], '--policy-gpus', '3'] + list(extra)
        smoke = pipeline.dest / 'smoke-C-gpu3-memorysafe10'
        pipeline.run('policy_smoke_C_gpu3_memorysafe10', command(smoke, extra=[
            'trainer.total_training_steps=2', 'trainer.val_before_train=false',
            'trainer.test_freq=-1', 'trainer.save_freq=1']), smoke / 'trajectories')
        # Previous C attempt stopped before update 1 while writing validation
        # metrics. Preserve its step-0 snapshot and restart the same SFT seed.
        run = pipeline.dest / 'C-jsonfix'
        pipeline.run('training_C_jsonfix', command(run), run / 'trajectories')
        validation = run / 'full-validation'
        pipeline.run('full_validation_C_jsonfix', command(validation, 'val_full', [
            '--eval-only', '--resume-path', str(run / 'checkpoints/global_step_128')]), validation / 'trajectories')
        pipeline.status('C_completed', child_pid=None, active_log=None)
    except BaseException as exc:
        pipeline.status('failed', failed_stage=pipeline.state.get('stage'), error=str(exc))
        raise
    finally:
        pipeline.stop_service(pipeline.child)
        pipeline.stop_service(pipeline.prm_service)
        pipeline.stop_service(pipeline.env_service)


if __name__ == '__main__':
    main()
