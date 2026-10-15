"""Source-anchored extension of the installed V1 patch for uneven DP batches."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
from spa_next.training.patch_verl_v1 import replace_once

MARKER = '# SPA_THREE_GPU_PADDING_V1'


def transform(name, source):
    if name == 'trainer/ppo/ray_trainer.py':
        for expression, method, training in (
            ('self.ref_policy_wg.compute_ref_log_prob(batch)', 'self.ref_policy_wg.compute_ref_log_prob', False),
            ('self.actor_rollout_wg.compute_log_prob(batch)', 'self.actor_rollout_wg.compute_log_prob', False),
            ('self.actor_rollout_wg.update_actor(batch)', 'self.actor_rollout_wg.update_actor', True)):
            source = replace_once(source, expression,
                f'_spa_v1.distributed_call(self, {method}, batch, training={training})')
    elif name == 'workers/fsdp_workers.py':
        anchor = '            self.config.actor.ppo_mini_batch_size //= self.device_mesh.size() // self.ulysses_sequence_parallel_size'
        source = replace_once(source, anchor,
            '            if self.config.actor.policy_loss.loss_mode.startswith("spa_v1_"):\n'
            '                self.config.actor.ppo_mini_batch_size += (self.device_mesh.size() // self.ulysses_sequence_parallel_size) - 1\n' + anchor)
    elif name == 'workers/actor/dp_actor.py':
        source = replace_once(source, '            select_keys.append("action_loss_weights")',
            '            select_keys.append("action_loss_weights")\n'
            '            if "spa_row_valid" in data.batch:\n'
            '                select_keys.append("spa_row_valid")')
    else:
        source = replace_once(source, '        assert real_train_batch_size % minimal_bsz == 0, (',
            '        spa_padding = (config.algorithm.adv_estimator == "spa_v1" and n_gpus == 3\n'
            '                       and real_train_batch_size == 16\n'
            '                       and config.actor_rollout_ref.actor.strategy == "fsdp"\n'
            '                       and config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu == 1)\n'
            '        assert spa_padding or real_train_batch_size % minimal_bsz == 0, (')
    return source + '\n' + MARKER + '\n'


def main():
    import verl
    from spa_next.training.patch_verl_v1 import PATCHERS
    assert importlib.metadata.version('verl') == '0.7.0'
    root = Path(verl.__file__).parent
    plans = []
    for name, original_patcher in PATCHERS.items():
        path = root / name
        source = path.read_text()
        original = path.with_suffix('.py.spa-v1-original').read_text()
        expected = original_patcher(original)
        upgraded = transform(name, expected)
        if source not in (expected, upgraded):
            raise RuntimeError('Unexpected installed source: ' + name)
        compile(upgraded, str(path), 'exec')
        plans.append((path, source, upgraded))
    for path, old, new in plans:
        if old != new:
            backup = path.with_suffix('.py.spa-before-three-gpu')
            if backup.exists():
                raise RuntimeError('Backup already exists: ' + str(backup))
            backup.write_text(old)
            path.write_text(new)
        print(json.dumps(dict(path=str(path), sha256=hashlib.sha256(new.encode()).hexdigest())))
    path = root / 'utils/config.py'
    backup = path.with_suffix('.py.spa-before-three-gpu')
    current = path.read_text()
    original = backup.read_text() if backup.exists() else current
    updated = transform('utils/config.py', original)
    if current not in (original, updated):
        raise RuntimeError('Config source changed')
    compile(updated, str(path), 'exec')
    if not backup.exists():
        backup.write_text(original)
    path.write_text(updated)


if __name__ == '__main__':
    main()
