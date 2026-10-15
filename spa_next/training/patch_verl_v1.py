"""Guarded narrow V1 integration for the pinned verl 0.7.0 installation.

All modifications are conditional on the V1 estimator/loss names. Pure
``patch_sources`` is usable by tests without changing the vendored wheel.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

MARKER = "# SPA_ACTION_GRPO_V1_INTEGRATION"


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError(f"Unexpected verl 0.7 source anchor: {old[:100]!r}")
    return source.replace(old, new, 1)


def patch_trainer(source):
    source = replace_once(source,
        '    # Back-compatible with trainers that do not compute response mask in fit',
        '    if adv_estimator == "spa_v1":\n'
        '        return _spa_v1.compute_advantage(data, config)\n'
        '    # Back-compatible with trainers that do not compute response mask in fit')
    source = replace_once(source,
        '        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):\n'
        '            val_metrics = self._validate()',
        '        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):\n'
        '            _spa_v1.before_validation(self)\n'
        '            val_metrics = self._validate()\n'
        '            _spa_v1.record_validation(self, val_metrics)')
    source = replace_once(source,
        '                    with marked_timer("testing", timing_raw, color="green"):\n'
        '                        val_metrics: dict = self._validate()',
        '                    _spa_v1.before_validation(self)\n'
        '                    with marked_timer("testing", timing_raw, color="green"):\n'
        '                        val_metrics: dict = self._validate()\n'
        '                        _spa_v1.record_validation(self, val_metrics)')
    source = replace_once(source,
        '                if self.config.trainer.save_freq > 0 and (',
        '                if (getattr(self, "_spa_v1_saved_step", -1) != self.global_steps\n'
        '                    and self.config.trainer.save_freq > 0) and (')
    source = replace_once(source,
        '                    # update critic\n',
        '                    metrics.update(batch.meta_info.get("spa_v1_metrics", {}))\n\n'
        '                    # update critic\n')
    return source + '\n' + MARKER + '\nfrom spa_next.training import verl_v1 as _spa_v1\n'


def patch_actor(source):
    # Insert only in update_policy, whose unique first field block contains advantages.
    source = replace_once(source,
        '            "advantages",\n        ]\n',
        '            "advantages",\n        ]\n'
        '        spa_v1 = _spa_v1.actor_mode(self.config)\n'
        '        if spa_v1:\n'
        '            select_keys.append("action_loss_weights")\n')
    source = replace_once(source,
        '                self.actor_optimizer.zero_grad()\n\n                for micro_batch in micro_batches:',
        '                spa_normalizer = (_spa_v1.minibatch_normalizer(mini_batch, self.config, get_device_id())\n'
        '                                  if spa_v1 else None)\n'
        '                self.actor_optimizer.zero_grad()\n\n                for micro_batch in micro_batches:')
    source = replace_once(source,
        '                    # all return: (bsz, response_length)\n',
        '                    if spa_v1:\n'
        '                        loss_scale_factor = 1.0  # normalizer includes complete global minibatch\n\n'
        '                    # all return: (bsz, response_length)\n')
    source = replace_once(source,
        '                        if on_policy:\n',
        '                        if on_policy and not spa_v1:\n')
    source = replace_once(source,
        '                    policy_loss_fn = get_policy_loss_fn(loss_mode)\n',
        '                    policy_loss_fn = get_policy_loss_fn(loss_mode) if not spa_v1 else None\n')
    start = '                    pg_loss, pg_metrics = policy_loss_fn(\n'
    end = '                    micro_batch_metrics.update(pg_metrics)'
    begin = source.index(start)
    finish = source.index(end, begin)
    old = source[begin:finish]
    nested = ''.join('    ' + line if line.strip() else line for line in old.splitlines(keepends=True))
    source = replace_once(source, old,
        '                    if spa_v1:\n'
        '                        pg_loss, pg_metrics = _spa_v1.policy_loss(\n'
        '                            model_inputs, old_log_prob, log_prob, self.config, spa_normalizer)\n'
        '                    else:\n' + nested)
    source = replace_once(source,
        '                    micro_batch_metrics.update(pg_metrics)\n',
        '                    if spa_v1:\n'
        '                        for key, value in pg_metrics.items():\n'
        '                            metrics[key] = metrics.get(key, 0.0) + value\n'
        '                    else:\n'
        '                        micro_batch_metrics.update(pg_metrics)\n')
    source = replace_once(source,
        '                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)',
        '                        if spa_v1:\n'
        '                            kl_loss = _spa_v1.kl_loss(model_inputs, log_prob, self.config, spa_normalizer)\n'
        '                        else:\n'
        '                            kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)')
    return source + '\n' + MARKER + '\nfrom spa_next.training import verl_v1 as _spa_v1\n'


def patch_worker(source):
    source = replace_once(source,
        '        from verl.workers.actor import DataParallelPPOActor\n',
        '        from verl.workers.actor import DataParallelPPOActor\n'
        '        if self.config.actor.policy_loss.loss_mode.startswith("spa_v1_"):\n'
        '            from transformers import set_seed\n'
        '            set_seed(int(self.config.actor.data_loader_seed))\n')
    return source + '\n' + MARKER + '\n'


PATCHERS = {"trainer/ppo/ray_trainer.py": patch_trainer,
            "workers/actor/dp_actor.py": patch_actor,
            "workers/fsdp_workers.py": patch_worker}


def patch_sources(root):
    results = {}
    for relative, patcher in PATCHERS.items():
        path = Path(root) / relative
        original = path.read_text(encoding="utf-8")
        if MARKER in original:
            backup = path.with_suffix(".py.spa-v1-original")
            if not backup.exists() or patcher(backup.read_text(encoding="utf-8")) != original:
                raise RuntimeError(f"V1 patched source changed since installation: {path}")
            results[path] = (original, original)
            continue
        patched = patcher(original)
        compile(patched, str(path), "exec")
        results[path] = (original, patched)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    root = args.source_root
    if root is None:
        import verl
        if importlib.metadata.version("verl") != "0.7.0":
            raise RuntimeError("V1 requires verl==0.7.0")
        root = Path(verl.__file__).parent
    sources = patch_sources(root)
    if not args.check_only:
        for path, (old, new) in sources.items():
            if old != new and path.with_suffix(".py.spa-v1-original").exists():
                raise RuntimeError(f"Refusing to overwrite backup: {path}")
    report = {}
    for path, (old, new) in sources.items():
        backup = path.with_suffix(".py.spa-v1-original")
        if not args.check_only and old != new:
            if backup.exists():
                raise RuntimeError(f"Refusing to overwrite backup: {backup}")
            backup.write_text(old, encoding="utf-8")
            path.write_text(new, encoding="utf-8")
        report[str(path)] = dict(original_sha256=hashlib.sha256(old.encode()).hexdigest(),
                                patched_sha256=hashlib.sha256(new.encode()).hexdigest(),
                                check_only=args.check_only)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
