"""Skip verl's auxiliary rollout-correction diagnostic for SPA padded rows."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
from spa_next.training.patch_verl_v1 import replace_once

MARKER = '# SPA_SKIP_ROLLOUT_CORR_PADDING_V1'


def main():
    import verl
    if importlib.metadata.version('verl') != '0.7.0':
        raise RuntimeError('Requires verl 0.7.0')
    path = Path(verl.__file__).parent / 'workers/actor/dp_actor.py'
    current = path.read_text()
    if MARKER in current:
        print(json.dumps(dict(status='already_patched', path=str(path))))
        return
    old = '                    if loss_mode != "bypass_mode" and rollout_log_prob is not None:\n'
    new = ('                    # SPA uses two driver-side padding rows on three GPUs.\n'
           '                    # verl\'s auxiliary off-policy diagnostic asserts a nonempty\n'
           '                    # response mask for every microbatch, which padding violates.\n'
           '                    if loss_mode != "bypass_mode" and rollout_log_prob is not None and not spa_v1:\n')
    updated = replace_once(current, old, new) + '\n' + MARKER + '\n'
    compile(updated, str(path), 'exec')
    backup = path.with_suffix('.py.spa-before-rollout-corr-padding')
    if backup.exists():
        raise RuntimeError('Backup already exists: ' + str(backup))
    backup.write_text(current)
    path.write_text(updated)
    print(json.dumps(dict(status='patched', path=str(path), backup=str(backup),
                          original_sha256=hashlib.sha256(current.encode()).hexdigest(),
                          patched_sha256=hashlib.sha256(updated.encode()).hexdigest())))


if __name__ == '__main__':
    main()
