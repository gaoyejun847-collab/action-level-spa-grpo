"""Avoid a discarded frozen vocabulary gradient in pinned verl's fused head."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
from spa_next.training.patch_verl_v1 import replace_once

MARKER = '# SPA_FUSED_FROZEN_HEAD_V1'


def transform(source):
    source = replace_once(source,
        '    dvocab_weights = dlogits.t() @ hidden_states',
        '    dvocab_weights = (dlogits.t() @ hidden_states) if vocab_weights.requires_grad else None')
    # Bound each vocabulary-sized workspace; output and loss formulas unchanged.
    if source.count('chunk_size: int = 512') != 2:
        raise RuntimeError('Unexpected fused head chunk-size anchors')
    return source.replace('chunk_size: int = 512', 'chunk_size: int = 32') + '\n' + MARKER + '\n'


def main():
    import verl
    if importlib.metadata.version('verl') != '0.7.0':
        raise RuntimeError('Requires verl 0.7.0')
    path = Path(verl.__file__).parent / 'utils/experimental/torch_functional.py'
    backup = path.with_suffix('.py.spa-before-frozen-head')
    current = path.read_text()
    original = backup.read_text() if backup.exists() else current
    updated = transform(original)
    if current not in (original, updated):
        raise RuntimeError('Installed fused head source changed')
    compile(updated, str(path), 'exec')
    if current != updated:
        if not backup.exists():
            backup.write_text(original)
        path.write_text(updated)
    print(json.dumps(dict(path=str(path), backup=str(backup),
        original_sha256=hashlib.sha256(original.encode()).hexdigest(),
        patched_sha256=hashlib.sha256(updated.encode()).hexdigest())))


if __name__ == '__main__':
    main()
