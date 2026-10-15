"""Patch pinned verl entropy calculation to bound the vocab softmax workspace."""
import hashlib
import json
from pathlib import Path

MARKER = '# SPA_MEMORY_SAFE_ENTROPY_V1'


def main():
    import verl
    path = Path(verl.__file__).parent / 'utils/torch_functional.py'
    source = path.read_text()
    if MARKER in source:
        print(json.dumps(dict(status='already_patched', path=str(path))))
        return
    start = source.index('def entropy_from_logits(')
    end = source.index('\n\ndef entropy_from_logits_with_chunking(', start)
    lines = [
        'def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:',
        '    """Calculate entropy with a bounded token-row softmax workspace."""',
        '    if logits.ndim < 1:',
        '        raise ValueError("Expected logits with a vocabulary dimension")',
        '    vocab = logits.shape[-1]',
        '    flat = logits.reshape(-1, vocab)',
        '    pieces = []',
        '    for start in range(0, flat.shape[0], 128):',
        '        rows = flat[start:start + 128]',
        '        probabilities = torch.nn.functional.softmax(rows, dim=-1)',
        '        pieces.append(torch.logsumexp(rows, dim=-1) - torch.sum(probabilities * rows, dim=-1))',
        '    return torch.cat(pieces).reshape(logits.shape[:-1])',
        '',
    ]
    replacement = '\n'.join(lines)
    patched = source[:start] + replacement + source[end:] + '\n' + MARKER + '\n'
    compile(patched, str(path), 'exec')
    backup = path.with_suffix('.py.spa-before-memory-safe-entropy')
    if backup.exists():
        raise RuntimeError('Refusing to overwrite existing backup: ' + str(backup))
    backup.write_text(source)
    path.write_text(patched)
    print(json.dumps(dict(status='patched', path=str(path), backup=str(backup),
                          original_sha256=hashlib.sha256(source.encode()).hexdigest(),
                          patched_sha256=hashlib.sha256(patched.encode()).hexdigest())))


if __name__ == '__main__':
    main()
