"""Checkpoint token-row log probabilities in the pinned verl actor update."""
import hashlib
import json
from pathlib import Path

MARKER = '# SPA_MEMORY_SAFE_LOGPROB_BACKWARD_V1'


def main():
    import verl

    path = Path(verl.__file__).parent / 'utils/torch_functional.py'
    source = path.read_text()
    if MARKER in source:
        print(json.dumps(dict(status='already_patched', path=str(path))))
        return
    start = source.index('def logprobs_from_logits_v2(')
    end = source.index('\n\ndef clip_by_value(', start)
    before = source[start:end]
    if ('chunk = 128' not in before or
            'probs = F.log_softmax(rows, dim=-1)' not in before or
            'pieces.append(torch.gather(probs, dim=-1, index=target.unsqueeze(-1)).squeeze(-1))' not in before):
        raise RuntimeError('Pinned memory-safe logprob anchor changed; refusing to patch')
    after = before.replace('chunk = 128', 'chunk = 32')
    after = after.replace(
        '            probs = F.log_softmax(rows, dim=-1)\n'
        '            pieces.append(torch.gather(probs, dim=-1, index=target.unsqueeze(-1)).squeeze(-1))',
        '            def selected_logprob(input_rows, input_target):\n'
        '                probs = F.log_softmax(input_rows, dim=-1)\n'
        '                return torch.gather(probs, dim=-1, index=input_target.unsqueeze(-1)).squeeze(-1)\n'
        '            if rows.requires_grad:\n'
        '                from torch.utils.checkpoint import checkpoint\n'
        '                pieces.append(checkpoint(selected_logprob, rows, target, use_reentrant=False))\n'
        '            else:\n'
        '                pieces.append(selected_logprob(rows, target))')
    patched = source[:start] + after + source[end:] + '\n' + MARKER + '\n'
    compile(patched, str(path), 'exec')
    backup = path.with_suffix('.py.spa-before-memory-safe-backward')
    if backup.exists():
        raise RuntimeError('Refusing to overwrite existing backup: ' + str(backup))
    backup.write_text(source)
    path.write_text(patched)
    print(json.dumps(dict(status='patched', path=str(path), backup=str(backup),
                          original_sha256=hashlib.sha256(source.encode()).hexdigest(),
                          patched_sha256=hashlib.sha256(patched.encode()).hexdigest())))


if __name__ == '__main__':
    main()
