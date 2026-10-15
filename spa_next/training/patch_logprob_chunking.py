"""Patch pinned verl logprob to bound the vocabulary softmax workspace."""
import hashlib
import json
from pathlib import Path

MARKER = '# SPA_MEMORY_SAFE_LOGPROB_V1'


def main():
    import verl
    path = Path(verl.__file__).parent / 'utils/torch_functional.py'
    source = path.read_text()
    if MARKER in source:
        print(json.dumps(dict(status='already_patched', path=str(path))))
        return
    start = source.index('def logprobs_from_logits_v2(')
    end = source.index('\n\ndef clip_by_value(', start)
    replacement = '''def logprobs_from_logits_v2(logits: torch.FloatTensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute target probabilities in bounded chunks of token rows."""
    if logits.shape[:-1] != labels.shape:
        raise ValueError("Logits leading dimensions must match labels")
    vocab = logits.shape[-1]
    flat_logits = logits.reshape(-1, vocab)
    flat_labels = labels.reshape(-1)
    chunk = 128
    pieces = []
    for start in range(0, flat_logits.shape[0], chunk):
        rows = flat_logits[start:start + chunk]
        target = flat_labels[start:start + chunk]
        if rows.dtype in (torch.float32, torch.float64):
            selected = torch.gather(rows, dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
            pieces.append(selected - torch.logsumexp(rows, dim=-1))
        else:
            probs = F.log_softmax(rows, dim=-1)
            pieces.append(torch.gather(probs, dim=-1, index=target.unsqueeze(-1)).squeeze(-1))
    return torch.cat(pieces).reshape(labels.shape)
'''
    patched = source[:start] + replacement + source[end:] + '\n' + MARKER + '\n'
    compile(patched, str(path), 'exec')
    backup = path.with_suffix('.py.spa-before-memory-safe')
    if backup.exists():
        raise RuntimeError('Refusing to overwrite existing backup: ' + str(backup))
    backup.write_text(source)
    path.write_text(patched)
    print(json.dumps(dict(status='patched', path=str(path), backup=str(backup),
                          original_sha256=hashlib.sha256(source.encode()).hexdigest(),
                          patched_sha256=hashlib.sha256(patched.encode()).hexdigest())))


if __name__ == '__main__':
    main()
