"""Apply a small, backed-up patch to the dedicated verl 0.7 installation."""
import hashlib
import importlib.metadata
import json
from pathlib import Path


def main():
    import verl
    assert importlib.metadata.version('verl') == '0.7.0'
    root = Path(verl.__file__).parent
    manager = root / 'utils/checkpoint/fsdp_checkpoint_manager.py'
    worker = root / 'workers/fsdp_workers.py'
    changes = {
        manager: [
            ('                    torch.save(model_state_dict, model_path)',
             '                    del model_state_dict\n                    _spa_release_cpu_cache("model_saved")'),
            ('                    torch.save(optimizer_state_dict, optim_path)',
             '                    del optimizer_state_dict\n                    _spa_release_cpu_cache("optimizer_saved")'),
            ('                self.model.load_state_dict(model_state_dict)',
             '                del model_state_dict\n                _spa_release_cpu_cache("model_loaded")'),
            ('                self.optimizer.load_state_dict(optimizer_state_dict)',
             '                del optimizer_state_dict\n                _spa_release_cpu_cache("optimizer_loaded")'),
        ],
        worker: [
            ('                        save_file(lora_params, os.path.join(lora_save_path, "adapter_model.safetensors"))',
             '                        del lora_params\n                        _spa_release_cpu_cache("adapter_saved")'),
        ],
    }
    report = {}
    for path, replacements in changes.items():
        text = path.read_text()
        if '# SPA_CPU_CHECKPOINT_MEMORY_V1' in text:
            report[str(path)] = 'already patched'
            continue
        original = text
        for anchor, addition in replacements:
            if text.count(anchor) != 1:
                raise RuntimeError(f'Unexpected verl source at {path}: {anchor}')
            text = text.replace(anchor, anchor + '\n' + addition)
        # Append import: it executes at module import before methods are called.
        text += '\n# SPA_CPU_CHECKPOINT_MEMORY_V1\nfrom spa_next.training.cpu_memory import release_cpu_cache as _spa_release_cpu_cache\n'
        compile(text, str(path), 'exec')
        backup = path.with_suffix('.py.spa-original')
        if backup.exists():
            raise RuntimeError(f'Unrecognized prior backup: {backup}')
        backup.write_text(original)
        path.write_text(text)
        report[str(path)] = dict(original_sha256=hashlib.sha256(original.encode()).hexdigest(),
                                patched_sha256=hashlib.sha256(text.encode()).hexdigest(), backup=str(backup))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
