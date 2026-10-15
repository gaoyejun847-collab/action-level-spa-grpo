"""Merge the reviewed SFT adapter once and verify its immutable provenance."""
import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_snapshot(path):
    path = Path(path)
    manifest = json.loads((path / 'spa_snapshot.json').read_text())
    if manifest.get('kind') != 'merged_sft_base' or not manifest.get('verified'):
        raise ValueError('A verified merged SFT snapshot is required')
    for name, expected in manifest['files'].items():
        if sha256(path / name) != expected:
            raise ValueError('SFT snapshot changed: ' + name)
    if any(path.glob('adapter*')):
        raise ValueError('The SFT snapshot must contain merged base weights, not an adapter')
    return manifest


def merge(args):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    base, adapter, dest = args.base.resolve(), args.adapter.resolve(), args.output.resolve()
    if dest.exists():
        manifest = verify_snapshot(dest)
        if manifest['source_base'] != str(base) or manifest['source_adapter'] != str(adapter):
            raise ValueError('Existing snapshot has different source paths')
        if manifest['adapter_sha256'] != sha256(adapter / 'adapter_model.safetensors'):
            raise ValueError('SFT adapter changed since the snapshot was created')
        print(json.dumps(manifest, indent=2), flush=True)
        return
    staging = dest.with_name(dest.name + '.building')
    staging.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(base, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation='sdpa', device_map={'': args.device})
    model = PeftModel.from_pretrained(model, adapter, is_trainable=False).eval()
    prompt = tokenizer.apply_chat_template([
        {'role': 'system', 'content': 'You are an assistant interacting with WebShop.'},
        {'role': 'user', 'content': 'Find a blue cotton shirt under 30 dollars.'}],
        tokenize=True, add_generation_prompt=True)
    ids = torch.tensor([prompt], device=args.device)
    with torch.inference_mode():
        before = model(input_ids=ids, logits_to_keep=1).logits.float().cpu()
    merged = model.merge_and_unload(safe_merge=True).eval()
    with torch.inference_mode():
        after = merged(input_ids=ids, logits_to_keep=1).logits.float().cpu()
    # BF16 merge rounds LoRA additions; record this rather than claiming bit equality.
    difference = (before - after).abs()
    if not torch.isfinite(after).all() or difference.mean() > 0.10:
        raise ValueError('Merged SFT verification failed: nonfinite or excessive logit drift')
    merged.save_pretrained(staging, safe_serialization=True, max_shard_size='2GB')
    tokenizer.save_pretrained(staging)
    fresh = get_peft_model(merged, LoraConfig(r=32, lora_alpha=64, lora_dropout=0.,
        target_modules='all-linear', bias='none', task_type='CAUSAL_LM')).eval()
    with torch.inference_mode():
        actor = fresh(input_ids=ids, logits_to_keep=1).logits.float().cpu()
        with fresh.disable_adapter():
            reference = fresh(input_ids=ids, logits_to_keep=1).logits.float().cpu()
    if not torch.equal(actor, reference) or not torch.equal(after, reference):
        raise ValueError('Initial actor and reference do not both match the merged SFT base')
    files = {str(p.relative_to(staging)): sha256(p) for p in sorted(staging.rglob('*')) if p.is_file()}
    version = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    manifest = dict(kind='merged_sft_base', version=version, verified=True,
        created_utc=datetime.now(timezone.utc).isoformat(), source_base=str(base),
        source_adapter=str(adapter), adapter_sha256=sha256(adapter / 'adapter_model.safetensors'),
        files=files, model_dtype='bfloat16', lora_rank=32, lora_alpha=64,
        verification=dict(logits_mean_absolute_difference=difference.mean().item(),
                          logits_max_absolute_difference=difference.max().item(),
                          fresh_actor_equals_reference=True, reference_equals_sft=True))
    (staging / 'spa_snapshot.json').write_text(json.dumps(manifest, indent=2))
    staging.rename(dest)
    # Checkpoints and all new adapters live elsewhere; never train into this path.
    if os.name == 'posix':
        for p in dest.rglob('*'):
            p.chmod(0o555 if p.is_dir() else 0o444)
        dest.chmod(0o555)
    print(json.dumps(manifest, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    merge(parser.parse_args())


if __name__ == '__main__':
    main()
