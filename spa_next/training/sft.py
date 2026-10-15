"""Independent Qwen SFT baseline; train-only demonstrations and assistant loss."""
import argparse
import collections
import hashlib
import json
import re
from pathlib import Path


def goal(text):
    match = re.search(r'Instruction:\s*\[SEP\]\s*(.*?)\s*\[SEP\]', text, re.S)
    if not match:
        raise ValueError('No task instruction in observation')
    return ' '.join(match.group(1).lower().split())


def goal_family(text):
    """Match WebShop task identity while ignoring its randomized price cap."""
    return re.sub(r'\d+(?:\.\d+)?(?= dollars)', '<price>', goal(text))


def encode_messages(tokenizer, messages):
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    labels = [-100] * len(ids)
    for i, message in enumerate(messages):
        if message['role'] != 'assistant':
            continue
        prefix = tokenizer.apply_chat_template(messages[:i], tokenize=True, add_generation_prompt=True)
        completed = tokenizer.apply_chat_template(messages[:i+1], tokenize=True, add_generation_prompt=False)
        assert ids[:len(prefix)] == prefix and ids[:len(completed)] == completed
        # Include assistant content and EOS, exclude the separator newline.
        end = len(completed)
        eos = tokenizer.convert_tokens_to_ids('<|im_end|>')
        while end > len(prefix) and ids[end-1] != eos:
            end -= 1
        assert end > len(prefix) and ids[end-1] == eos
        labels[len(prefix):end] = ids[len(prefix):end]
    assert any(x != -100 for x in labels)
    return dict(input_ids=ids, labels=labels)


def prepare(args):
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    dest = args.output
    dest.mkdir(parents=True, exist_ok=False)
    lookup = collections.defaultdict(list)
    for split in ('train', 'val', 'test'):
        for row in pq.read_table(args.split_dir / (split + '.parquet')).to_pylist():
            key_fn = goal_family if args.match_price_variants else goal
            lookup[key_fn(row['prompt'][-1]['content'])].append((split, row))
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    source = json.loads(args.source.read_text(encoding='utf-8'))
    counts = collections.Counter()
    accepted, seen, audit = [], set(), []
    for index, item in enumerate(source):
        reason, record = None, {'source_index': index, 'source_id': item.get('id')}
        conv = item['conversations']
        starts = [i for i, m in enumerate(conv) if m['from'] == 'human' and 'WebShop' in m['value'] and 'Instruction:' in m['value']]
        if not starts:
            reason = 'no_initial_observation'
        else:
            start = starts[0]
            key_fn = goal_family if args.match_price_variants else goal
            matches = lookup.get(key_fn(conv[start]['value']), [])
            record['matched_tasks'] = [(s, r['extra_info']['task_id']) for s, r in matches]
            if not matches:
                reason = 'unmatched_goal'
            elif any(s != 'train' for s, _ in matches):
                reason = 'heldout_goal'
            elif float(item.get('true_reward', item.get('reward', 0))) < 1 - 1e-6:
                reason = 'not_fully_successful'
            else:
                # Keep the source task and observations intact. The frozen split
                # row is used only to assign the underlying task family.
                messages = [dict(role='system', content=matches[0][1]['prompt'][0]['content']),
                            dict(role='user', content=conv[start]['value'].strip())]
                for turn, m in enumerate(conv[start+1:]):
                    role = 'assistant' if turn % 2 == 0 else 'user'
                    if m['from'] != ('gpt' if role == 'assistant' else 'human'):
                        reason = 'invalid_roles'
                        break
                    content = m['value'].strip()
                    if role == 'user':
                        content = 'Observation:\n' + re.sub(r'^Observation:\s*', '', content)
                    elif len(re.findall(r'Action:\s*(?:search|click)\[[^\n]*\]', content)) != 1:
                        reason = 'invalid_action_format'
                        break
                    messages.append(dict(role=role, content=content))
                if not reason and (messages[-1]['role'] != 'assistant' or
                                   not re.search(r'Action:\s*click\[buy now\]', messages[-1]['content'], re.I)):
                    reason = 'no_final_purchase'
                if not reason:
                    key = hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()
                    if key in seen:
                        reason = 'duplicate_trajectory'
                    else:
                        encoded = encode_messages(tok, messages)
                        record['tokens'] = len(encoded['input_ids'])
                        if len(encoded['input_ids']) > args.max_length:
                            reason = 'over_context_limit'
                        else:
                            seen.add(key)
                            accepted.append(encoded)
                            record['trajectory_sha256'] = key
                            reason = 'accepted'
        counts[reason] += 1
        record['decision'] = reason
        audit.append(record)
    assert accepted, 'No eligible SFT examples'
    with (dest / 'train.jsonl').open('w') as f:
        for row in accepted:
            f.write(json.dumps(row) + '\n')
    report = dict(source=str(args.source), source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                  split_dir=str(args.split_dir), model=str(args.model), max_length=args.max_length,
                  match_price_variants=args.match_price_variants,
                  counts=dict(counts), examples=len(accepted),
                  total_tokens=sum(len(x['input_ids']) for x in accepted),
                  supervised_tokens=sum(sum(t != -100 for t in x['labels']) for x in accepted),
                  max_tokens=max(len(x['input_ids']) for x in accepted),
                  split_sha256={s: hashlib.sha256((args.split_dir / (s+'.parquet')).read_bytes()).hexdigest()
                                for s in ('train', 'val', 'test')})
    (dest / 'data_report.json').write_text(json.dumps(report, indent=2))
    (dest / 'source_audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(report, indent=2))


def train(args):
    import os
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed
    set_seed(42)
    rank = int(os.environ.get('LOCAL_RANK', '0'))
    torch.cuda.set_device(rank)
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
              torch_dtype=torch.bfloat16, attn_implementation='sdpa', device_map={'': rank})
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=32, lora_alpha=64, lora_dropout=0.,
                           target_modules='all-linear', bias='none', task_type='CAUSAL_LM'))
    rows = [json.loads(l) for l in (args.output / 'train.jsonl').read_text().splitlines()]
    # Run longest trajectories first for a bounded memory/load probe.
    if args.probe:
        rows = sorted(rows, key=lambda r: len(r['input_ids']), reverse=True)[:16]
    class Dataset(torch.utils.data.Dataset):
        def __len__(self): return len(rows)
        def __getitem__(self, i): return rows[i]
    def collate(batch):
        assert len(batch) == 1
        ids = torch.tensor([batch[0]['input_ids']], dtype=torch.long)
        return dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                    labels=torch.tensor([batch[0]['labels']], dtype=torch.long))
    class AssistantTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop('labels')
            # Single example per GPU. Produce logits only at supervised next-token
            # positions, avoiding a full 4096 x 151936 float32 logits allocation.
            positions = torch.where(labels[0, 1:] != -100)[0]
            outputs = model(**inputs, logits_to_keep=positions)
            targets = labels[0, positions+1]
            loss = sum(F.cross_entropy(outputs.logits[0, i:i+128].float(), targets[i:i+128], reduction='sum')
                       for i in range(0, len(targets), 128)) / len(targets)
            return (loss, outputs) if return_outputs else loss
    out = args.output / ('probe' if args.probe else 'trainer')
    training_args = TrainingArguments(output_dir=str(out), num_train_epochs=1,
        max_steps=2 if args.probe else -1, learning_rate=1e-4, lr_scheduler_type='cosine', warmup_ratio=.03,
        per_device_train_batch_size=1, gradient_accumulation_steps=4, bf16=True, tf32=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        ddp_find_unused_parameters=False, logging_steps=1, save_strategy='no' if args.probe else 'steps',
        save_steps=20, save_total_limit=2, report_to='none', remove_unused_columns=False,
        dataloader_num_workers=0, seed=42, data_seed=42, optim='adamw_torch', max_grad_norm=1.)
    trainer = AssistantTrainer(model=model, args=training_args, train_dataset=Dataset(), data_collator=collate)
    trainer.model_accepts_loss_kwargs = False
    result = trainer.train()
    trainer.save_metrics('train', result.metrics)
    trainer.save_state()
    if not args.probe:
        trainer.save_model(str(args.output / 'adapter'))
        if trainer.is_world_process_zero(): tok.save_pretrained(args.output / 'adapter')
    if trainer.is_world_process_zero():
        print('SFT_FINISHED', json.dumps(result.metrics), flush=True)
    if torch.distributed.is_initialized(): torch.distributed.destroy_process_group()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['prepare', 'train'])
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source', type=Path)
    p.add_argument('--split-dir', type=Path)
    p.add_argument('--max-length', type=int, default=4096)
    p.add_argument('--match-price-variants', action='store_true')
    p.add_argument('--probe', action='store_true')
    args = p.parse_args()
    (prepare if args.mode == 'prepare' else train)(args)


if __name__ == '__main__': main()
