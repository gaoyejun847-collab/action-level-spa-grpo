"""Deterministic WebShop evaluation of a PEFT SFT adapter."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from urllib.request import Request, urlopen


def request(url, op, body):
    req = Request(url + '/' + op, data=json.dumps(body).encode(),
                  headers={'Content-Type': 'application/json'})
    with urlopen(req, timeout=120) as response:
        return json.load(response)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--adapter', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--limit', type=int)
    p.add_argument('--url', default='http://127.0.0.1:8765')
    args = p.parse_args()
    import pyarrow.parquet as pq
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from spa_next.agents.tokens import TokenTrace

    rank = int(os.environ['LOCAL_RANK'])
    world = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(rank)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation='sdpa', device_map={'': rank})
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model.eval()
    model.config.use_cache = True
    rows = pq.read_table(args.data).to_pylist()
    if args.limit is not None:
        rows = rows[:args.limit]
    rows = rows[rank::world]
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / f'rank-{rank}.jsonl'
    done_ids = set()
    if output.exists():
        done_ids = {json.loads(x)['task_id'] for x in output.read_text().splitlines() if x.strip()}
    with output.open('a', buffering=1) as stream, torch.inference_mode():
        for row in rows:
            info = row['extra_info']
            task_id = int(info['task_id'])
            if task_id in done_ids:
                continue
            episode = request(args.url, 'reset', {'task_id': task_id, 'env_seed': int(info['env_seed'])})
            started = time.monotonic()
            try:
                if episode['initial_state_id'] != info['initial_state_id']:
                    raise RuntimeError('Environment reset differs from frozen validation data')
                prompt_ids = tokenizer.apply_chat_template(row['prompt'], tokenize=True,
                                                           add_generation_prompt=True)
                if len(prompt_ids) > 1024:
                    raise ValueError('Initial prompt exceeds 1024 tokens')
                trace = TokenTrace(prompt_ids, 3072)
                turns, outcome, success, reason = [], 0., False, 'max_steps'
                for step in range(10):
                    max_new = min(256, trace.remaining)
                    if max_new <= 0:
                        reason = 'context_budget'
                        break
                    context = trace.context
                    tensor = torch.tensor([context], dtype=torch.long, device=rank)
                    generated = model.generate(
                        input_ids=tensor, attention_mask=torch.ones_like(tensor),
                        max_new_tokens=max_new, do_sample=False,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.eos_token_id,
                        use_cache=True)[0, len(context):].tolist()
                    complete = bool(generated and generated[-1] == tokenizer.eos_token_id)
                    text = tokenizer.decode(generated, skip_special_tokens=True)
                    turn = {'step_id': step, 'generated_token_ids': generated,
                            'text': text, 'complete': complete}
                    if not complete:
                        trace.append(generated)
                        turns.append(turn)
                        reason = 'incomplete_action'
                        break
                    trace.append(generated, action_id=step,
                                 logprobs=[0.] * len(generated), train=True)
                    result = request(args.url, 'step', {'episode_id': episode['episode_id'],
                                                       'step_id': step, 'text': text})
                    turn['environment'] = result
                    turns.append(turn)
                    outcome, success = result['outcome_reward'], result['success']
                    if result['terminated'] or result['truncated']:
                        reason = 'terminated' if result['terminated'] else 'max_steps'
                        break
                    suffix = trace.observation_suffix(tokenizer, result['observation'])
                    if len(suffix) >= trace.remaining:
                        reason, outcome = 'context_budget', 0.
                        break
                    trace.append(suffix)
                record = {'task_id': task_id, 'initial_state_id': episode['initial_state_id'],
                          'outcome_reward': outcome, 'success': success, 'steps': len(turns),
                          'budget_exhausted': reason != 'terminated', 'termination_reason': reason,
                          'elapsed_seconds': time.monotonic() - started, 'turns': turns,
                          'model': str(args.model), 'adapter': str(args.adapter),
                          'template_sha256': hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()}
                stream.write(json.dumps(record) + '\n')
            finally:
                request(args.url, 'close', {'episode_id': episode['episode_id']})
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
