"""Exercise TP2 inference and the rank-32 LoRA-capable engine before RL."""
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    args = parser.parse_args()
    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, tensor_parallel_size=2, dtype="bfloat16",
              enable_lora=True, max_lora_rank=32, max_model_len=1024,
              max_num_seqs=4, gpu_memory_utilization=0.5, enforce_eager=True,
              disable_custom_all_reduce=True)
    tokenizer = llm.get_tokenizer()
    prompt = tokenizer.apply_chat_template([
        dict(role="system", content="Reply with exactly one shopping action."),
        dict(role="user", content="Search for red shoes. Use Action: search[red shoes]."),
    ], tokenize=False, add_generation_prompt=True)
    outputs = llm.generate([prompt], SamplingParams(temperature=0, max_tokens=64, logprobs=0))
    result = outputs[0].outputs[0]
    assert result.token_ids and result.logprobs
    assert result.token_ids[-1] == tokenizer.eos_token_id
    print({"text": result.text, "tokens": len(result.token_ids), "finish_reason": result.finish_reason})


if __name__ == "__main__":
    main()
