"""Verify the incremental ChatML serialization against the selected real tokenizer."""
import argparse

from spa_next.agents.tokens import TokenTrace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    messages = [dict(role="system", content="You are web shopping."),
                dict(role="user", content="Find shoes"),
                dict(role="assistant", content="Action: search[shoes]"),
                dict(role="user", content="Observation:\nresult")]
    prefix = tokenizer.apply_chat_template(messages[:2], add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    ids = tokenizer.encode("Action: search[shoes]<|im_end|>", add_special_tokens=False)
    assert ids[-1] == tokenizer.eos_token_id
    trace = TokenTrace(prefix, 1000)
    trace.append(ids, action_id=0, train=True, logprobs=[-1.] * len(ids))
    trace.append(trace.observation_suffix(tokenizer, "result"))
    assert trace.context == full, "Incremental tokens differ from the model chat template"
    print("Incremental ChatML matches the model tokenizer exactly.")


if __name__ == "__main__":
    main()
