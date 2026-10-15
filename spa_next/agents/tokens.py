"""Dependency-free token/mask invariants for the Qwen ChatML rollout."""


class TokenTrace:
    def __init__(self, prompt_ids, response_budget):
        self.prompt_ids = list(prompt_ids)
        self.response_ids, self.mask, self.logprobs, self.action_ids = [], [], [], []
        self.budget = response_budget

    @property
    def remaining(self):
        return self.budget - len(self.response_ids)

    @property
    def context(self):
        return self.prompt_ids + self.response_ids

    def append(self, ids, *, action_id=-1, logprobs=None, train=False):
        ids = list(ids)
        if len(ids) > self.remaining:
            raise ValueError("Token budget exceeded; never silently slice an action")
        if train and (not ids or action_id < 0 or logprobs is None or len(logprobs) != len(ids)):
            raise ValueError("Trainable actions require matching token IDs and behavior logprobs")
        start = len(self.response_ids)
        self.response_ids.extend(ids)
        self.mask.extend([int(train)] * len(ids))
        self.action_ids.extend([action_id if train else -1] * len(ids))
        self.logprobs.extend(logprobs if train else [0.] * len(ids))
        return [start, len(self.response_ids)]

    def observation_suffix(self, tokenizer, observation):
        # Prior assistant EOS is already among the original generated IDs. Only
        # encode NEW environment text and template delimiters, never old output.
        safe = observation.replace("<|im_start|>", "[im_start]").replace("<|im_end|>", "[im_end]")
        return tokenizer.encode("\n<|im_start|>user\nObservation:\n" + safe
                                + "<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)
