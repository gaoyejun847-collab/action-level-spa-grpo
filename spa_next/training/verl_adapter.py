"""Integration against the pinned verl 0.7.0 wheel, with no PRM or critic."""
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from uuid import uuid4

import httpx
import ray
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase, AgentLoopOutput, AgentLoopWorkerBase, AgentLoopManager,
)

from spa_next.agents.tokens import TokenTrace


@ray.remote
class SPAWorker(AgentLoopWorkerBase):
    async def _run_agent_loop(self, sampling_params, trajectory, **kwargs):
        kwargs["spa_version"] = dict(trajectory)
        return await super()._run_agent_loop(sampling_params, trajectory, **kwargs)


class SPAManager(AgentLoopManager):
    def __init__(self, *args, **kwargs):
        self.agent_loop_workers_class = SPAWorker
        super().__init__(*args, **kwargs)


class SPAAgentLoop(AgentLoopBase):
    async def run(self, sampling_params, **kwargs):
        cfg = self.config
        info = kwargs["extra_info"]
        version = kwargs["spa_version"]
        if version["step"] < 0:
            raise RuntimeError("Missing trainer step; cannot attribute rollout to a policy version")
        url = os.environ.get("SPA_ENV_URL", "http://127.0.0.1:8765")
        episode = None
        generated_seconds, env_seconds = 0., 0.
        async with httpx.AsyncClient(base_url=url, timeout=120., trust_env=False) as client:
            async def request(op, data):
                nonlocal env_seconds
                start = time.monotonic()
                # No automatic retry: a lost response may follow an executed action.
                response = await client.post("/" + op, json=data)
                response.raise_for_status()
                env_seconds += time.monotonic() - start
                return response.json()

            try:
                episode = await request("reset", {"task_id": int(info["task_id"]),
                                                   "env_seed": int(info["env_seed"])})
                if episode["initial_state_id"] != info["initial_state_id"]:
                    raise RuntimeError("Environment reset differs from the fixed dataset initial state")
                max_steps = int(os.environ.get("SPA_MAX_STEPS", "10"))
                if max_steps != episode["max_steps"]:
                    raise ValueError("Agent and environment max_steps differ")
                action_budget = int(os.environ.get("SPA_MAX_ACTION_TOKENS", "256"))
                if action_budget < 1:
                    raise ValueError("Action token budget must be positive")
                prompt = await self.apply_chat_template(list(kwargs["raw_prompt"]))
                if len(prompt) > cfg.data.max_prompt_length:
                    raise ValueError("Initial prompt exceeds max_prompt_length")
                if self.tokenizer.convert_tokens_to_ids("<|im_end|>") != self.tokenizer.eos_token_id:
                    raise ValueError("V0 requires Qwen Instruct ChatML with im_end as EOS")
                trace = TokenTrace(prompt, cfg.data.max_response_length)
                turns, outcome, success, reason = [], 0., False, "max_steps"
                request_id = uuid4().hex
                for step in range(max_steps):
                    context = trace.context
                    params = dict(sampling_params, max_tokens=min(action_budget, trace.remaining))
                    if params["max_tokens"] <= 0:
                        reason = "context_budget"
                        break
                    start = time.monotonic()
                    output = await self.server_manager.generate(
                        request_id=request_id, prompt_ids=context, sampling_params=params)
                    generated_seconds += time.monotonic() - start
                    if output.stop_reason == "aborted":
                        raise RuntimeError("Inference engine aborted generation")
                    ids = output.token_ids
                    if not ids or ids[-1] != self.tokenizer.eos_token_id:
                        # Retain raw partial output in the audit record but give it no
                        # policy/KL mask, and never execute a truncated action.
                        span = trace.append(ids)
                        turns.append(dict(step_id=step, context_token_ids=context,
                                          generated_token_ids=ids, behavior_logprobs=output.log_probs,
                                          action_span=span, sampling_config=dict(params), complete=False))
                        reason = "incomplete_action"
                        break
                    span = trace.append(ids, action_id=step, logprobs=output.log_probs, train=True)
                    text = self.tokenizer.decode(ids, skip_special_tokens=True)
                    result = await request("step", dict(episode_id=episode["episode_id"], step_id=step, text=text))
                    turns.append(dict(step_id=step, context_token_ids=context,
                                      generated_token_ids=ids, behavior_logprobs=output.log_probs,
                                      action_span=span, sampling_config=dict(params), complete=True,
                                      **{k: v for k, v in result.items() if k != "step_id"}))
                    outcome, success = result["outcome_reward"], result["success"]
                    if result["terminated"] or result["truncated"]:
                        reason = "terminated" if result["terminated"] else "max_steps"
                        break
                    suffix = trace.observation_suffix(self.tokenizer, result["observation"])
                    if len(suffix) >= trace.remaining:
                        reason, outcome = "context_budget", 0.
                        break
                    trace.append(suffix)
                if not any(trace.mask) and cfg.algorithm.adv_estimator != "spa_v1":
                    # A zero-mask sample cannot silently participate in GRPO group
                    # statistics. Abort for a larger budget instead of biasing groups.
                    raise RuntimeError("No complete action; increase action/context budget before training")
                record = dict(schema_version=0, task_id=int(info["task_id"]),
                              episode_id=episode["episode_id"], group_id=f"{version['step']}:{version['sample_index']}",
                              initial_state_id=episode["initial_state_id"], env_seed=info["env_seed"],
                              policy_version=version, prm_version=None,
                              model=cfg.actor_rollout_ref.model.path,
                              tokenizer_template="qwen-instruct-chatml-v0",
                              template_sha256=hashlib.sha256(self.tokenizer.chat_template.encode()).hexdigest(),
                              sampling_config=dict(sampling_params), steps=turns,
                              prompt_ids=trace.prompt_ids, response_ids=trace.response_ids,
                              response_mask=trace.mask, action_ids=trace.action_ids,
                              outcome_reward=outcome, success=success, termination_reason=reason)
                dest = Path(os.environ.get("SPA_TRAJECTORY_DIR", "outputs/v0/trajectories"))
                if cfg.algorithm.adv_estimator == "spa_v1":
                    record["schema_version"] = 1
                    record["prm_version"] = cfg.algorithm.spa_v1.prm_version if cfg.algorithm.spa_v1.arm == "C" else None
                    record["sft_version"] = cfg.algorithm.spa_v1.sft_version
                    record["actor_version"] = dict(base_sft=cfg.algorithm.spa_v1.sft_version,
                                                   experiment=cfg.algorithm.spa_v1.arm, **version)
                dest.mkdir(parents=True, exist_ok=True)
                (dest / (episode["episode_id"] + ".json")).write_text(json.dumps(record), encoding="utf-8")
                extra_fields = {"action_ids": trace.action_ids,
                                "reward_extra_info": {"success": float(success),
                                                      "steps": len(turns),
                                                      "budget_exhausted": float(reason != "terminated")}}
                if cfg.algorithm.adv_estimator == "spa_v1":
                    # Preserve original, unpadded tokens. PRM action endpoints
                    # include the generated EOS; no decoded/re-tokenized history.
                    extra_fields["spa_v1_trace"] = dict(
                        input_ids=trace.prompt_ids + trace.response_ids,
                        prompt_length=len(trace.prompt_ids),
                        action_end_positions=[len(trace.prompt_ids) + turn["action_span"][1] - 1
                                              for turn in turns if turn["complete"]],
                        complete_flags=[turn["complete"] for turn in turns],
                        termination_reason=reason, episode_id=episode["episode_id"],
                        initial_state_id=episode["initial_state_id"],
                        task_id=int(info["task_id"]), policy_version=version,
                        actor_version=record["actor_version"], sft_version=record["sft_version"],
                        prm_version=record["prm_version"],
                        trajectory_path=str(dest / (episode["episode_id"] + ".json")))
                return AgentLoopOutput(prompt_ids=trace.prompt_ids, response_ids=trace.response_ids,
                                       response_mask=trace.mask, response_logprobs=trace.logprobs,
                                       reward_score=outcome, num_turns=2 * len(turns),
                                       metrics=dict(generate_sequences=generated_seconds, tool_calls=env_seconds),
                                       extra_fields=extra_fields)
            finally:
                if episode is not None:
                    try:
                        await request("close", {"episode_id": episode["episode_id"]})
                    except Exception:
                        logging.exception("Environment cleanup failed; service TTL will release the session")
