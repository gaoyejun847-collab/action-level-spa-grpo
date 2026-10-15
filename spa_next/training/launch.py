"""Build the pinned verl configuration, then run its online training loop."""
import argparse
import importlib.metadata
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="spa_next/configs/v0_qwen3_4b_lora32.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    import verl
    if importlib.metadata.version("verl") != "0.7.0":
        raise RuntimeError("This adapter is verified against the verl 0.7.0 wheel schema")
    config_dir = Path(verl.__file__).parent / "trainer" / "config"
    with initialize_config_dir(config_dir=str(config_dir.resolve()), version_base=None):
        cfg = compose(config_name="ppo_trainer")
    OmegaConf.set_struct(cfg, False)
    cfg = OmegaConf.merge(cfg, OmegaConf.load(args.config), OmegaConf.from_dotlist(args.overrides))
    root = Path(__file__).resolve().parents[2]
    # Ray subprocesses must import this checkout even outside the driver's cwd.
    runtime_vars = dict(PYTHONPATH=str(root), OMP_NUM_THREADS="1")
    for name in ('MALLOC_ARENA_MAX', 'MALLOC_TRIM_THRESHOLD_'):
        if name in os.environ:
            runtime_vars[name] = os.environ[name]
    if Path(cfg.actor_rollout_ref.model.path).is_dir():
        # The weights/tokenizer are already local. vLLM otherwise tries to resolve
        # verl's in-memory LoRA placeholder as a Hugging Face repository.
        runtime_vars.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    for name in ("SPA_ENV_URL", "SPA_MAX_STEPS", "SPA_MAX_ACTION_TOKENS", "SPA_TRAJECTORY_DIR"):
        if name in os.environ:
            runtime_vars[name] = os.environ[name]
    runtime_vars.setdefault("SPA_TRAJECTORY_DIR", str(root / "outputs/v0/trajectories"))
    existing = OmegaConf.select(cfg, "ray_kwargs.ray_init.runtime_env.env_vars", default={})
    OmegaConf.update(cfg, "ray_kwargs.ray_init.runtime_env.env_vars",
                     {**runtime_vars, **dict(existing)}, merge=False)
    if OmegaConf.select(cfg, "ray_kwargs.ray_init._temp_dir") is None:
        OmegaConf.update(cfg, "ray_kwargs.ray_init._temp_dir", str(root / "ray"))
    for key in ("data.train_files", "data.val_files", "trainer.default_local_dir",
                "actor_rollout_ref.rollout.agent.agent_loop_config_path"):
        value = OmegaConf.select(cfg, key)
        OmegaConf.update(cfg, key, str(Path(value).resolve()))
    if Path(cfg.actor_rollout_ref.model.path).exists():
        cfg.actor_rollout_ref.model.path = str(Path(cfg.actor_rollout_ref.model.path).resolve())
    if cfg.algorithm.adv_estimator != "grpo" or cfg.reward_model.enable:
        raise ValueError("V0 uses outcome GRPO without a learned reward model")
    if cfg.actor_rollout_ref.model.lora_rank != 32:
        raise ValueError("This experiment fixes LoRA rank at 32")
    if cfg.actor_rollout_ref.rollout.skip_rollout:
        raise ValueError("Cached rollouts are forbidden for the V0 online baseline")
    context_limit = cfg.data.max_prompt_length + cfg.data.max_response_length
    if (cfg.actor_rollout_ref.rollout.max_model_len != context_limit
            or cfg.actor_rollout_ref.model.override_config.max_position_embeddings != context_limit):
        raise ValueError("verl 0.7 requires matching data, rollout and HF context limits")
    if args.dry_run:
        print(OmegaConf.to_yaml(cfg, resolve=True))
        return
    dest = Path(cfg.trainer.default_local_dir)
    dest.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, dest / "resolved_config.yaml", resolve=True)
    from verl.trainer.main_ppo import run_ppo
    run_ppo(cfg)


if __name__ == "__main__":
    main()
