"""Launch one controlled SFT-initialized V1 experiment or its validation."""
import argparse
import hashlib
import importlib.metadata
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", choices=("A", "B", "C"), required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--val", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prm-url", default="http://127.0.0.1:8775")
    p.add_argument("--prm-version")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--evaluation-adapter", type=Path,
                   help="Load a historical adapter for evaluation only; never for V1 training")
    p.add_argument("--resume-path", type=Path)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--policy-gpus", type=int, choices=(3, 4), default=4)
    p.add_argument("overrides", nargs="*")
    args = p.parse_args()
    if args.evaluation_adapter and (not args.eval_only or args.resume_path):
        p.error('--evaluation-adapter requires --eval-only and no --resume-path')
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    import verl
    if importlib.metadata.version("verl") != "0.7.0":
        raise RuntimeError("V1 requires verl==0.7.0")
    root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(config_dir=str(Path(verl.__file__).parent / "trainer/config"), version_base=None):
        cfg = compose(config_name="ppo_trainer")
    OmegaConf.set_struct(cfg, False)
    cfg = OmegaConf.merge(cfg,
        OmegaConf.load(root / "spa_next/configs/v0_qwen3_4b_lora32.yaml"),
        OmegaConf.load(root / "spa_next/configs/v1_qwen3_4b_lora32.yaml"))
    cfg.actor_rollout_ref.model.path = str(args.model.resolve())
    cfg.data.train_files, cfg.data.val_files = str(args.train.resolve()), str(args.val.resolve())
    cfg.trainer.default_local_dir = str(args.output.resolve())
    run_id = hashlib.sha256(str(args.output.resolve()).encode()).hexdigest()[:8]
    cfg.trainer.experiment_name = f"sft-spa-v1-{args.experiment}-{args.output.parent.name}-{run_id}"
    cfg.algorithm.spa_v1.arm = args.experiment
    cfg.algorithm.spa_v1.prm_url = args.prm_url
    cfg.algorithm.spa_v1.prm_version = args.prm_version
    cfg.algorithm.spa_v1.audit_dir = str(args.output.resolve().parent / "audit")
    cfg.actor_rollout_ref.actor.policy_loss.loss_mode = "spa_v1_" + args.experiment
    cfg.trainer.val_only = args.eval_only
    if args.resume_path:
        cfg.trainer.resume_mode = "resume_path"
        cfg.trainer.resume_from_path = str(args.resume_path.resolve())
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    if args.evaluation_adapter:
        cfg.actor_rollout_ref.model.lora_adapter_path = str(args.evaluation_adapter.resolve())
    if args.policy_gpus == 3:
        cfg.trainer.n_gpus_per_node = 3
        cfg.trainer.balance_batch = False
        cfg.actor_rollout_ref.actor.fsdp_config.fsdp_size = 3
        cfg.actor_rollout_ref.rollout.tensor_model_parallel_size = 1
        cfg.actor_rollout_ref.rollout.gpu_memory_utilization = 0.55
        # Use verl's fused linear cross-entropy path for Qwen3. It computes
        # token log-probs in chunks and avoids retaining [tokens, vocab] logits
        # through the actor backward pass.
        cfg.actor_rollout_ref.model.use_fused_kernels = True
        cfg.actor_rollout_ref.model.fused_kernel_options = {"impl_backend": "torch"}
        cfg.actor_rollout_ref.rollout.free_cache_engine = False
        cfg.actor_rollout_ref.rollout.layered_summon = True
    if (cfg.algorithm.adv_estimator != "spa_v1" or cfg.critic.enable or cfg.reward_model.enable
            or (cfg.actor_rollout_ref.model.lora_adapter_path is not None and not args.evaluation_adapter)
            or cfg.actor_rollout_ref.model.lora_rank != 32):
        raise ValueError("V1 requires merged SFT base + fresh rank32 LoRA, no critic or built-in reward model")
    if (cfg.data.train_batch_size != 4 or cfg.actor_rollout_ref.rollout.n != 4
            or cfg.data.shuffle or cfg.actor_rollout_ref.actor.ppo_mini_batch_size != 4
            or cfg.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu != 1
            or cfg.actor_rollout_ref.actor.ppo_epochs != 1
            or cfg.actor_rollout_ref.actor.use_dynamic_bsz
            or cfg.actor_rollout_ref.actor.entropy_coeff != 0
            or cfg.actor_rollout_ref.actor.calculate_entropy
            or cfg.algorithm.rollout_correction is not None
            or cfg.algorithm.use_kl_in_reward):
        raise ValueError("V1 controlled batch, normalization or loss settings changed")
    if cfg.algorithm.spa_v1.arm != args.experiment or cfg.actor_rollout_ref.actor.policy_loss.loss_mode != "spa_v1_" + args.experiment:
        raise ValueError("Experiment arm and actor loss disagree")
    cfg.actor_rollout_ref.rollout.agent.agent_loop_config_path = str(root / "spa_next/configs/agent_loop.yaml")
    runtime = dict(PYTHONPATH=str(root), OMP_NUM_THREADS="1", HF_HUB_OFFLINE="1",
                   TRANSFORMERS_OFFLINE="1", MALLOC_ARENA_MAX="2", MALLOC_TRIM_THRESHOLD_="131072",
                   SPA_MAX_STEPS="10", SPA_MAX_ACTION_TOKENS="256",
                   SPA_TRAJECTORY_DIR=str(args.output.resolve().parent / "trajectories"))
    for name in ("SPA_ENV_URL", "SPA_TRAJECTORY_DIR"):
        if name in os.environ:
            runtime[name] = os.environ[name]
    OmegaConf.update(cfg, "ray_kwargs.ray_init.runtime_env.env_vars", runtime, merge=False)
    OmegaConf.update(cfg, "ray_kwargs.ray_init._temp_dir", str(root / "ray"))
    if args.dry_run:
        print(OmegaConf.to_yaml(cfg, resolve=True))
        return
    from spa_next.training.patch_verl_v1 import MARKER, PATCHERS
    for relative in PATCHERS:
        if MARKER not in (Path(verl.__file__).parent / relative).read_text():
            raise RuntimeError("Run spa_next.training.patch_verl_v1 before V1 launch")
    from spa_next.training.sft_snapshot import verify_snapshot
    if args.evaluation_adapter:
        import json
        from spa_next.training.sft_snapshot import sha256
        adapter_config = json.loads((args.evaluation_adapter / 'adapter_config.json').read_text())
        if adapter_config['r'] != 32:
            raise ValueError('Historical evaluation expects rank 32 adapter')
        index = json.loads((args.model / 'model.safetensors.index.json').read_text())
        provenance = dict(kind='historical_adapter_evaluation', base=str(args.model.resolve()),
            adapter=str(args.evaluation_adapter.resolve()),
            base_weights={name: sha256(args.model / name) for name in sorted(set(index['weight_map'].values()))},
            adapter_sha256=sha256(args.evaluation_adapter / 'adapter_model.safetensors'))
        cfg.algorithm.spa_v1.sft_version = 'historical-eval-' + hashlib.sha256(
            json.dumps(provenance, sort_keys=True).encode()).hexdigest()
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'evaluation_provenance.json').write_text(json.dumps(provenance, indent=2))
    else:
        snapshot = verify_snapshot(args.model)
        cfg.algorithm.spa_v1.sft_version = snapshot["version"]
    if args.experiment == "C" and not args.eval_only:
        if not args.prm_version:
            raise ValueError("Experiment C requires a pinned PRM version")
        from spa_next.prm.client import PRMClient
        health = PRMClient(args.prm_url, args.prm_version).health()
        if not (health.get("precision_gate_passed") or health.get("quantization_gate_passed")):
            raise RuntimeError("PRM quantization comparison gate not passed")
    args.output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, args.output / "resolved_config.yaml", resolve=True)
    from verl.trainer.main_ppo import run_ppo
    run_ppo(cfg)


if __name__ == "__main__":
    main()
