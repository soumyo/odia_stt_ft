"""
MLX Fine tuning pipleine
"""

import os
import sys
import argparse
import types
from pathlib import Path
from typing import Optional
from config import PipelineConfig
import mlx_lm.lora as mlx_lora


def run_full_fine_tuning(
    config: Optional[PipelineConfig] = None,
    model_override: Optional[str] = None,
    iters_override: Optional[int] = None,
    batch_size_override: Optional[int] = None,
    lr_override: Optional[float] = None,
    num_layers_override: Optional[int] = None,
    data_dir_override: Optional[str] = None,
    checkpoint_dir_override: Optional[str] = None,
    dry_run: bool = False
):
    cfg = config or PipelineConfig()
    if model_override:
        cfg.set_base_model(model_override)

    active_model = cfg.active_model
    ft_cfg = cfg.full_fine_tune
    base_dir = cfg.config_path.parent

    model_id = active_model.get("mlx_repo")
    data_dir = Path(data_dir_override or (base_dir / cfg.dataset.get("manifest_dir", "./data/manifests"))).resolve()
    train_file = data_dir / "train.jsonl"
    if not train_file.exists():
        if dry_run:
            print(f"[NOTE] Missing training set: {train_file}")
        else:
            raise FileNotFoundError(f"Missing training set: {train_file}")

    ckpt_dir = Path(checkpoint_dir_override or (base_dir / ft_cfg.get("checkpoint_dir", "./data/checkpoints/full_ft"))).resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    token = cfg.get_token()
    if token:
        os.environ["HF_TOKEN"] = token

    # Training parameters
    iters = iters_override or ft_cfg.get("iters", 1000)          # Iteration steps
    batch_size = batch_size_override or ft_cfg.get("batch_size", 1)
    lr = lr_override or ft_cfg.get("learning_rate", 1.5e-5)      # Target learning rate
    grad_accum = ft_cfg.get("grad_accumulation_steps", 8)        # Gradient accumulation
    num_layers = num_layers_override or ft_cfg.get("num_layers", 8) # Number of top layers

    print(f"[Full-FT] Model: {active_model.get('name')} | Layers: {num_layers} | LR: {lr} | Iters: {iters}")

    if dry_run:
        print("[DRY RUN] Configuration validated successfully.")
        return

    training_args = types.SimpleNamespace(
        model=model_id,
        train=True,
        fine_tune_type="full",
        optimizer=ft_cfg.get("optimizer", "adamw"),
        optimizer_config={},
        data=str(data_dir),
        seed=42,
        num_layers=num_layers,
        batch_size=batch_size,
        iters=iters,
        val_batches=25,
        learning_rate=lr,
        steps_per_report=ft_cfg.get("steps_per_report", 10),
        steps_per_eval=ft_cfg.get("steps_per_eval", 100),
        resume_adapter_file=None,
        adapter_path=str(ckpt_dir),
        save_every=ft_cfg.get("save_every", 200),
        test=False,
        test_batches=50,
        max_seq_length=ft_cfg.get("max_seq_length", 2048),
        config=None,
        grad_checkpoint=ft_cfg.get("grad_checkpoint", True),
        grad_accumulation_steps=grad_accum,
        clear_cache_threshold=ft_cfg.get("clear_cache_threshold", 0.85),
        lr_schedule=None,
        lora_parameters=None,
        mask_prompt=True,
        report_to=None,
        project_name="odia-stt-full-ft",
    )

    mlx_lora.run(training_args)
    print(f"[Full-FT] Completed. Checkpoint saved to: {ckpt_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Odia Speech Full Fine-Tuning (MLX)")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model", type=str, choices=["qwen_72b", "gemma_32b_it"], default=None)
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = PipelineConfig(args.config)
    run_full_fine_tuning(
        config=cfg,
        model_override=args.model,
        iters_override=args.iters,
        batch_size_override=args.batch_size,
        lr_override=args.lr,
        num_layers_override=args.num_layers,
        data_dir_override=args.data,
        checkpoint_dir_override=args.checkpoint_dir,
        dry_run=args.dry_run
    )
