"""
LoRA on MLX. Base weights frozen
"""

import os
import sys
import argparse
import types
from pathlib import Path
from typing import Optional
from config import PipelineConfig
import mlx_lm.lora as mlx_lora


def run_lora_training(
    config: Optional[PipelineConfig] = None,
    model_override: Optional[str] = None,
    iters_override: Optional[int] = None,
    batch_size_override: Optional[int] = None,
    lr_override: Optional[float] = None,
    rank_override: Optional[int] = None,
    data_dir_override: Optional[str] = None,
    adapter_path_override: Optional[str] = None,
    dry_run: bool = False
):
    cfg = config or PipelineConfig()
    if model_override:
        cfg.set_base_model(model_override)

    active_model = cfg.active_model
    lora_cfg = cfg.lora
    base_dir = cfg.config_path.parent

    model_id = active_model.get("mlx_repo")
    data_dir = Path(data_dir_override or (base_dir / cfg.dataset.get("manifest_dir", "./data/manifests"))).resolve()
    train_file = data_dir / "train.jsonl"
    if not train_file.exists():
        if dry_run:
            print(f"[NOTE] Missing training set: {train_file}")
        else:
            raise FileNotFoundError(f"Missing training set: {train_file}")

    adapter_dir = Path(adapter_path_override or (base_dir / lora_cfg.get("adapter_dir", "./data/adapters/lora"))).resolve()
    adapter_dir.mkdir(parents=True, exist_ok=True)

    token = cfg.get_token()
    if token:
        os.environ["HF_TOKEN"] = token

    # LoRA hypercoefficients
    rank = rank_override or lora_cfg.get("rank", 16)            # Adapter rank r
    alpha = lora_cfg.get("alpha", 32.0)                         # Scaling factor alpha
    dropout = lora_cfg.get("dropout", 0.05)
    lora_scale = alpha / rank if rank > 0 else 2.0             # Scale coefficient
    iters = iters_override or lora_cfg.get("iters", 1500)       # Optimization iterations
    batch_size = batch_size_override or lora_cfg.get("batch_size", 2)
    lr = lr_override or lora_cfg.get("learning_rate", 1.0e-4)   # Peak learning rate
    grad_accum = lora_cfg.get("grad_accumulation_steps", 4)
    num_layers = lora_cfg.get("num_layers", 16)                 # Number of layers to adapt

    print(f"[LoRA] Model: {active_model.get('name')} | Rank: {rank} | Scale: {lora_scale:.2f} | LR: {lr} | Iters: {iters}")

    if dry_run:
        print("[DRY RUN] Configuration validated successfully.")
        return

    training_args = types.SimpleNamespace(
        model=model_id,
        train=True,
        fine_tune_type="lora",
        optimizer=lora_cfg.get("optimizer", "adamw"),
        optimizer_config={},
        data=str(data_dir),
        seed=42,
        num_layers=num_layers,
        batch_size=batch_size,
        iters=iters,
        val_batches=25,
        learning_rate=lr,
        steps_per_report=lora_cfg.get("steps_per_report", 10),
        steps_per_eval=lora_cfg.get("steps_per_eval", 100),
        resume_adapter_file=None,
        adapter_path=str(adapter_dir),
        save_every=lora_cfg.get("save_every", 250),
        test=False,
        test_batches=50,
        max_seq_length=lora_cfg.get("max_seq_length", 2048),
        config=None,
        grad_checkpoint=lora_cfg.get("grad_checkpoint", True),
        grad_accumulation_steps=grad_accum,
        clear_cache_threshold=lora_cfg.get("clear_cache_threshold", 0.8),
        lr_schedule=None,
        lora_parameters={"rank": rank, "scale": lora_scale, "dropout": dropout},
        mask_prompt=True,
        report_to=None,
        project_name="odia-stt-lora",
    )

    mlx_lora.run(training_args)
    print(f"[LoRA] Completed. Weights saved to: {adapter_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Odia Speech LoRA Fine-Tuning (MLX)")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model", type=str, choices=["qwen_72b", "gemma_32b_it"], default=None)
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = PipelineConfig(args.config)
    run_lora_training(
        config=cfg,
        model_override=args.model,
        iters_override=args.iters,
        batch_size_override=args.batch_size,
        lr_override=args.lr,
        rank_override=args.rank,
        data_dir_override=args.data,
        adapter_path_override=args.adapter_path,
        dry_run=args.dry_run
    )
