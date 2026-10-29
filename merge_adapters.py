"""
Fuses trained LoRA adapters into base model weights.
Generates a standalone model directory for local deployment using MLX.
"""

import os
import argparse
from pathlib import Path
from typing import Optional
from config import PipelineConfig
from mlx_lm.utils import load, save, tree_unflatten, dequantize_model


def merge_lora_adapters(
    config: Optional[PipelineConfig] = None,
    model_override: Optional[str] = None,
    adapter_path_override: Optional[str] = None,
    save_path_override: Optional[str] = None,
    dequantize: bool = False,
    dry_run: bool = False
):
    cfg = config or PipelineConfig()
    if model_override:
        cfg.set_base_model(model_override)

    active_model = cfg.active_model
    base_dir = cfg.config_path.parent

    model_id = active_model.get("mlx_repo")
    adapter_path = Path(adapter_path_override or (base_dir / cfg.lora.get("adapter_dir", "./data/adapters/lora"))).resolve()
    save_path = Path(save_path_override or (base_dir / "./data/fused_model")).resolve()

    adapter_weights = adapter_path / "adapters.safetensors"
    if not adapter_weights.exists():
        if dry_run:
            print(f"[NOTE] Missing adapter weights: {adapter_weights}")
        else:
            raise FileNotFoundError(f"Missing adapter weights: {adapter_weights}")

    print(f"[Merge] Model: {active_model.get('name')} | Adapters: {adapter_path} | Out: {save_path}")

    if dry_run:
        print("[DRY RUN] Fusion configuration validated successfully.")
        return

    save_path.mkdir(parents=True, exist_ok=True)
    token = cfg.get_token()
    if token:
        os.environ["HF_TOKEN"] = token

    model, tokenizer, model_config = load(
        model_id,
        adapter_path=str(adapter_path),
        return_config=True
    )

    fused_linears = [
        (n, m.fuse(dequantize=dequantize))
        for n, m in model.named_modules()
        if hasattr(m, "fuse")
    ]

    if fused_linears:
        model.update_modules(tree_unflatten(fused_linears))

    if dequantize:
        model = dequantize_model(model)
        model_config.pop("quantization", None)
        model_config.pop("quantization_config", None)

    save(save_path, model, tokenizer=tokenizer, config=model_config)
    print(f"[Merge] Completed. Fused model saved to: {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Merge LoRA Adapters into Base Model (MLX)")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model", type=str, choices=["qwen_72b", "gemma_32b_it"], default=None)
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--dequantize", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = PipelineConfig(args.config)
    merge_lora_adapters(
        config=cfg,
        model_override=args.model,
        adapter_path_override=args.adapter_path,
        save_path_override=args.save_path,
        dequantize=args.dequantize,
        dry_run=args.dry_run
    )
