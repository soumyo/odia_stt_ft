"""
Master CLI orchestrator for the Odia Speech & Dialect Fine-Tuning Pipeline.
Dispatches audio ingestion, dataset preparation, LoRA/full training, and evaluation.
"""

import time
import argparse
from typing import Optional
from config import PipelineConfig
from audio_preprocessor import AudioPreprocessor
from dataset_builder import DatasetBuilder
from generate_sample_data import generate_sample_dataset
from train_lora import run_lora_training
from train_full import run_full_fine_tuning
from merge_adapters import merge_lora_adapters
from eval_suite import OdiaEvalSuite
from inference import OdiaInferenceEngine, interactive_session


def execute_pipeline_step(
    step: str,
    cfg: PipelineConfig,
    model_override: Optional[str] = None,
    dry_run: bool = False,
    prompt: Optional[str] = None,
    audio: Optional[str] = None,
    dialect: str = "or_std",
    interactive: bool = False
):
    start = time.time()
    if model_override:
        cfg.set_base_model(model_override)

    print(f"\n[Pipeline] Step: '{step.upper()}' | Model: {cfg.active_model.get('name')}")

    if step == "sample":
        generate_sample_dataset(cfg)

    elif step == "ingest":
        chunks = AudioPreprocessor(cfg).process_directory()
        print(f"[Ingest] Processed {len(chunks)} chunks.")

    elif step == "prepare":
        DatasetBuilder(cfg).create_golden_eval_suite()
        print("[Prepare] Generated dataset manifests.")

    elif step == "lora":
        run_lora_training(cfg, dry_run=dry_run)

    elif step == "full":
        run_full_fine_tuning(cfg, dry_run=dry_run)

    elif step == "merge":
        merge_lora_adapters(cfg, dry_run=dry_run)

    elif step == "eval":
        OdiaEvalSuite(cfg).run_golden_suite()

    elif step == "infer":
        engine = OdiaInferenceEngine(cfg)
        if interactive:
            interactive_session(engine)
        elif audio:
            print(engine.transcribe_audio_clip(audio, dialect=dialect))
        elif prompt:
            tag = f"<|{dialect}|>" if not dialect.startswith("<|") else dialect
            print(engine.generate(prompt, dialect=tag))

    elif step == "all":
        generate_sample_dataset(cfg)
        DatasetBuilder(cfg).create_golden_eval_suite()
        run_lora_training(cfg, dry_run=dry_run)
        OdiaEvalSuite(cfg).run_golden_suite()

    else:
        raise ValueError(f"Unknown step: {step}")

    print(f"[Pipeline] Step '{step}' finished in {time.time() - start:.2f}s.\n")


def parse_cli():
    parser = argparse.ArgumentParser(description="Odia Speech Fine-Tuning CLI (MLX)")
    parser.add_argument("--step", type=str, required=True,
                        choices=["sample", "ingest", "prepare", "lora", "full", "merge", "eval", "infer", "all"])
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model", type=str, choices=["qwen_72b", "gemma_32b_it"], default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--audio", type=str, default=None)
    parser.add_argument("--dialect", type=str, default="or_std")
    parser.add_argument("--interactive", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_cli()
    cfg = PipelineConfig(args.config)
    execute_pipeline_step(
        step=args.step,
        cfg=cfg,
        model_override=args.model,
        dry_run=args.dry_run,
        prompt=args.prompt,
        audio=args.audio,
        dialect=args.dialect,
        interactive=args.interactive
    )
