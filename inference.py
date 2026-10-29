"""
Local inference engine for fine-tuned Odia STT models on Apple Silicon (MLX).
Supports prompt evaluation, interactive REPL, and audio chunk transcription.
"""

import os
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
from config import PipelineConfig
from odia_phonetics import OdiaDialectG2P
from audio_preprocessor import AudioPreprocessor
import mlx_lm


class OdiaInferenceEngine:
    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        model_override: Optional[str] = None,
        model_path_override: Optional[str] = None,
        adapter_path_override: Optional[str] = None,
        temp: float = 0.1,
        max_tokens: int = 512,
    ):
        self.cfg = config or PipelineConfig()
        if model_override:
            self.cfg.set_base_model(model_override)

        self.g2p = OdiaDialectG2P()
        self.temp = temp
        self.max_tokens = max_tokens
        base_dir = self.cfg.config_path.parent

        active = self.cfg.active_model
        self.model_id = model_path_override or active.get("mlx_repo")

        self.adapter_path = None
        if adapter_path_override:
            p = Path(adapter_path_override).resolve()
            if p.exists() and (p / "adapters.safetensors").exists():
                self.adapter_path = str(p)
        else:
            default_adapters = (base_dir / self.cfg.lora.get("adapter_dir", "./data/adapters/lora")).resolve()
            if (default_adapters / "adapters.safetensors").exists():
                self.adapter_path = str(default_adapters)

        token = self.cfg.get_token()
        if token:
            os.environ["HF_TOKEN"] = token

        self.model, self.tokenizer = mlx_lm.load(
            self.model_id,
            adapter_path=self.adapter_path
        )

    def generate(self, user_prompt: str, dialect: str = "<|or_std|>") -> str:
        system_prompt = (
            f"You are an Indic Speech-to-Text and Odia Dialect Transcription assistant. "
            f"Accurately transcribe spoken Odia and regional dialects ({dialect}) preserving phonetic cadence."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        if hasattr(self.tokenizer, "apply_chat_template"):
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            formatted_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"

        response = mlx_lm.generate(
            self.model,
            self.tokenizer,
            prompt=formatted_prompt,
            temp=self.temp,
            max_tokens=self.max_tokens,
            verbose=False
        )
        return response.strip()

    def transcribe_audio_clip(self, audio_path: str, dialect: str = "or_std") -> Dict[str, Any]:
        clean_dialect = dialect.replace("<|", "").replace("|>", "")
        prep = AudioPreprocessor(self.cfg)
        chunks = prep.process_file(audio_path, dialect_id=clean_dialect)
        
        if not chunks:
            return {"error": f"Failed to segment audio from: {audio_path}"}

        results = []
        for c in chunks:
            prompt = f"Transcribe speech segment with dialect tag <|{clean_dialect}|>.\nAudio chunk: {c['utt_id']} ({c['duration']}s)"
            text = self.generate(prompt, dialect=f"<|{clean_dialect}|>")
            results.append({"chunk_id": c["utt_id"], "duration": c["duration"], "transcription": text})

        return {"audio_path": audio_path, "dialect": clean_dialect, "segments": results}


def interactive_session(engine: OdiaInferenceEngine):
    print("Odia STT Interactive REPL. Type 'exit' to quit.")
    curr_dialect = "<|or_std|>"
    while True:
        try:
            line = input(f"[{curr_dialect}] > ").strip()
            if not line:
                continue
            if line.lower() in ("exit", "quit"):
                break
            if line.startswith("<|or_") and line.endswith("|>"):
                curr_dialect = line
                continue
            print(f"Result:\n{engine.generate(line, dialect=curr_dialect)}\n")
        except KeyboardInterrupt:
            break


def parse_args():
    parser = argparse.ArgumentParser(description="Odia STT Inference Engine (MLX)")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model", type=str, choices=["qwen_72b", "gemma_32b_it"], default=None)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--audio", type=str, default=None)
    parser.add_argument("--dialect", type=str, default="or_std")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--temp", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=512)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = PipelineConfig(args.config)
    engine = OdiaInferenceEngine(
        config=cfg,
        model_override=args.model,
        model_path_override=args.model_path,
        adapter_path_override=args.adapter_path,
        temp=args.temp,
        max_tokens=args.max_tokens
    )
    if args.interactive:
        interactive_session(engine)
    elif args.audio:
        print(engine.transcribe_audio_clip(args.audio, dialect=args.dialect))
    elif args.prompt:
        tag = f"<|{args.dialect}|>" if not args.dialect.startswith("<|") else args.dialect
        print(engine.generate(args.prompt, dialect=tag))
