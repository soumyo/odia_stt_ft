"""
Loads and validates pipeline configuration from YAML and environment variables.
Provides hardware profiling and model configuration resolution.
"""

import os
import re
import sys
import yaml
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional


def resolve_env_vars(val: Any) -> Any:
    """Recursively interpolates ${VAR_NAME} using environment variables."""
    if isinstance(val, str):
        pattern = re.compile(r"\$\{([^}^{]+)\}")
        for var in pattern.findall(val):
            val = val.replace(f"${{{var}}}", os.getenv(var, ""))
        return val
    elif isinstance(val, dict):
        return {k: resolve_env_vars(v) for k, v in val.items()}
    elif isinstance(val, list):
        return [resolve_env_vars(v) for v in val]
    return val


class PipelineConfig:
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = str(Path(__file__).parent / "config.yaml")

        self.config_path = Path(config_path).resolve()
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            raw_cfg = yaml.safe_load(f)

        self._cfg: Dict[str, Any] = resolve_env_vars(raw_cfg)
        self._ensure_directories()

    def _ensure_directories(self):
        base_dir = self.config_path.parent
        paths_to_create = [
            base_dir / self._cfg["audio"]["src_dir"],
            base_dir / self._cfg["audio"]["processed_dir"],
            base_dir / self._cfg["dataset"]["manifest_dir"],
            base_dir / self._cfg["lora"]["adapter_dir"],
            base_dir / self._cfg["full_fine_tune"]["checkpoint_dir"],
            base_dir / self._cfg["eval"]["results_dir"],
        ]
        for p in paths_to_create:
            p.mkdir(parents=True, exist_ok=True)

    @property
    def raw(self) -> Dict[str, Any]:
        return self._cfg

    @property
    def audio(self) -> Dict[str, Any]:
        return self._cfg["audio"]

    @property
    def api(self) -> Dict[str, Any]:
        return self._cfg["api"]

    @property
    def base_model_key(self) -> str:
        return self._cfg.get("base_model", "qwen_72b")

    def set_base_model(self, model_key: str):
        models = self._cfg.get("models", {})
        if model_key not in models:
            raise ValueError(f"Unknown model '{model_key}'. Valid options: {list(models.keys())}")
        self._cfg["base_model"] = model_key

    @property
    def active_model(self) -> Dict[str, Any]:
        key = self.base_model_key
        models = self._cfg.get("models", {})
        if key not in models:
            raise KeyError(f"Base model '{key}' is not configured.")
        return models[key]

    @property
    def lora(self) -> Dict[str, Any]:
        return self._cfg["lora"]

    @property
    def full_fine_tune(self) -> Dict[str, Any]:
        return self._cfg["full_fine_tune"]

    @property
    def dialects(self) -> Dict[str, Any]:
        return self._cfg["dialects"]

    @property
    def dataset(self) -> Dict[str, Any]:
        return self._cfg["dataset"]

    @property
    def eval(self) -> Dict[str, Any]:
        return self._cfg["eval"]

    def get_token(self) -> Optional[str]:
        token = self.api.get("hf_token") or self.api.get("model_api_key")
        if not token:
            token = os.getenv("HF_TOKEN") or os.getenv("MODEL_API_KEY")
        return token if token else None

    def check_hardware(self) -> Dict[str, Any]:
        hw_info = {
            "platform": sys.platform,
            "is_mac": sys.platform == "darwin",
            "arch": subprocess.getoutput("uname -m"),
            "ram_gb": 0.0,
            "mlx_device": "Unknown",
        }
        if hw_info["is_mac"]:
            try:
                mem_bytes = int(subprocess.getoutput("sysctl -n hw.memsize"))
                hw_info["ram_gb"] = round(mem_bytes / (1024 ** 3), 1)
            except Exception:
                pass

        try:
            import mlx.core as mx
            hw_info["mlx_device"] = str(mx.default_device())
        except ImportError:
            hw_info["mlx_device"] = "mlx not installed"

        return hw_info

    def print_summary(self):
        hw = self.check_hardware()
        model = self.active_model
        print(f"[Config] Model: {model.get('name')} ({model.get('quantization')}) | Device: {hw['mlx_device']} | RAM: {hw['ram_gb']}GB")


if __name__ == "__main__":
    cfg = PipelineConfig()
    cfg.print_summary()
