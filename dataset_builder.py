"""
Generates training, validation, and test dataset manifests in MLX chat format.
Builds the reference evaluation suite across Odia dialects.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from config import PipelineConfig
from odia_phonetics import OdiaDialectG2P

GOLDEN_EVAL_PROMPTS = [
    {
        "dialect_id": "or_std",
        "category": "schwa_retention_trap",
        "text_orthographic": "ସକାଳୁ ସକାଳୁ ଗରମ ଭାତ ଆଉ ଡାଲମା ଖାଇବାକୁ ଭଲ ଲାଗେ।",
        "expected_schwa_words": ["ଭାତ", "ଭଲ", "ଗରମ"],
        "speaker_id": "std_female_01"
    },
    {
        "dialect_id": "or_std",
        "category": "schwa_retention_trap",
        "text_orthographic": "ଘର କାମ ସାରି ସେ ନିଜ ପାଠ ପଢ଼ିବାକୁ ବସିଲା।",
        "expected_schwa_words": ["କାମ", "ପାଠ"],
        "speaker_id": "std_male_01"
    },
    {
        "dialect_id": "or_bam",
        "category": "schwa_retention_trap",
        "text_orthographic": "ଆଜି ସଞ୍ଜ ବେଳେ ବଡ଼ ବଜାର ଯାଇ ଖଟା ମିଠା ଆଚାର ଆଣିବ।",
        "expected_schwa_words": ["ସଞ୍ଜ", "ବଜାର", "ଆଚାର"],
        "speaker_id": "bam_female_01"
    },
    {
        "dialect_id": "or_sbp",
        "category": "dialect_rhythm_clusters",
        "text_orthographic": "ତୁଇ କେନ୍ ଆଡ଼େ ଯାଉଛୁ ବୋଇଲେ ଜଲ୍‌ଦି କହ, ବେଳ ବୁଡ଼ି ଯାଉଛେ!",
        "expected_schwa_words": [],
        "speaker_id": "sbp_male_01"
    },
    {
        "dialect_id": "or_sbp",
        "category": "dialect_rhythm_clusters",
        "text_orthographic": "କାଲି ସକାଲୁ ଆମର ଗାଁ କେ ଯିବାର ଅଛେ, ସବୁ ଜିନିଷ ସଜାଡ଼ି ରଖ!",
        "expected_schwa_words": [],
        "speaker_id": "sbp_female_01"
    },
    {
        "dialect_id": "or_bam",
        "category": "pitch_contour_melody",
        "text_orthographic": "ଆରେ ବାପା! ତମେ ସତରେ ବ୍ରହ୍ମପୁରରୁ ଆଚାର ଆଣିଚ ନା କ’ଣ?",
        "expected_schwa_words": ["ଆଚାର"],
        "speaker_id": "bam_male_01"
    },
    {
        "dialect_id": "or_bls",
        "category": "code_switching_numerals",
        "text_orthographic": "ମୋର ବ୍ୟାଙ୍କ ଆକାଉଣ୍ଟରୁ 2,450 ଟଙ୍କା UPI ରେ ଟ୍ରାନ୍ସଫର ହେଇଗଲା।",
        "expected_schwa_words": ["ବ୍ୟାଙ୍କ"],
        "speaker_id": "bls_female_01"
    },
    {
        "dialect_id": "or_std",
        "category": "duplex_turn_taking",
        "text_orthographic": "ହଁ ଆଜ୍ଞା, ଆପଣ ଯାହା କହିଲେ ମୁଁ ସବୁ ବୁଝିପାରିଲି, ଠିକ୍ ଅଛି।",
        "expected_schwa_words": ["ଠିକ୍"],
        "speaker_id": "std_female_02"
    }
]


class DatasetBuilder:
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.cfg = config or PipelineConfig()
        self.g2p = OdiaDialectG2P()
        base_dir = self.cfg.config_path.parent
        self.manifest_dir = (base_dir / self.cfg.dataset.get("manifest_dir", "./data/manifests")).resolve()
        self.manifest_dir.mkdir(parents=True, exist_ok=True)

        self.train_path = self.manifest_dir / "train.jsonl"
        self.valid_path = self.manifest_dir / "valid.jsonl"
        self.test_path = self.manifest_dir / "test.jsonl"
        self.golden_path = self.manifest_dir / "eval_golden_odia.jsonl"

    def build_from_records(
        self,
        records: List[Dict[str, Any]],
        train_ratio: float = 0.8,
        valid_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42
    ) -> Dict[str, int]:
        random.seed(seed)
        shuffled = records.copy()
        random.shuffle(shuffled)

        n = len(shuffled)
        n_train = max(1, int(n * train_ratio))
        n_valid = max(1, int(n * valid_ratio)) if (n - n_train) > 1 else 0

        train_recs = shuffled[:n_train]
        valid_recs = shuffled[n_train:n_train + n_valid]
        test_recs = shuffled[n_train + n_valid:]
        if not test_recs and valid_recs:
            test_recs = valid_recs

        splits = [
            ("train", train_recs, self.train_path),
            ("valid", valid_recs, self.valid_path),
            ("test", test_recs, self.test_path)
        ]

        counts = {}
        for split_name, split_data, split_file in splits:
            with open(split_file, "w", encoding="utf-8") as f_mlx:
                for r in split_data:
                    chat_sample = self.g2p.format_mlx_chat_sample(r)
                    f_mlx.write(json.dumps(chat_sample, ensure_ascii=False) + "\n")

            rich_file = self.manifest_dir / f"{split_name}_manifest.jsonl"
            with open(rich_file, "w", encoding="utf-8") as f_rich:
                for r in split_data:
                    f_rich.write(json.dumps(r, ensure_ascii=False) + "\n")

            counts[split_name] = len(split_data)
        return counts

    def create_golden_eval_suite(self) -> int:
        golden_records = []
        for idx, item in enumerate(GOLDEN_EVAL_PROMPTS):
            utt_id = f"golden_{item['dialect_id']}_{idx:03d}"
            rec = self.g2p.generate_dual_text_record(
                utt_id=utt_id,
                text_orthographic=item["text_orthographic"],
                dialect_id=item["dialect_id"],
                speaker_id=item["speaker_id"],
                duration=4.5
            )
            rec["category"] = item["category"]
            rec["expected_schwa_words"] = item.get("expected_schwa_words", [])
            golden_records.append(rec)

        with open(self.golden_path, "w", encoding="utf-8") as f:
            for r in golden_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        return len(golden_records)


if __name__ == "__main__":
    builder = DatasetBuilder()
    builder.create_golden_eval_suite()
