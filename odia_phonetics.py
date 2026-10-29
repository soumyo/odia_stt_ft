"""
Odia Grapheme-to-Phoneme (G2P) engine and dialect normalizer.
Handles phonological rules for standard Odia, Sambalpuri, Berhampuri, and Baleswari.
"""

import re
from typing import Dict, Any, Optional

ODIA_CONSONANTS = {
    'କ': 'k', 'ଖ': 'kʰ', 'ଗ': 'ɡ', 'ଘ': 'ɡʰ', 'ଙ': 'ŋ',
    'ଚ': 't͡ʃ', 'ଛ': 't͡ʃʰ', 'ଜ': 'd͡ʒ', 'ଝ': 'd͡ʒʰ', 'ଞ': 'ɲ',
    'ଟ': 'ʈ', 'ଠ': 'ʈʰ', 'ଡ': 'ɖ', 'ଢ': 'ɖʰ', 'ଣ': 'ɳ',
    'ତ': 't̪', 'ଥ': 't̪ʰ', 'ଦ': 'd̪', 'ଧ': 'd̪ʰ', 'ନ': 'n',
    'ପ': 'p', 'ଫ': 'pʰ', 'ବ': 'b', 'ଭ': 'bʰ', 'ମ': 'm',
    'ଯ': 'd͡ʒ', 'ୟ': 'j', 'ର': 'r', 'ଲ': 'l', 'ଳ': 'ɭ',
    'ୱ': 'w', 'ଶ': 's', 'ଷ': 's', 'ସ': 's', 'ହ': 'h'
}

ODIA_VOWELS_MATRA = {
    'ା': 'aː', 'ି': 'i', 'ୀ': 'iː', 'ୁ': 'u', 'ୂ': 'uː',
    'ୃ': 'ru', 'େ': 'e', 'ୈ': 'ɔi', 'ୋ': 'o', 'ୌ': 'ɔu',
    '୍': ''  # Virama suppresses inherent vowel
}

ODIA_INDEPENDENT_VOWELS = {
    'ଅ': 'ɔ', 'ଆ': 'aː', 'ଇ': 'i', 'ଈ': 'iː', 'ଉ': 'u', 'ଊ': 'uː',
    'ଋ': 'ru', 'ଏ': 'e', 'ଐ': 'ɔi', 'ଓ': 'o', 'ଔ': 'ɔu'
}

ODIA_SIGNS = {
    'ଂ': 'ŋ',
    'ଁ': '̃',
    'ଃ': 'h'
}

SBP_TO_STD_LEXICON = {
    "ତୁଇ": "ତୁମେ",
    "କାଣା": "କ’ଣ",
    "କେନ୍": "କେଉଁ",
    "ବୋଇଲେ": "ବୋଲି",
    "ଯାଉଛେ": "ଯାଉଛି",
    "ଖାଉଛେ": "ଖାଉଛି",
    "କରୁଛେ": "କରୁଛି",
    "କେନ୍ତା": "କିପରି",
    "ଆସୁଛେ": "ଆସୁଛି",
    "ନାଇଁ": "ନାହିଁ",
    "ଇହାଦେ": "ବର୍ତ୍ତମାନ",
    "ବେଳ": "ସମୟ",
}

BAM_TO_STD_LEXICON = {
    "କରୁଚୁ": "କରୁଛୁ",
    "ଯାଉଚୁ": "ଯାଉଛୁ",
    "ଖାଉଚୁ": "ଖାଉଛୁ",
    "ଆସୁଚୁ": "ଆସୁଛୁ",
    "ଆଣିଚ": "ଆଣିଛ",
    "କହୁଚ": "କହୁଛ",
}


class OdiaDialectG2P:
    def __init__(self):
        pass

    def text_to_ipa(self, text: str, dialect: str = "or_std") -> str:
        out = []
        chars = list(text.strip())
        n = len(chars)
        punct_or_space = {' ', '\t', '\n', ',', '.', '।', '?', '!', ';', ':', '-', '—'}

        i = 0
        while i < n:
            ch = chars[i]

            if ch in ODIA_INDEPENDENT_VOWELS:
                vowel = ODIA_INDEPENDENT_VOWELS[ch]
                if dialect == "or_sbp" and ch == 'ଅ':
                    vowel = 'a'
                out.append(vowel)
                i += 1
                continue

            if ch in ODIA_CONSONANTS:
                base_c = ODIA_CONSONANTS[ch]

                # Sambalpuri retroflex softening (ଣ -> n, ଳ -> l)
                if dialect == "or_sbp":
                    if ch == 'ଣ':
                        base_c = 'n'
                    elif ch == 'ଳ':
                        base_c = 'l'

                next_ch = chars[i + 1] if (i + 1 < n) else ' '

                if next_ch == '୍':
                    out.append(base_c)
                    i += 2
                    continue
                elif next_ch in ODIA_VOWELS_MATRA:
                    out.append(base_c + ODIA_VOWELS_MATRA[next_ch])
                    i += 2
                    continue
                else:
                    is_word_end = (next_ch in punct_or_space)
                    # Sambalpuri word-final schwa deletion vs standard [ɔ] retention
                    if dialect == "or_sbp" and is_word_end and i > 0:
                        out.append(base_c)
                    else:
                        out.append(base_c + 'ɔ')
                    i += 1
                    continue

            if ch in ODIA_SIGNS:
                out.append(ODIA_SIGNS[ch])
                i += 1
                continue

            out.append(ch)
            i += 1

        ipa_str = "".join(out)

        # Berhampuri sentence-final vowel elongation
        if dialect == "or_bam":
            ipa_str = re.sub(r'([aːeiouɔ])(\s*[।?!]?)$', r'\1ː\2', ipa_str)

        # Baleswari nasalization
        if dialect == "or_bls":
            ipa_str = re.sub(r'([aːeouɔ])(\s*[।?!]?)$', r'\1̃\2', ipa_str)

        return re.sub(r'\s+', ' ', ipa_str).strip()

    def dialect_to_standard_odia(self, text: str, dialect: str) -> str:
        words = text.split()
        translated = []
        lexicon = SBP_TO_STD_LEXICON if dialect == "or_sbp" else (BAM_TO_STD_LEXICON if dialect == "or_bam" else {})

        for w in words:
            match = re.match(r"^([\w\u0B00-\u0B7F]+)([\W_]*)$", w)
            if match:
                core, punct = match.groups()
                translated.append(lexicon.get(core, core) + punct)
            else:
                translated.append(w)
        return " ".join(translated)

    def generate_dual_text_record(
        self,
        utt_id: str,
        text_orthographic: str,
        dialect_id: str = "or_std",
        speaker_id: str = "spk01",
        audio_filepath: Optional[str] = None,
        duration: Optional[float] = None
    ) -> Dict[str, Any]:
        clean = dialect_id.replace("<|", "").replace("|>", "").strip()
        return {
            "utt_id": utt_id,
            "dialect_id": f"<|{clean}|>",
            "speaker_id": speaker_id,
            "text_orthographic": text_orthographic,
            "text_standard_odia_equiv": self.dialect_to_standard_odia(text_orthographic, dialect=clean),
            "phonemes_ipa": self.text_to_ipa(text_orthographic, dialect=clean),
            "audio_filepath": audio_filepath or "",
            "duration": duration or 0.0,
        }

    def format_mlx_chat_sample(self, record: Dict[str, Any]) -> Dict[str, Any]:
        dialect = record["dialect_id"]
        ortho = record["text_orthographic"]
        ipa = record.get("phonemes_ipa", "")
        std = record.get("text_standard_odia_equiv", ortho)

        system_msg = (
            f"You are an Indic Speech-to-Text and Odia Dialect Transcription assistant. "
            f"Accurately transcribe spoken Odia and regional dialects ({dialect}) preserving phonetic cadence."
        )
        user_msg = f"Transcribe speech with dialect tag {dialect}.\nPhonetic cue: {ipa}"
        assistant_msg = ortho if dialect == "<|or_std|>" else f"{ortho}\n[Standard Odia: {std}]"

        return {
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg}
            ]
        }
