"""
Audio ingestion, resampling, loudness normalization, and VAD chunker.
Segments audio while preserving dialect prosody and word-final vowel contours.
"""

import os
import wave
import subprocess
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from config import PipelineConfig


class AudioPreprocessor:
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.cfg = config or PipelineConfig()
        audio_cfg = self.cfg.audio
        
        # Audio & acoustic coefficients
        self.target_sr = audio_cfg.get("sample_rate", 24000)
        self.target_lufs = audio_cfg.get("target_lufs", -20.0)      # Integrated loudness (LUFS)
        self.peak_limit_dbfs = audio_cfg.get("peak_limit_dbfs", -1.5) # Peak ceiling (dBFS)
        self.min_dur = audio_cfg.get("min_duration_sec", 1.5)       # Min segment length (s)
        self.max_dur = audio_cfg.get("max_duration_sec", 12.0)      # Max segment length (s)
        self.pad_start_ms = audio_cfg.get("pad_start_ms", 150)      # Onset pad (ms)
        self.pad_end_ms = audio_cfg.get("pad_end_ms", 250)          # Trailing pad (ms) for word-final [ɔ]
        self.silence_ms = audio_cfg.get("vad_silence_threshold_ms", 350)

        base_dir = self.cfg.config_path.parent
        self.out_dir = (base_dir / audio_cfg.get("processed_dir", "./data/processed_audio")).resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def convert_to_mono_wav(self, input_path: str, output_path: str) -> bool:
        cmd = [
            "ffmpeg", "-y", "-i", str(input_path),
            "-ac", "1",
            "-ar", str(self.target_sr),
            "-f", "wav",
            str(output_path)
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def load_wav_pcm16(self, wav_path: str) -> Tuple[np.ndarray, int]:
        with wave.open(str(wav_path), "rb") as wf:
            sr = wf.getframerate()
            n_channels = wf.getnchannels()
            raw_data = wf.readframes(wf.getnframes())
            audio_int16 = np.frombuffer(raw_data, dtype=np.int16)
            if n_channels > 1:
                audio_int16 = audio_int16.reshape(-1, n_channels).mean(axis=1).astype(np.int16)
            return audio_int16.astype(np.float32) / 32768.0, sr

    def save_wav_pcm16(self, output_path: str, audio: np.ndarray, sr: int):
        audio_clamped = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio_clamped * 32767.0).astype(np.int16)
        with wave.open(str(output_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(audio_int16.tobytes())

    def normalize_loudness(self, audio: np.ndarray, sr: int) -> np.ndarray:
        try:
            import pyloudnorm as pyln
            meter = pyln.Meter(sr)
            loudness = meter.integrated_loudness(audio)
            normed = pyln.normalize.loudness(audio, loudness, self.target_lufs) if not np.isinf(loudness) else audio
        except ImportError:
            # RMS approximation when pyloudnorm is unavailable
            rms = np.sqrt(np.mean(audio ** 2) + 1e-9)
            target_rms = 10.0 ** (self.target_lufs / 20.0)
            normed = audio * (target_rms / max(rms, 1e-5))

        # Peak limiter ceiling (-1.5 dBFS -> ~0.841)
        peak_ceiling = 10.0 ** (self.peak_limit_dbfs / 20.0)
        current_peak = np.max(np.abs(normed))
        if current_peak > peak_ceiling:
            normed = normed * (peak_ceiling / current_peak)
        return normed

    def detect_speech_segments(self, audio: np.ndarray, sr: int) -> List[Tuple[int, int]]:
        frame_len = int(sr * 0.02)  # 20ms frames
        hop_len = int(sr * 0.01)    # 10ms hop
        if len(audio) < frame_len:
            return []

        num_frames = (len(audio) - frame_len) // hop_len + 1
        energies = np.array([np.mean(audio[i * hop_len:i * hop_len + frame_len] ** 2) for i in range(num_frames)])

        threshold = max(np.percentile(energies, 15) * 2.5, np.percentile(energies, 95) * 0.05, 1e-4)
        is_speech = energies > threshold
        min_silence_frames = int((self.silence_ms / 1000.0) * sr / hop_len)

        # Merge short pause gaps
        silence_count = 0
        merged = is_speech.copy()
        for i in range(len(merged)):
            if not merged[i]:
                silence_count += 1
            else:
                if 0 < silence_count < min_silence_frames:
                    merged[i - silence_count:i] = True
                silence_count = 0

        pad_start = int((self.pad_start_ms / 1000.0) * sr)
        pad_end = int((self.pad_end_ms / 1000.0) * sr)

        segments = []
        in_segment = False
        seg_start = 0

        for i, speech in enumerate(merged):
            sample_idx = i * hop_len
            if speech and not in_segment:
                in_segment = True
                seg_start = max(0, sample_idx - pad_start)
            elif not speech and in_segment:
                in_segment = False
                seg_end = min(len(audio), sample_idx + pad_end)
                if (seg_end - seg_start) / sr >= self.min_dur:
                    segments.append((seg_start, seg_end))

        if in_segment:
            seg_end = min(len(audio), len(merged) * hop_len + pad_end)
            if (seg_end - seg_start) / sr >= self.min_dur:
                segments.append((seg_start, seg_end))

        # Split long segments exceeding max_dur
        final_segments = []
        max_samples = int(self.max_dur * sr)
        for s, e in segments:
            if (e - s) <= max_samples:
                final_segments.append((s, e))
            else:
                curr = s
                while curr < e:
                    sub_end = min(curr + max_samples, e)
                    final_segments.append((curr, sub_end))
                    curr = sub_end

        return final_segments

    def process_file(
        self,
        audio_path: str,
        dialect_id: str = "or_std",
        speaker_id: str = "spk01"
    ) -> List[Dict[str, Any]]:
        src = Path(audio_path)
        if not src.exists():
            return []

        temp_wav = self.out_dir / f"__tmp_{src.stem}.wav"
        if not self.convert_to_mono_wav(str(src), str(temp_wav)):
            return []

        try:
            audio, sr = self.load_wav_pcm16(str(temp_wav))
            segments = self.detect_speech_segments(audio, sr)

            results = []
            for idx, (s, e) in enumerate(segments):
                chunk = self.normalize_loudness(audio[s:e], sr)
                chunk_name = f"{dialect_id}_{speaker_id}_{src.stem}_{idx:04d}.wav"
                chunk_path = self.out_dir / chunk_name
                self.save_wav_pcm16(str(chunk_path), chunk, sr)

                results.append({
                    "utt_id": chunk_path.stem,
                    "audio_filepath": str(chunk_path.resolve()),
                    "duration": round(len(chunk) / sr, 3),
                    "sample_rate": sr,
                    "dialect_id": f"<|{dialect_id}|>",
                    "speaker_id": speaker_id,
                })
            return results
        finally:
            if temp_wav.exists():
                temp_wav.unlink()

    def process_directory(
        self,
        input_dir: Optional[str] = None,
        default_dialect: str = "or_std"
    ) -> List[Dict[str, Any]]:
        base_dir = self.cfg.config_path.parent
        src_dir = Path(input_dir) if input_dir else (base_dir / self.cfg.audio.get("src_dir", "./data/raw_audio")).resolve()
        
        if not src_dir.exists():
            return []

        exts = {".wav", ".m4a", ".mp3", ".flac", ".aac", ".ogg"}
        files = [p for p in src_dir.iterdir() if p.suffix.lower() in exts]

        all_chunks = []
        for f in files:
            dialect = default_dialect
            stem = f.stem.lower()
            if "sbp" in stem or "sambalpuri" in stem:
                dialect = "or_sbp"
            elif "bam" in stem or "berhampur" in stem:
                dialect = "or_bam"
            elif "bls" in stem or "baleswar" in stem:
                dialect = "or_bls"

            speaker = next((p for p in f.stem.split("_") if p.startswith("spk")), "spk01")
            all_chunks.extend(self.process_file(str(f), dialect_id=dialect, speaker_id=speaker))
        return all_chunks


if __name__ == "__main__":
    preprocessor = AudioPreprocessor()
    preprocessor.process_directory()
