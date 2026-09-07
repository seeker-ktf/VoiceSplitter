"""
voicesplitter.py — Speaker diarization tool.
Splits a long audio (or video) file into per-speaker clip subfolders.

Usage:
    python voicesplitter.py                              # uses voicesplitter.yaml in same dir
    python voicesplitter.py --config my_config.yaml      # uses specified config

Requirements:
    pip install pyannote.audio pydub pyyaml torch torchaudio
    ffmpeg must be on PATH.
    A HuggingFace token with access to pyannote models.
"""

import argparse
import os
import sys
import time
import subprocess
import tempfile
import yaml
from pathlib import Path

import torch
from pydub import AudioSegment
from pyannote.audio import Pipeline


# ── Config ───────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_token(cfg: dict) -> str:
    token = cfg.get("hf_token") or os.environ.get("HF_TOKEN", "")
    if not token:
        print("ERROR: No HuggingFace token. Set hf_token in config or HF_TOKEN env var.")
        sys.exit(1)
    return token


def parse_time(val) -> float | None:
    """Parse a time value to seconds.  Accepts HH:MM:SS, MM:SS, or bare seconds."""
    if val is None or val == "":
        return None
    if isinstance(val, (int, float)):
        return float(val)
    parts = str(val).split(":")
    parts = [float(p) for p in parts]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0]


# ── Audio loading ────────────────────────────────────────────────────────

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".aac", ".wma", ".m4a"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".ts", ".flv"}


def extract_audio_from_video(video_path: str) -> str:
    """Use ffmpeg to pull the audio track from a video file into a temp wav."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        tmp.name,
    ]
    print(f"Extracting audio from video: {Path(video_path).name}")
    subprocess.run(cmd, check=True, capture_output=True)
    return tmp.name


def load_audio(cfg: dict) -> tuple[str, bool]:
    """Return (path_to_wav_for_pyannote, is_temp_file)."""
    input_file = cfg["input_file"]
    ext = Path(input_file).suffix.lower()

    if ext in VIDEO_EXTENSIONS:
        return extract_audio_from_video(input_file), True

    if ext in AUDIO_EXTENSIONS:
        return input_file, False

    # Unknown extension — try to let ffmpeg handle it as video/audio
    print(f"Unknown extension '{ext}', attempting ffmpeg extraction...")
    try:
        return extract_audio_from_video(input_file), True
    except subprocess.CalledProcessError:
        print(f"ERROR: ffmpeg could not process '{input_file}'.")
        sys.exit(1)


# ── Trimming ─────────────────────────────────────────────────────────────

def trim_audio_if_needed(audio_path: str, cfg: dict) -> tuple[str, bool]:
    """Trim to start_time/end_time if specified.  Returns (path, is_temp)."""
    start = parse_time(cfg.get("start_time"))
    end = parse_time(cfg.get("end_time"))
    if start is None and end is None:
        return audio_path, False

    args = ["ffmpeg", "-y", "-i", audio_path]
    if start is not None:
        args += ["-ss", str(start)]
    if end is not None:
        args += ["-to", str(end)]

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    args += ["-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", tmp.name]
    print(f"Trimming audio: start={start}, end={end}")
    subprocess.run(args, check=True, capture_output=True)
    return tmp.name, True


# ── Diarization ──────────────────────────────────────────────────────────

def apply_offline_mode(cfg: dict):
    """Set HF_HUB_OFFLINE before any HuggingFace code touches the network."""
    if cfg.get("offline_mode", False):
        os.environ["HF_HUB_OFFLINE"] = "1"
        print("Offline mode: using cached models only.")


def run_diarization(audio_path: str, cfg: dict, token: str):
    print("Loading pyannote diarization pipeline (this may download models on first run)...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=token,
    )
    pipeline.to(torch.device("cuda"))

    # Build speaker params
    params = {}
    min_sp = cfg.get("min_speakers")
    max_sp = cfg.get("max_speakers")
    if min_sp is not None and max_sp is not None and min_sp == max_sp:
        params["num_speakers"] = int(min_sp)
    else:
        if min_sp is not None:
            params["min_speakers"] = int(min_sp)
        if max_sp is not None:
            params["max_speakers"] = int(max_sp)

    print(f"Running diarization (speaker params: {params or 'auto'})...")
    t0 = time.time()
    diarization = pipeline(audio_path, **params)
    elapsed = time.time() - t0
    print(f"Diarization complete in {elapsed:.1f}s")
    return diarization


# ── Merge segments ───────────────────────────────────────────────────────

def merge_segments(diarization, merge_gap: float) -> list[tuple[float, float, str]]:
    """Collect segments and merge consecutive same-speaker segments within merge_gap."""
    raw = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        raw.append((turn.start, turn.end, speaker))

    if not raw:
        return raw

    merged = [raw[0]]
    for start, end, speaker in raw[1:]:
        prev_start, prev_end, prev_speaker = merged[-1]
        if speaker == prev_speaker and (start - prev_end) <= merge_gap:
            merged[-1] = (prev_start, end, speaker)
        else:
            merged.append((start, end, speaker))

    return merged


# ── Slicing and export ───────────────────────────────────────────────────

def export_clips(segments, cfg: dict, source_audio_path: str):
    output_folder = Path(cfg["output_folder"])
    fmt = cfg.get("output_format", "wav").lower()

    print(f"Loading source audio for slicing...")
    ext = Path(source_audio_path).suffix.lower()
    audio = AudioSegment.from_file(source_audio_path)

    # Group segments by speaker
    by_speaker: dict[str, list[tuple[float, float]]] = {}
    for start, end, speaker in segments:
        by_speaker.setdefault(speaker, []).append((start, end))

    total_clips = 0
    for speaker, clips in sorted(by_speaker.items()):
        speaker_dir = output_folder / speaker
        speaker_dir.mkdir(parents=True, exist_ok=True)

        for i, (start, end) in enumerate(clips, 1):
            start_ms = int(start * 1000)
            end_ms = int(end * 1000)
            clip = audio[start_ms:end_ms]

            filename = f"clip_{i:04d}.{fmt}"
            clip_path = speaker_dir / filename
            clip.export(str(clip_path), format=fmt)
            total_clips += 1

        print(f"  {speaker}: {len(clips)} clips -> {speaker_dir}")

    print(f"Exported {total_clips} clips across {len(by_speaker)} speakers.")


# ── Manifest ─────────────────────────────────────────────────────────────

def write_manifest(segments, cfg: dict):
    import csv
    output_folder = Path(cfg["output_folder"])
    manifest_path = output_folder / "manifest.csv"
    fmt = cfg.get("output_format", "wav").lower()

    # Number clips per speaker
    counters: dict[str, int] = {}
    rows = []
    for start, end, speaker in segments:
        counters[speaker] = counters.get(speaker, 0) + 1
        idx = counters[speaker]
        rows.append({
            "speaker": speaker,
            "clip": f"clip_{idx:04d}.{fmt}",
            "start": f"{start:.3f}",
            "end": f"{end:.3f}",
            "duration": f"{end - start:.3f}",
        })

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["speaker", "clip", "start", "end", "duration"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Manifest written: {manifest_path}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    # Find config
    parser = argparse.ArgumentParser(description="VoiceSplitter — speaker diarization tool")
    parser.add_argument("--config", default=str(Path(__file__).with_name("voicesplitter.yaml")),
                        help="Path to config YAML (default: voicesplitter.yaml in script dir)")
    args = parser.parse_args()
    config_path = args.config

    if not Path(config_path).exists():
        print(f"ERROR: Config not found: {config_path}")
        sys.exit(1)

    cfg = load_config(str(config_path))
    apply_offline_mode(cfg)
    token = resolve_token(cfg)

    # Validate input
    input_file = cfg.get("input_file", "")
    if not input_file or not Path(input_file).exists():
        print(f"ERROR: Input file not found: {input_file}")
        sys.exit(1)

    # Load audio (extract from video if needed)
    audio_path, audio_is_temp = load_audio(cfg)

    # Trim if start_time / end_time specified
    trimmed_path, trim_is_temp = trim_audio_if_needed(audio_path, cfg)

    try:
        # Diarize
        diarization = run_diarization(trimmed_path, cfg, token)

        # Unwrap DiarizeOutput dataclass to get the Annotation
        if hasattr(diarization, 'speaker_diarization'):
            diarization = diarization.speaker_diarization

        # Merge close segments
        merge_gap = float(cfg.get("merge_gap", 1.5))
        segments = merge_segments(diarization, merge_gap)

        # Filter out short clips
        min_clip = float(cfg.get("min_clip_length", 0))
        if min_clip > 0:
            before = len(segments)
            segments = [(s, e, sp) for s, e, sp in segments if (e - s) >= min_clip]
            dropped = before - len(segments)
            if dropped:
                print(f"Dropped {dropped} clips shorter than {min_clip}s")

        if not segments:
            print("No speech segments found.")
            sys.exit(0)

        print(f"Found {len(segments)} segments across "
              f"{len(set(s[2] for s in segments))} speakers (after merging).")

        # Use the original (non-trimmed-for-diarization) audio for slicing
        # so clip quality matches the source.  But we need the trimmed version
        # if start/end were specified, since timestamps are relative to it.
        slice_source = trimmed_path if (parse_time(cfg.get("start_time")) is not None
                                        or parse_time(cfg.get("end_time")) is not None) else audio_path

        # Export
        export_clips(segments, cfg, slice_source)
        write_manifest(segments, cfg)

    finally:
        # Clean up temp files
        if audio_is_temp and Path(audio_path).exists():
            os.unlink(audio_path)
        if trim_is_temp and Path(trimmed_path).exists():
            os.unlink(trimmed_path)

    print("Done.")


if __name__ == "__main__":
    main()
