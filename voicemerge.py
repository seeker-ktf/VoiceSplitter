"""
voicemerge.py — Merge curated speaker clips into one file per speaker.
Reads the same config as voicesplitter.py to find the output folder and format.

Usage:
    python voicemerge.py                              # uses voicesplitter.yaml in same dir
    python voicemerge.py --config my_config.yaml      # uses specified config
"""

import argparse
import sys
from pathlib import Path

import yaml
from pydub import AudioSegment


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="VoiceMerge — merge curated speaker clips")
    parser.add_argument("--config", default=str(Path(__file__).with_name("voicesplitter.yaml")),
                        help="Path to config YAML (default: voicesplitter.yaml in script dir)")
    args = parser.parse_args()

    if not Path(args.config).exists():
        print(f"ERROR: Config not found: {args.config}")
        sys.exit(1)

    cfg = load_config(args.config)
    output_folder = Path(cfg["output_folder"])
    fmt = cfg.get("output_format", "wav").lower()

    if not output_folder.exists():
        print(f"ERROR: Output folder not found: {output_folder}")
        sys.exit(1)

    # Find speaker subdirectories
    speaker_dirs = sorted([d for d in output_folder.iterdir() if d.is_dir()])
    if not speaker_dirs:
        print("No speaker folders found.")
        sys.exit(0)

    for speaker_dir in speaker_dirs:
        clips = sorted(speaker_dir.glob(f"*.{fmt}"))
        if not clips:
            print(f"  {speaker_dir.name}: no .{fmt} clips, skipping")
            continue

        print(f"  {speaker_dir.name}: merging {len(clips)} clips...")
        combined = AudioSegment.empty()
        for clip_path in clips:
            combined += AudioSegment.from_file(str(clip_path))

        merged_name = f"{speaker_dir.name}_merged.{fmt}"
        merged_path = output_folder / merged_name
        combined.export(str(merged_path), format=fmt)
        print(f"    -> {merged_path}")

    print("Done.")


if __name__ == "__main__":
    main()
