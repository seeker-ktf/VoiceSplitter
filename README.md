# VoiceSplitter

Speaker diarization tool that splits a long audio (or **video**) file into per-speaker clip subfolders.

Uses [pyannote.audio](https://github.com/pyannote/pyannote-audio) for speaker detection. Requires an NVIDIA GPU and runs on Windows.

## Setup

### 1. HuggingFace Access

You need a free [HuggingFace](https://huggingface.co) account. Visit each of these model pages and accept the terms:

- [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
- [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
- [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)

Then create an access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

### 2. Install

Create a directory (I called mine "VoiceSplitter" in my "AI" directory), then cd to the directory in a command window:

```
cd C:\AI\VoiceSplitter
git clone https://github.com/seeker-ktf/VoiceSplitter
```

From then on, you can continue in the command window or do everything in Windows Explorer.

Run `installVoiceSplitter.bat`. This creates a local Python virtual environment and installs all dependencies. I did it this way so none of your other environments python environments get messed up. The tradeoff is that you use more space:

- **venv size:** ~3.2 GB
- **HuggingFace model cache:** ~1.84 GB (downloaded on first run, stored in `~/.cache/huggingface/`)

*You will also need ffmpeg installed and in your path variable. It's probably already there but just FYI.

### 3. Configure

Edit `voicesplitter.yaml`:

```yaml
input_file: 'C:\audio\test file1.wav'
output_folder: 'C:\audio\test file1\'
output_format: wav
start_time:
end_time:
min_speakers: 2
max_speakers: 7
merge_gap: 1.5
min_clip_length: 1.0
offline_mode: false
hf_token: 'your_token_here'
```

Use single quotes for Windows paths. The `HF_TOKEN` environment variable can be used instead of putting the token in the file.

### 4. Run

From the command line:

```
python voicesplitter.py --config voicesplitter.yaml
```

The --config is the only run flag and it defaults to 'voicesplitter.yaml' in the install directory. I have found that with stuff like this it is nice to have many config files for different tasks and I still like to run from the command line, but...

I have included a .bat file in case you want to run from inside the file explorer window.  `voicesplitter.bat` 

### 5. Curate the output and run the merge

From the command line:

```
python voicemerge.py --config voicesplitter.yaml
```

Yes, the same config file. Once again, the --config is the only run flag and it defaults to 'voicesplitter.yaml' in the install directory. It will go through all of the voice directories that were created in the split and join whatever files are there into one merged file for use in whatever you are using it for.

## Config Reference

| Setting | Description |
|---|---|
| `input_file` | Audio or video file (anything ffmpeg can read) |
| `output_folder` | Where speaker subfolders are created |
| `output_format` | `wav`, `mp3`, or `flac` |
| `start_time` / `end_time` | Process a region only (HH:MM:SS or seconds). Leave blank for entire file |
| `min_speakers` / `max_speakers` | Constrain speaker count. Set both the same to force an exact number |
| `merge_gap` | Merge same-speaker segments closer than this (seconds) |
| `min_clip_length` | Discard clips shorter than this (seconds) |
| `offline_mode` | `true` skips HuggingFace server checks — use after first run |
| `hf_token` | HuggingFace access token |

## Output

```
output_folder/
  SPEAKER_00/
    clip_0001.wav
    clip_0002.wav
  SPEAKER_01/
    clip_0001.wav
  manifest.csv
```

`manifest.csv` lists every clip with speaker label, filename, start/end timestamps, and duration.

### Other useful notes.

1. If you are running ComfyUI (and you almost certainly are) it's always good to have these nodes installed:

   https://github.com/kijai/ComfyUI-MelBandRoFormer

   I think the initial intent of this was just to remove music, but it's good for other things that aren't voices too. This technology is actually amazing. It's good for wind noise and fight noise... even explosions! 

2. If you are specifically trying to get voice models for TTS  or MiniMax H3 cloning, remember that the more audio you have, the better. The MiniMax H3 documentation suggests between 10 seconds and **5 minutes** of audio for proper voice cloning. You can have too much, but it's hard.

3. Once you have all the clips in your destination folder, listen to each one and delete the ones you don't like. The VoiceSPlitter doesn't get everything right every time. This step is important.

4. While this can use just about any media, the best input you can have is a good audio version of studio interview/podcast or audio book. For non-celebs, try to use the best recording setup you can. Anything like hiss or echoes will show up on the clone. (Although you can always try cleaning it up with the MelBanRoFormer.)

## Requirements

- Windows
- NVIDIA GPU with CUDA
- Python 3.11+
- ffmpeg on PATH
