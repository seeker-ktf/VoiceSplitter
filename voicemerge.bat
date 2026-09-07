@echo off
call venv\Scripts\activate.bat
python voicemerge.py --config voicesplitter.yaml
pause
