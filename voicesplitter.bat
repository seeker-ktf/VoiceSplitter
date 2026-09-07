@echo off
call venv\Scripts\activate.bat
python voicesplitter.py --config voicesplitter.yaml
pause
