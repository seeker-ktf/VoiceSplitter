@echo off
echo Creating Python virtual environment...
python -m venv venv
if errorlevel 1 goto :error

echo.
echo Upgrading pip...
venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 goto :error

echo.
echo Installing PyTorch (this may take a while)...
venv\Scripts\pip install torch==2.11.0+cu130 torchaudio --index-url https://download.pytorch.org/whl/cu130
if errorlevel 1 goto :error

echo.
echo Installing remaining dependencies...
venv\Scripts\pip install pyannote.audio pydub pyyaml audioop-lts
if errorlevel 1 goto :error

echo.
echo Done. Run voicesplitter.bat to start.
pause
exit /b 0

:error
echo.
echo ERROR: Installation failed at the step above.
pause
exit /b 1
