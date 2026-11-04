@echo off
rem Launches the Dance Video Sync GUI. Creates the .venv and installs the
rem requirements on first run (or if they are missing), then starts the GUI
rem without a console window.
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 goto :fail
)

".venv\Scripts\python.exe" -c "import PySide6, tqdm" >nul 2>&1
if errorlevel 1 (
    echo Installing requirements...
    ".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt
    if errorlevel 1 goto :fail
)

where ffmpeg >nul 2>&1
if errorlevel 1 (
    echo ffmpeg was not found on PATH. Install it and try again.
    goto :fail
)

start "" ".venv\Scripts\pythonw.exe" "%~dp0gui.py"
exit /b 0

:fail
echo.
echo Setup failed - see the messages above.
pause
exit /b 1
