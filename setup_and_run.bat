@echo off
REM ================================================================
REM  PDF Violation Agent — vLLM + Qwen3 — Windows Setup
REM  AMD CPU works. AMD GPU on Windows needs WSL2 (see README).
REM ================================================================
setlocal enabledelayedexpansion

echo.
echo  ╔════════════════════════════════════════════════════════╗
echo  ║   PDF Violation Agent  ^|  vLLM + Qwen3  ^|  Windows   ║
echo  ╚════════════════════════════════════════════════════════╝
echo.

REM ── Python check ─────────────────────────────────────────────────
echo [1/5] Checking Python...
python --version >NUL 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python not found.
    echo Download Python 3.10+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH"
    pause & exit /b 1
)
python --version
echo OK: Python found

REM ── Virtual environment ───────────────────────────────────────────
echo [2/5] Setting up virtual environment...
IF NOT EXIST venv ( python -m venv venv )
call venv\Scripts\activate.bat
pip install --upgrade pip -q
echo OK: venv ready

REM ── vLLM install ─────────────────────────────────────────────────
echo [3/5] Installing vLLM (CPU mode for Windows)...
echo NOTE: AMD GPU on Windows requires WSL2. See README for details.
pip install vllm -q
echo OK: vLLM installed

REM ── App dependencies ──────────────────────────────────────────────
echo [4/5] Installing app dependencies...
pip install -r requirements.txt -q
echo OK: Dependencies installed

REM ── Folders ──────────────────────────────────────────────────────
echo [5/5] Creating folders...
if not exist uploads   mkdir uploads
if not exist reports   mkdir reports
if not exist vector_db mkdir vector_db
echo OK: Folders created

REM ── Model choice ─────────────────────────────────────────────────
echo.
echo Choose Qwen3 model:
echo   1) Qwen/Qwen3-1.7B  ~1 GB   fastest
echo   2) Qwen/Qwen3-4B    ~2.5 GB
echo   3) Qwen/Qwen3-8B    ~5 GB   recommended  [default]
echo.
set /p CHOICE="Enter 1-3 [default 3]: "
IF "%CHOICE%"=="1" SET MODEL=Qwen/Qwen3-1.7B
IF "%CHOICE%"=="2" SET MODEL=Qwen/Qwen3-4B
IF NOT DEFINED MODEL  SET MODEL=Qwen/Qwen3-8B

echo Using model: %MODEL%
echo VLLM_MODEL=%MODEL% > .env

REM ── Write helper batch files ──────────────────────────────────────
echo @echo off > start_vllm.bat
echo call venv\Scripts\activate.bat >> start_vllm.bat
echo echo Starting vLLM server with %MODEL% ... >> start_vllm.bat
echo echo First run downloads the model — please wait >> start_vllm.bat
echo python -m vllm.entrypoints.openai.api_server --model %MODEL% --served-model-name %MODEL% --host 0.0.0.0 --port 8000 --max-model-len 8192 --dtype auto --trust-remote-code >> start_vllm.bat

echo @echo off > start_app.bat
echo call venv\Scripts\activate.bat >> start_app.bat
echo echo Starting web app ... open http://localhost:8080 >> start_app.bat
echo uvicorn app:app --host 0.0.0.0 --port 8080 --reload >> start_app.bat

echo.
echo  ╔════════════════════════════════════════════════════════╗
echo  ║               Setup Complete!                         ║
echo  ║                                                       ║
echo  ║  STEP 1: Double-click  start_vllm.bat                 ║
echo  ║          (wait for "Application startup complete")    ║
echo  ║                                                       ║
echo  ║  STEP 2: Double-click  start_app.bat                  ║
echo  ║                                                       ║
echo  ║  STEP 3: Open http://localhost:8080                   ║
echo  ╚════════════════════════════════════════════════════════╝
echo.
pause
