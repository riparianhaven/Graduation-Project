@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Virtual environment not found.
  echo Create:  py -m venv .venv
  echo Install: .venv\Scripts\python.exe -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
  pause
  exit /b 1
)
".venv\Scripts\python.exe" tools\run_ipynb.py %*
echo.
pause
