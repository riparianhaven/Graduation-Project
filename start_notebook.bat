@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Virtual environment not found.
  echo Create:  py -m venv .venv
  echo Install: .venv\Scripts\python.exe -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
  pause
  exit /b 1
)
echo.
echo   Thesis JupyterLab  --  Python from .venv
echo   Working directory: %CD%
echo   A browser window opens automatically.
echo   Keep this window open; close it to stop the server.
echo.
".venv\Scripts\python.exe" -m jupyter lab --notebook-dir=. --port=8888
echo.
echo JupyterLab stopped.
pause
