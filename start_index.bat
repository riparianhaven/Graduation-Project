@echo off
setlocal
chcp 65001 >nul 2>&1
cd /d "%~dp0"

set "URL=http://127.0.0.1:8765/index.html"

where py >nul 2>&1
if errorlevel 1 (
  echo Python launcher "py" was not found.
  echo Please run this manually:  py tools\serve_index.py 8765
  pause
  exit /b 1
)

rem Refresh the offline HTML copies so double-clicking index.html also works.
py tools\build_static.py >nul 2>&1

rem If the server is already running, just open the browser.
py -c "import socket,sys; s=socket.socket(); s.settimeout(0.5); sys.exit(0 if s.connect_ex(('127.0.0.1',8765))==0 else 1)" >nul 2>&1
if errorlevel 1 (
  start "BiShe Index Server" /min py tools\serve_index.py 8765
  timeout /t 2 /nobreak >nul
)

start "" "%URL%"
echo.
echo   Index : %URL%
echo   Stop  : close the minimized server window (or press Ctrl+C in it)
echo.
timeout /t 5 /nobreak >nul
endlocal
