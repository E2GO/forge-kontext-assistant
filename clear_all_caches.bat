@echo off
echo Clearing all caches that might cause h11 errors...
echo.

cd /d "J:\Stability Matrix\Packages\Stable Diffusion WebUI Forge"

echo 1. Clearing Python __pycache__ directories...
for /d /r . %%d in (__pycache__) do @if exist "%%d" (
    echo Removing %%d
    rd /s /q "%%d"
)

echo.
echo 2. Clearing pip cache...
call venv\Scripts\python.exe -m pip cache purge

echo.
echo 3. Clearing Gradio temp files...
if exist "%TEMP%\gradio" (
    echo Removing %TEMP%\gradio
    rd /s /q "%TEMP%\gradio"
)

echo.
echo 4. Clearing temporary uploaded files...
for /d %%d in ("%TEMP%\tmp*") do @if exist "%%d" (
    echo Removing %%d
    rd /s /q "%%d" 2>nul
)

echo.
echo 5. Clearing browser cache (recommended to do manually)...
echo Please clear your browser cache for localhost:7860
echo - Chrome: Ctrl+Shift+Delete
echo - Firefox: Ctrl+Shift+Delete
echo - Edge: Ctrl+Shift+Delete

echo.
echo 6. Reinstalling h11 clean...
call venv\Scripts\python.exe -m pip uninstall -y h11
call venv\Scripts\python.exe -m pip install h11==0.12.0 --no-cache-dir

echo.
echo Done! Please:
echo 1. Close WebUI completely
echo 2. Clear browser cache for localhost:7860
echo 3. Restart WebUI
echo.
pause