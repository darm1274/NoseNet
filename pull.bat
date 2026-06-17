@echo off
REM ============================================================
REM  Actualiza el repositorio NoseNet a la ultima version
REM  (hace "git pull" en la carpeta donde esta este .bat)
REM ============================================================
cd /d "%~dp0"
echo ============================================
echo   Actualizando NoseNet  (git pull)
echo ============================================
echo Carpeta: %cd%
echo.
git pull
echo.
if %errorlevel% neq 0 (
    echo [ERROR] git pull fallo. Revisa tu conexion o conflictos.
) else (
    echo [OK] Repositorio actualizado.
)
echo.
pause
