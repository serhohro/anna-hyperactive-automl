@echo off
chcp 65001 >nul
title 🤖 ANNA AI v23.0 - FINAL WORKING (MLP FEHLER BEHOBEN) + CONFUSION MATRIX
+ GESTENSTEUERUNG startet externes Skript 'mouse.py'
color 0A
echo ==============================================
echo   🤖 ANNA AI v23.0 - FINAL WORKING (MLP FEHLER BEHOBEN) + CONFUSION MATRIX
+ GESTENSTEUERUNG startet externes Skript 'mouse.py')
echo ==============================================
echo.

cd /d "./"
if errorlevel 1 (
    echo ❌ Папка не найдена! Проверьте путь.
    pause
    exit /b
)

if not exist "app.py" (
    echo ❌ app.py nicht gefunden in %cd%
    pause
    exit /b
)

echo 🚀 Starte ANNA AI v23.0 - FINAL WORKING (MLP FEHLER BEHOBEN) + CONFUSION MATRIX
+ GESTENSTEUERUNG startet externes Skript 'mouse.py'
echo 📦 Verwende Python: 
where python
echo.

:: Запуск через модуль Python (надёжнее)
python -m streamlit run app.py --server.port 8501

pause