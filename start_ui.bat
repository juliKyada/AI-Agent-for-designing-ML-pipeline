@echo off
REM MetaFlow Web UI Launcher for Windows

echo ========================================
echo    MetaFlow - AI ML Pipeline Designer
echo ========================================
echo.
echo Starting web interface...
echo.
echo Once started, open your browser to:
echo http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

streamlit run app.py
