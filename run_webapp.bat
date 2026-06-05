@echo off
cd /d "%~dp0"

REM 가상환경 python 자동 탐색 (.venv 우선, 없으면 시스템 python)
IF EXIST ".venv\Scripts\python.exe" (
    SET PYTHON=".venv\Scripts\python.exe"
) ELSE IF EXIST "project\.venv\Scripts\python.exe" (
    SET PYTHON="project\.venv\Scripts\python.exe"
) ELSE (
    SET PYTHON=python
)

echo [pet-eye] Using python: %PYTHON%
%PYTHON% project\webapp\manage.py runserver 127.0.0.1:8000
