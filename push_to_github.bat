@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo [1/5] git 위치 확인 중...

REM git 경로 자동 탐색
SET GIT=git
IF EXIST "C:\Program Files\Git\cmd\git.exe" SET GIT="C:\Program Files\Git\cmd\git.exe"
IF EXIST "C:\Program Files (x86)\Git\cmd\git.exe" SET GIT="C:\Program Files (x86)\Git\cmd\git.exe"

REM GitHub Desktop 내장 git 탐색
FOR /F "delims=" %%i IN ('dir /b /s "%LOCALAPPDATA%\GitHubDesktop\app-*\resources\app\git\cmd\git.exe" 2^>nul') DO SET GIT="%%i"

echo git 경로: %GIT%
echo.

echo [2/5] 기존 .git 정리 중...
IF EXIST ".git" (
    rmdir /s /q ".git" 2>nul
    IF EXIST ".git" echo 주의: .git 폴더 삭제 실패. 계속 진행합니다.
)

echo [3/5] git 초기화 및 원격 연결...
%GIT% init -b main
%GIT% remote add origin https://github.com/yejin2003/pet-eye-disease-prediction.git

echo [4/5] 파일 추가 및 커밋...
%GIT% add project/ run_webapp.bat .gitignore GITHUB_SETUP.md
%GIT% add "AI 모델 소스코드/" 2>nul
%GIT% config user.name "24013778 김예진"
%GIT% config user.email "orangered1111@gmail.com"
%GIT% commit -m "Initial commit: EfficientNet-B0 기반 반려동물 안구 질환 예측 시스템"

echo [5/5] GitHub에 push 중... (GitHub 로그인 창이 뜰 수 있어요)
%GIT% push -u origin main

echo.
echo ====================================
echo  완료! https://github.com/yejin2003/pet-eye-disease-prediction 확인하세요
echo ====================================
pause
