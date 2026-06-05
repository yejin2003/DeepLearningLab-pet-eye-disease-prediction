# GitHub 업로드 가이드

## 1. GitHub 레포 생성

1. https://github.com 로그인
2. 우상단 **+** → **New repository**
3. 설정:
   - Repository name: `pet-eye-disease-prediction`
   - Description: `딥러닝 기반 반려동물 안구 질환 의심 예측 시스템 | EfficientNet-B0 + Django`
   - **Public** 선택 (가산점 조건: 공개 필수)
   - README, .gitignore, license **체크 해제** (이미 있음)
4. **Create repository** 클릭

---

## 2. 로컬 PC에서 git 초기화 및 첫 push

PowerShell 또는 Git Bash에서 아래 명령을 순서대로 실행합니다.

```powershell
# 프로젝트 폴더로 이동
cd "C:\Users\orang\OneDrive\Desktop\딥러닝 실습_반려동물 안구질환"

# git 초기화
git init -b main

# 사용자 정보 설정
git config user.name "24013778 김예진"
git config user.email "orangered1111@gmail.com"

# 파일 스테이징 (모델 가중치, 원본 데이터 제외)
git add .gitignore run_webapp.bat project/ "AI 모델 소스코드/"

# 첫 번째 커밋
git commit -m "Initial commit: EfficientNet-B0 기반 반려동물 안구 질환 예측 시스템

- AIHub 반려동물 안구 질환 데이터 기반 11클래스 다중분류 모델
- PyTorch EfficientNet-B0 전이학습, CrossEntropyLoss, AdamW
- 질환별 이진 분류 모델(10개) + 다중분류 모델(1개) 구성
- Django 웹 앱: 이미지 업로드, 품질 점검, 상위 3개 질환 예측"

# 두 번째 커밋 (코드 정비)
git add run_webapp.bat project/requirements.txt .gitignore project/README.md
git commit -m "Refactor: 하드코딩 경로 제거 및 실행 스크립트 이식성 개선

- run_webapp.bat: 하드코딩된 C:\Users\orang 경로 제거, .venv 자동 탐색
- requirements.txt: 패키지 버전 범위 명시
- .gitignore: 패턴 정비
- README.md: 설치~실행 순서 명확화"

# GitHub 원격 연결 (YOUR_USERNAME을 본인 GitHub 아이디로 변경)
git remote add origin https://github.com/YOUR_USERNAME/pet-eye-disease-prediction.git

# push
git push -u origin main
```

---

## 3. 가산점을 위한 이후 커밋 계획

교수님이 보시는 기준은 **"지속적이고 공개적인 프로젝트 관리"** 입니다.
아래처럼 며칠에 걸쳐 작은 개선을 커밋하면 됩니다.

| 시기 | 커밋 내용 예시 |
|------|---------------|
| 오늘 | Initial commit + Refactor (위에서 완료) |
| 2~3일 후 | README에 성능 표 추가 / 스크린샷 추가 |
| 기말 발표 전 | 웹 UI 소개 이미지 추가, 최종 보고서 링크 |

### 이후 커밋 예시 명령어

```powershell
# 파일 수정 후
git add 수정된파일
git commit -m "Docs: README에 웹 UI 스크린샷 및 성능 결과 추가"
git push
```

---

## 4. 모델 가중치(.pt) 공유 방법 (선택)

.pt 파일은 각 16MB로 GitHub에 올리기 어렵습니다. 공유하려면:

- **Google Drive** 에 업로드 후 README에 링크 추가
- 또는 README에 "직접 학습 방법" 안내로 대체 (현재 README에 이미 포함됨)
