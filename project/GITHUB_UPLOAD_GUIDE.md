# GitHub 공개 관리 가이드

가산점 항목인 `프로젝트 공개(Github 관리)`를 받기 위해서는 코드와 문서는 공개하되, 대용량 원본 데이터와 학습 모델 가중치는 제외하는 방식이 안전합니다.

## 업로드 권장 항목

- `project/*.py`
- `project/webapp/`
- `project/README.md`
- `project/FINAL_REPORT.md`
- `project/MODEL_CARD.md`
- `project/RUBRIC_RESPONSE.md`
- `project/USED_DATA_README.md`
- `project/SUBMISSION_GUIDE.md`
- `project/requirements.txt`
- `run_webapp.bat`

## 업로드 제외 권장 항목

- `153.반려동물 안구질환 데이터/`
- `.venv/`
- `project/outputs/*.pt`
- `project/webapp/media/uploads/`
- `project/webapp/db.sqlite3`
- `__pycache__/`

## GitHub README에 적을 핵심 문구

```text
본 프로젝트는 AIHub 반려동물 안구 질환 데이터를 활용한 학습 프로젝트입니다.
원본 데이터는 용량과 이용정책 문제로 저장소에 포함하지 않았으며,
AIHub 데이터셋 페이지에서 별도로 신청/다운로드해야 합니다.
```

## 공개 순서

```powershell
git init
git add .
git commit -m "Add pet eye disease prediction final project"
git branch -M main
git remote add origin <GitHub repository URL>
git push -u origin main
```

업로드 전에 `.gitignore`가 적용되었는지 `git status`로 확인합니다.
