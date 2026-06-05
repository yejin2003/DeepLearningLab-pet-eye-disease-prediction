# 제출 가이드

교수님 공지 기준에 맞춰 `보고서`, `발표자료`, `소스코드`, `사용 데이터`를 제출합니다. 본 프로젝트는 1인 프로젝트이므로 동료평가 항목은 적용 대상이 아니며, 공지에 따라 발표자료 평가로 대체되는 항목을 자료 자체의 완성도로 대응합니다.

## 제출 파일 구성

최종 제출 폴더는 `deliverables/`입니다.

| 제출 항목 | 제출 파일 |
|---|---|
| 보고서 | `24013778김예진_기말프로젝트_보고서.docx` |
| 발표자료 | `24013778김예진_기말프로젝트_발표자료.pptx` |
| 소스코드 | `source_code.zip` |
| 사용 데이터 | `used_data_submission.zip` |
| 제출물 설명 | `README_FIRST.md` |

## 사용 데이터 제출 방식

원본 AIHub 데이터는 용량이 매우 크므로 교수님 공지의 “오픈데이터셋이면 관련 URL 제출” 기준을 따릅니다.

- 데이터명: AIHub 반려동물 안구 질환 데이터
- URL: https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=562
- 실제 사용 범위: `153.반려동물 안구질환 데이터/01.데이터/2.Validation/원천데이터`
- 학습용 CSV: `project/datasets/*.csv`
- 데이터 설명: `project/USED_DATA_README.md`

AIHub FAQ에 따르면 다운로드한 원본 jpg/json 데이터의 재배포는 제한될 수 있으므로, 대용량 원본 이미지 전체를 임의로 압축해 공개 제출하기보다 데이터셋 URL과 사용 내역을 제출하는 방식이 안전합니다. 교수님이 별도 원본 제출을 요구하면 AIHub 이용정책을 함께 확인한 뒤 대용량 메일/드라이브 방식으로 전달하면 됩니다.

## 실행 매뉴얼

1. Python 환경을 준비합니다.

```powershell
pip install -r project\requirements.txt
```

2. 웹 서버를 실행합니다.

```powershell
C:\Users\orang\pet_eye_env\.venv\Scripts\python.exe project\webapp\manage.py runserver 127.0.0.1:8000
```

3. 브라우저에서 접속합니다.

```text
http://127.0.0.1:8000/
```

4. 반려동물 눈 이미지를 업로드합니다.

5. 업로드 직후 사각형 미리보기 영역에 이미지가 표시되는지 확인합니다.

6. `예측 실행`을 누릅니다.

7. 결과 화면에서 다음 항목을 확인합니다.

- 이미지 품질 점검 결과
- 정상/질환 의심 상태
- 상위 3개 질환 후보
- 최고 확률 질환의 설명
- 예방법 및 관리 방법

## API 테스트

```powershell
curl -X POST http://127.0.0.1:8000/api/predict/ -F "image=@sample.jpg"
```

응답 예시는 다음 구조입니다.

```json
{
  "status": "ok",
  "result": {
    "top_disease": "궤양성 각막질환",
    "risk_label": "주의",
    "diseases": [
      {"name": "궤양성 각막질환", "probability": 39.17},
      {"name": "안검염", "probability": 39.12},
      {"name": "결막염", "probability": 19.48}
    ]
  }
}
```

## 평가 때 강조할 내용

- 계획서의 흐름인 이미지 업로드, 품질 점검, 전처리, EfficientNet/ResNet 계열 전이학습, softmax 확률, 질환 설명 제공을 실제 웹 서비스로 구현했습니다.
- AIHub 제공 모델 코드를 그대로 복사하지 않고, 핵심 구조를 PyTorch와 Django 환경에 맞게 재구현했습니다.
- 단일 질환만 예측하는 수준이 아니라 10개 안구 질환 후보를 지원합니다.
- 질환별 이진 모델의 한계를 확인하고, 최종 웹 서비스는 다중분류 모델로 질환 간 순위를 비교하도록 개선했습니다.
- 제출자가 아닌 다른 사람이 따라 할 수 있도록 실행 명령, 데이터 생성 명령, API 사용법, 제출물 구성을 문서화했습니다.

## GitHub 가산점 준비

GitHub 공개 제출을 할 경우 루트의 `.gitignore`를 유지하고, `153.반려동물 안구질환 데이터/`, `.venv/`, `project/outputs/*.pt`, `project/webapp/media/uploads/`는 업로드하지 않는 것을 권장합니다. 대용량 모델 파일은 필요 시 GitHub Release 또는 별도 드라이브 링크로 안내합니다.
