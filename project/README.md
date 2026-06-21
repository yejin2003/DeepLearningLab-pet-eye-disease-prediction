# 🐾 반려동물 안구 질환 의심 예측 시스템

> 딥러닝실습 기말 프로젝트 | 24013778 김예진 | 세종대학교

AIHub `반려동물 안구 질환 데이터`와 **EfficientNet-B0 전이학습**을 활용해, 반려동물 눈 이미지를 업로드하면 안구 질환 의심 후보 상위 3개와 가장 가능성 높은 질환의 설명·예방법을 제공하는 **Django 웹 애플리케이션**입니다.

| 항목 | 내용 |
|------|------|
| 모델 | EfficientNet-B0 (ImageNet 전이학습) |
| 프레임워크 | PyTorch, Django |
| 분류 | 정상 + 10개 질환 (11클래스 다중분류) |
| 성능 | 질환별 이진모델 F1 Score 0.983 ~ 1.000 |
| 데이터 | AIHub 반려동물 안구 질환 데이터 (25,113개 샘플 사용) |

## 프로젝트 요약

- 주제: 딥러닝 기반 반려동물 안구 질환 의심 예측 시스템
- 데이터: AIHub 반려동물 안구 질환 데이터
- 모델: ImageNet 사전학습 EfficientNet-B0 전이학습
- 프레임워크: PyTorch, Django
- 출력: 이미지 품질 점검, 정상/질환 의심 판단, 상위 3개 질환 후보, 질환 설명, 예방법 및 병원 방문 안내

## 지원 질환

웹 서비스는 정상 클래스를 포함한 다중분류 모델을 우선 사용하고, 질환 후보는 다음 10개 질환 중 상위 3개만 표시합니다.

1. 결막염
2. 안검염
3. 유루증
4. 핵경화
5. 백내장
6. 궤양성 각막질환
7. 비궤양성 각막질환
8. 색소침착성 각막염
9. 안검내반증
10. 안검종양

## AIHub 제공 소스코드 반영

`AI 모델 소스코드` 폴더의 TensorFlow/Keras 예제는 EfficientNetB0 또는 ResNet 계열 사전학습 모델, 224 x 224 입력, 이미지 증강, softmax 분류, checkpoint 저장 구조를 사용합니다.

본 프로젝트는 같은 설계 원리를 PyTorch와 Django 환경에 맞게 재구현했습니다.

- Backbone: `torchvision.models.efficientnet_b0`
- 입력 크기: 224 x 224 RGB
- 증강: 좌우 반전, 회전, 밝기/대비 조정
- 손실 함수: `CrossEntropyLoss`
- Optimizer: `AdamW`
- 저장 기준: validation F1 또는 validation loss 기준 best model

## 실행 방법

### 0. 배포된 웹 서비스로 바로 사용하기 (권장)

설치 없이 바로 사용해볼 수 있습니다.

🔗 **https://yejin2003-pet-eye-disease-prediction.hf.space**

### 1. 환경 설치

```bash
# 가상환경 생성 및 패키지 설치
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

pip install -r project/requirements.txt
```

### 2. 모델 가중치 준비

모델 가중치(`.pt`)는 용량 문제로 Git에 포함되지 않습니다.  
직접 학습하려면 아래 **학습 방법** 섹션을 참고하거나, 별도로 제공된 가중치 파일을 `project/outputs/`에 배치하세요.

### 3. 웹 서버 실행

```bash
python project/webapp/manage.py runserver 127.0.0.1:8000
```

또는 Windows에서 루트의 `run_webapp.bat`를 더블클릭합니다.

브라우저 접속: **http://127.0.0.1:8000/**

> 로컬 설치 없이 바로 확인하려면 위 [배포된 웹 서비스](https://yejin2003-pet-eye-disease-prediction.hf.space)를 이용하세요.

## API 사용 방법

```text
POST /api/predict/
Content-Type: multipart/form-data
image: JPG, JPEG, PNG 이미지 파일
```

응답에는 이미지 품질 점검 결과, 정상/질환 의심 상태, 상위 3개 질환 후보, 가장 높은 질환의 설명 및 관리 방법이 포함됩니다.

## 학습 방법

데이터는 [AIHub 반려동물 안구 질환 데이터](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=562)에서 다운로드 후 `153.반려동물 안구질환 데이터/` 폴더에 배치합니다.

```bash
# 1. 다중분류 CSV 생성
python project/prepare_multiclass_csv.py \
    --output project/datasets/eye_multiclass_dataset.csv

# 2. 다중분류 모델 학습 (EfficientNet-B0, 11클래스)
python project/train_multiclass_pytorch.py \
    --csv project/datasets/eye_multiclass_dataset.csv \
    --epochs 6 --batch-size 32 --num-workers 0

# 3. 질환별 이진 모델 학습 예시 (결막염)
python project/train_disease_pytorch.py \
    --disease conjunctivitis \
    --csv project/datasets/conjunctivitis_dataset.csv \
    --epochs 5 --batch-size 16 --num-workers 0

# 4. 단일 이미지 추론
python project/predict_multiclass_eye.py --image "path/to/eye_image.jpg"
```

## 주요 파일

- `project/webapp/`: Django 웹 애플리케이션
- `project/prepare_multiclass_csv.py`: 정상+10개 질환 다중분류 CSV 생성
- `project/train_multiclass_pytorch.py`: EfficientNet-B0 다중분류 학습 코드
- `project/predict_multiclass_eye.py`: 다중분류 추론 코드
- `project/disease_definitions.py`: 질환 설명, 예방법, 병원 방문 안내 문구
- `project/image_quality.py`: 이미지 품질 점검 코드
- `project/outputs/best_eye_multiclass_efficientnet_b0.pt`: 웹 서비스 우선 사용 모델
- `project/evaluation_summary.csv`: 질환별 이진 모델 평가 요약
- `project/FINAL_REPORT.md`: 최종 보고서 원본
- `project/RUBRIC_RESPONSE.md`: 평가 기준 대응표
- `project/USED_DATA_README.md`: 사용 데이터 제출 안내
- `project/SUBMISSION_GUIDE.md`: 제출 패키지와 실행 매뉴얼

## 제출 산출물

최종 제출용 파일은 루트의 `deliverables/` 폴더에 정리됩니다.

- 보고서: `24013778김예진_기말프로젝트_보고서.docx`
- 발표자료: `24013778김예진_기말프로젝트_발표자료.pptx`
- 소스코드: `source_code.zip`
- 사용 데이터 안내: `used_data_submission.zip`

AIHub 원본 데이터는 용량이 크므로 원본 전체를 중복 압축하지 않고, `USED_DATA_README.md`에 데이터셋 URL, 사용 범위, 생성 CSV, 학습에 사용한 클래스 구성을 명시합니다.
