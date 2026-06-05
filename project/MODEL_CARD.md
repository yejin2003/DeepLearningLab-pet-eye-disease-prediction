# Model Card

## 모델 개요

- 목적: 반려동물 눈 이미지 기반 안구 질환 의심 후보 예측
- Backbone: EfficientNet-B0
- 학습 방식: ImageNet 사전학습 가중치를 활용한 transfer learning
- 구현: PyTorch
- 입력: 224 x 224 RGB image
- 웹 추론 우선 모델: 정상 + 10개 질환 다중분류 모델
- 보조 모델: 질환별 이진 분류 모델

## 지원 클래스

다중분류 모델은 `정상` 클래스를 포함해 총 11개 클래스를 분류합니다.

| 번호 | 클래스 |
|---:|---|
| 0 | 정상 |
| 1 | 결막염 |
| 2 | 안검염 |
| 3 | 유루증 |
| 4 | 핵경화 |
| 5 | 백내장 |
| 6 | 궤양성 각막질환 |
| 7 | 비궤양성 각막질환 |
| 8 | 색소침착성 각막염 |
| 9 | 안검내반증 |
| 10 | 안검종양 |

## AIHub 제공 코드 반영

AIHub 제공 `AI 모델 소스코드`는 TensorFlow/Keras 기반 EfficientNetB0/ResNet 전이학습 예제입니다. 본 프로젝트는 해당 구조의 핵심 원리를 PyTorch로 재구현했습니다.

| AIHub 제공 코드 | 본 프로젝트 구현 |
|---|---|
| TensorFlow/Keras | PyTorch |
| EfficientNetB0/ResNet pretrained model | torchvision EfficientNet-B0 pretrained model |
| ImageDataGenerator 증강 | torchvision transforms 증강 |
| 224 x 224 입력 | 224 x 224 입력 |
| softmax 분류 | softmax 기반 다중분류/이진분류 |
| checkpoint 저장 | best `.pt` 모델 저장 |

## 추론 방식

초기 구현은 10개 질환별 이진 분류 모델의 positive 확률을 비교해 상위 질환을 고르는 방식이었습니다. 이 방식은 한 이미지에서 여러 이진 모델이 동시에 높은 확률을 낼 수 있어, 질환 간 순위가 실제 폴더 라벨과 어긋나는 문제가 있었습니다.

현재 웹 서비스는 정상 + 10개 질환을 한 번에 비교하는 다중분류 모델을 최종 질환 순위 산정에 우선 사용합니다. 질환별 이진 모델은 성능 비교와 보고서 근거 자료로 유지하되, 웹 화면의 상위 3개 질환은 다중분류 모델 결과를 사용합니다.

또한 AIHub 폴더에서 잘라낸 `crop_` 이미지가 업로드된 경우, 원본 파일명이 일치하면 학습 데이터 분포에 가까운 원본 이미지 경로를 찾아 추론에 사용하도록 보정했습니다. 이는 사용자가 데이터셋 내부 이미지를 직접 시험할 때 라벨과 예측의 일관성을 높이기 위한 처리입니다.

## 성능 요약

질환별 이진 분류 모델의 상세 성능은 `project/evaluation_summary.csv`에 정리했습니다. 모든 지원 질환에서 계획서 목표였던 F1 Score 0.8 이상을 달성했습니다.

다중분류 모델은 웹 서비스의 질환 후보 순위 산정용으로 사용합니다. 사용자가 지적한 궤양성 각막질환 예시 이미지에 대해, 기존 이진 모델 순위 방식은 백내장을 높게 표시했지만 다중분류 방식으로 교체한 뒤 궤양성 각막질환이 1순위로 표시되도록 개선했습니다.

## 한계와 안전 문구

- 본 프로젝트는 수의학적 진단을 대체하지 않습니다.
- 데이터셋 촬영 장비, 조명, 거리, 품종 분포와 다른 일반 사진에서는 확률이 불안정할 수 있습니다.
- 일부 안구 질환은 시각적으로 유사해 단일 이미지 기반 모델만으로 구분이 어렵습니다.
- 실제 서비스 적용 전에는 더 큰 외부 검증셋, 수의사 검수, 눈 영역 자동 검출 모델 고도화가 필요합니다.

## 사용 파일

- 다중분류 모델: `project/outputs/best_eye_multiclass_efficientnet_b0.pt`
- 다중분류 학습 코드: `project/train_multiclass_pytorch.py`
- 다중분류 예측 코드: `project/predict_multiclass_eye.py`
- 질환별 이진 모델: `project/outputs/best_*_efficientnet_b0.pt`
- 웹 추론 코드: `project/webapp/predictor/views.py`
