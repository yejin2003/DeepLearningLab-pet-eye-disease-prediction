# 사용 데이터 설명 및 제출 안내

## 데이터 출처

- 데이터명: AIHub 반려동물 안구 질환 데이터
- 데이터 URL: https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=562
- 구축년도: 2021
- 데이터 유형: jpg 이미지, json 라벨
- 주요 라벨: 반려견/반려묘 안구 질환, 바운딩박스, 세그멘테이션, 질환명, 질환 정도

AIHub 소개에 따르면 이 데이터는 반려동물 5,000마리 이상, 안구질환 이미지 및 라벨링 정보를 포함하는 대규모 이미지 데이터셋입니다.

## 실제 사용 범위

전체 데이터를 모두 사용하면 저장 공간과 처리 시간이 매우 크기 때문에, 본 프로젝트에서는 다음 범위를 중심으로 사용했습니다.

```text
153.반려동물 안구질환 데이터/
└─ 01.데이터/
   └─ 2.Validation/
      └─ 원천데이터/
```

웹 서비스와 학습 CSV는 위 폴더 구조에서 질환명과 중증도 폴더를 읽어 생성했습니다.

## 생성한 학습 CSV

| 파일 | 목적 | 샘플 수 |
|---|---|---:|
| `eye_multiclass_dataset.csv` | 정상 + 10개 질환 다중분류 | 25,113 |
| `eye_disease_type_dataset.csv` | 질환 이미지 대상 질환 종류 분류 | 14,127 |
| `dog_general_abnormal_dataset.csv` | 정상/질환 의심 이진분류 | 22,058 |
| `conjunctivitis_dataset.csv` | 결막염 이진분류 | 3,186 |
| `blepharitis_dataset.csv` | 안검염 이진분류 | 1,965 |
| `epiphora_dataset.csv` | 유루증 이진분류 | 2,396 |
| `sclerosis_dataset.csv` | 핵경화 이진분류 | 2,399 |
| `cataract_dataset.csv` | 백내장 이진분류 | 5,462 |
| `ulcerative_keratitis_dataset.csv` | 궤양성 각막질환 이진분류 | 2,587 |
| `non_ulcerative_keratitis_dataset.csv` | 비궤양성 각막질환 이진분류 | 1,786 |
| `pigmentary_keratitis_dataset.csv` | 색소침착성 각막염 이진분류 | 1,732 |
| `entropion_dataset.csv` | 안검내반증 이진분류 | 2,424 |
| `eyelid_tumor_dataset.csv` | 안검종양 이진분류 | 1,176 |

## 다중분류 데이터 구성

`eye_multiclass_dataset.csv` 기준 split은 다음과 같습니다.

| split | 샘플 수 |
|---|---:|
| train | 17,574 |
| validation | 3,768 |
| test | 3,771 |

주요 클래스 분포는 다음과 같습니다.

| 클래스 | 샘플 수 |
|---|---:|
| 정상 | 10,986 |
| 백내장 | 3,585 |
| 궤양성 각막질환 | 1,727 |
| 결막염 | 1,599 |
| 안검내반증 | 1,224 |
| 핵경화 | 1,198 |
| 유루증 | 1,196 |
| 안검염 | 984 |
| 색소침착성 각막염 | 852 |
| 비궤양성 각막질환 | 586 |
| 안검종양 | 576 |

## 제출 방식

교수님 공지에 따르면 사용 데이터의 용량이 큰 경우 오픈데이터셋은 관련 URL을 제출하면 됩니다. 따라서 제출 파일에는 원본 이미지 전체를 중복 포함하지 않고 다음을 포함합니다.

- AIHub 데이터셋 URL
- 사용한 폴더 구조 설명
- 생성한 CSV 파일
- 클래스 및 split 요약
- 데이터 전처리 코드

원본 이미지가 꼭 필요한 경우에는 AIHub 이용정책을 확인한 뒤, 교수님께 별도 대용량 제출 방식으로 전달하는 것을 권장합니다.
