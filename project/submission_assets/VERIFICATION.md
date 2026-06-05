# 최종 검증 기록

검증일: 2026-06-01

## 코드 검증

다음 파일을 Python bytecode compile로 확인했다.

- `project/webapp/predictor/views.py`
- `project/predict_multiclass_eye.py`
- `project/train_multiclass_pytorch.py`
- `project/prepare_multiclass_csv.py`

결과: 문법 오류 없음

## 웹 앱 검증

로컬 Django 서버 실행 후 다음 주소에 접속했다.

```text
http://127.0.0.1:8000/
```

응답 상태: HTTP 200

API 테스트:

```text
POST /api/predict/
```

AIHub 궤양성 각막질환 예시 이미지 업로드 결과:

- 1순위: 궤양성 각막질환
- 2순위: 안검염
- 3순위: 결막염

이는 기존 이진 모델 순위 방식에서 백내장이 높게 나오던 문제를 다중분류 모델로 개선했음을 확인한 결과이다.

## 보고서 검증

- DOCX 생성 완료: `deliverables/24013778김예진_기말프로젝트_보고서.docx`
- 내부 Word 패키지 확인 완료
- LibreOffice/soffice가 로컬 환경에 설치되어 있지 않아 DOCX -> PNG 자동 렌더링은 수행하지 못했다.

## 발표자료 검증

- PPTX 생성 완료: `deliverables/24013778김예진_기말프로젝트_발표자료.pptx`
- 내부 PowerPoint 패키지 확인 완료
- slide XML 기준 10개 슬라이드 존재 확인
- PowerPoint COM이 설치되어 있지 않아 PPTX -> PNG 자동 렌더링은 수행하지 못했다.

## 제출 압축 파일 검증

- `deliverables/source_code.zip`: 61개 파일 포함, 주 다중분류 모델 포함
- `deliverables/used_data_submission.zip`: 16개 파일 포함, 데이터 설명서와 학습 CSV 포함
