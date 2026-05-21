# SECOM 반도체 공정 불량 예측

UCI/Kaggle SECOM 데이터(`paresh2047/uci-semcom`) 기반 Tabular 분류 프로젝트.

## Streamlit Cloud 배포

| 설정 | 값 |
|------|-----|
| **Main file** | `streamlit_app.py` |
| **Requirements** | `requirements.txt` |
| **System packages** | `packages.txt` (LightGBM) |

1. [share.streamlit.io](https://share.streamlit.io) → New app → GitHub repo 연결
2. Main file: `streamlit_app.py`
3. Deploy

로컬 실행:

```powershell
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## 로컬 학습 파이프라인

1. EDA + 데이터 품질 리포트
2. Stratified 5-fold OOF 비교실험 (전처리 × 샘플링 × 모델)
3. Wilcoxon + Holm 통계검정
4. 최종 모델 Dev 재학습 + Test 1회 평가
5. Streamlit 대시보드

```powershell
pip install -r requirements-train.txt
python run_quick_experiments.py
streamlit run streamlit_app.py
```

`.env`에 `KAGGLE_USERNAME`, `KAGGLE_API_TOKEN` 필요 (데이터 다운로드 시).

## 주요 산출물

| 경로 | 설명 |
|------|------|
| `data/raw/uci-secom.csv` | 원본 |
| `artifacts/experiments/experiments_leaderboard.csv` | OOF 실험 순위 |
| `models/final_config.json` | 최종 모델 설정·Test 지표 |
| `models/predictions_test.csv` | Test 예측 |
