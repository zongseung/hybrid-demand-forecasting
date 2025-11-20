# hybrid model 기반의 전력 수요 예측 시스템

<div align="center">

**하이브리드 딥러닝 기반 24시간 전력 수요 예측**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org/)

</div>

---

## 📊 개요

Open-STEF는 **Trend + Fourier + LSTM** 하이브리드 모델을 사용하여 24시간 전력 수요를 예측하는 시스템입니다.

### 핵심 특징

- 🎯 **완전한 24시간 예측**: 모든 컴포넌트가 미래 24시간 예측
- 🔧 **자동 하이퍼파라미터 튜닝**: Random Search로 최적 LSTM 파라미터 탐색
- 📈 **MLflow 통합**: 모든 학습 과정 추적 및 버전 관리
- 📊 **Grafana 대시보드**: 예측 결과 및 평가지표 실시간 시각화
- ⏰ **자동 재학습**: 매주 일요일 02:00 자동 재학습

## 🚀 빠른 시작

### 1. 설치 (1분)

```bash
git clone https://github.com/yourusername/open-stef.git
cd open-stef

# uv로 설치 (10-100배 빠름! 🚀)
make install
```

### 2. 서비스 시작

```bash
# 모든 서비스 한 번에 시작
make up
```

**자동 실행**: Prefect, MLflow, InfluxDB, Grafana, PostgreSQL, FastAPI

**포트 정보** (충돌 방지를 위해 변경됨):
- Prefect UI: `http://localhost:14200`
- FastAPI: `http://localhost:18000`
- MLflow: `http://localhost:15000`
- Grafana: `http://localhost:13000`
- InfluxDB: `http://localhost:18086`
- PostgreSQL: `localhost:15432`

### 3. 모델 학습

```bash
# 빠른 테스트 (15-30분)
make train

# 프로덕션 학습 (1-2시간)
make train-prod
```

### 4. 예측 실행

```bash
make inference
```

### 5. 자동 재학습 배포

```bash
# 매주 일요일 02:00 자동 재학습
make deploy
```

### 6. 대시보드 확인

```bash
make mlflow-ui    # MLflow UI
make grafana-ui   # Grafana UI
make prefect-ui   # Prefect UI
```

## 📖 문서

- **[빠른 시작 가이드](QUICKSTART_HYBRID.md)** - 5분 만에 시작하기
- **[전체 문서](README_HYBRID_FORECASTING.md)** - 상세한 사용 방법
- **[아키텍처](ARCHITECTURE_HYBRID.md)** - 시스템 구조 및 설계

## 🏗️ 모델 구조

```
최종 예측 = Trend(24h) + Seasonality(24h) + Residual(24h)
```

### 1. Trend Component
- **방법**: 로그-선형 회귀 (OLS)
- **기여도**: ~70%

### 2. Seasonality Component
- **방법**: Fourier 분석 (일간 + 주간 + 연간)
- **파라미터**: Kd=3, Kw=13, Ky=3 (고정)
- **기여도**: ~20%

### 3. Residual Component
- **방법**: Seq2Seq LSTM (Encoder-Decoder)
- **입력**: 168시간 (7일)
- **출력**: 24시간
- **기여도**: ~10%

## 📁 프로젝트 구조

```
open-stef/
├── src/
│   ├── models/
│   │   ├── seq2seq_lstm.py          # Seq2Seq LSTM 모델
│   │   ├── train_pipeline.py        # 학습 파이프라인
│   │   └── inference_pipeline.py    # 추론 파이프라인
│   ├── utils/
│   │   └── grafana_client.py        # Grafana 연동
│   └── data/
│       └── preprocess.py            # 데이터 전처리
├── flows/
│   └── weekly_retrain_hybrid.py     # 주간 재학습 플로우
├── inference_demo.py                # 추론 데모 스크립트
├── deploy_weekly_retrain.py         # 자동 재학습 배포
└── models/production/               # 학습된 모델
```

## 📊 성능

| Metric | 값 |
|--------|-----|
| **MAE** | 500-800 MW |
| **RMSE** | 800-1200 MW |
| **MAPE** | 2-4% |
| **R²** | 0.95-0.98 |

## 🔧 기술 스택

- **Deep Learning**: PyTorch 2.1+
- **ML Framework**: scikit-learn, statsmodels
- **Orchestration**: Prefect
- **Experiment Tracking**: MLflow
- **Visualization**: Grafana + InfluxDB
- **API**: FastAPI

## 📈 사용 예시

### Python 코드에서 사용

```python
from src.models.inference_pipeline import PowerDemandForecaster

# 모델 로드
forecaster = PowerDemandForecaster(
    model_dir="models/production",
    window_size=168,
    horizon=24
)
forecaster.load_models()

# 24시간 예측
forecast_df = forecaster.forecast_with_timestamps(
    historical_data=historical_data,
    exog_features_future=exog_future,
    device='cuda'
)

print(forecast_df[['trend', 'seasonality', 'residual', 'forecast']])
```

## 🤝 기여하기

이슈와 Pull Request를 환영합니다!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

