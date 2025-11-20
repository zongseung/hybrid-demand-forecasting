# 하이브리드 전력 수요 예측 시스템

## 개요

이 시스템은 **Trend + Fourier + LSTM** 하이브리드 모델을 사용하여 24시간 전력 수요를 예측합니다.

### 모델 구조

```
최종 예측 = Trend (24h) + Seasonality (24h) + Residual (24h)
```

1. **Trend**: 로그-선형 회귀로 장기 추세 예측
2. **Seasonality**: Fourier 분석으로 다중 계절성 예측 (일간 + 주간 + 연간)
3. **Residual**: Seq2Seq LSTM으로 잔차 예측

### 주요 특징

- ✅ **완전한 24시간 예측**: 모든 컴포넌트가 미래 24시간 예측
- ✅ **하이퍼파라미터 튜닝**: Random Search로 최적 LSTM 파라미터 탐색
- ✅ **MLflow 트래킹**: 모든 학습 과정 추적 및 버전 관리
- ✅ **Grafana 연동**: 예측 결과 및 평가지표 실시간 시각화
- ✅ **자동 재학습**: 매주 일요일 02:00에 자동 재학습

## 설치

### 필수 패키지

```bash
pip install -r requirements.txt
```

추가로 필요한 패키지:
```bash
pip install influxdb-client
```

## 사용 방법

### 1. 모델 학습

#### 1.1. 전체 파이프라인 학습

```bash
python flows/weekly_retrain_hybrid.py
```

이 스크립트는:
- Trend 모델 학습 (로그-선형 회귀)
- Fourier 모델 학습 (Grid Search)
- LSTM 모델 학습 (Random Search)
- 모든 모델을 `models/production/`에 저장
- MLflow에 학습 과정 기록

#### 1.2. 파라미터 설정

```python
python flows/weekly_retrain_hybrid.py --help

# 예시
python flows/weekly_retrain_hybrid.py \
    --csv_path /path/to/data.csv \
    --window_size 168 \
    --horizon 24 \
    --n_lstm_iter 50 \
    --lstm_epochs 100
```

### 2. 모델 추론

#### 2.1. 실시간 추론

```bash
python inference_demo.py
```

이 스크립트는:
- 학습된 모델 로드
- 최근 168시간(7일) 데이터 사용
- 미래 24시간 예측 생성
- 결과를 Grafana로 전송

#### 2.2. Python 코드에서 사용

```python
from src.models.inference_pipeline import PowerDemandForecaster, create_exog_features
import pandas as pd

# 1. 모델 로드
forecaster = PowerDemandForecaster(
    model_dir="models/production",
    window_size=168,
    horizon=24
)
forecaster.load_models()

# 2. 예측
forecast_df = forecaster.forecast_with_timestamps(
    historical_data=historical_data,  # 최근 168시간 데이터
    exog_features_future=exog_future,  # 미래 24시간 외부 변수
    device='cuda'
)

# 3. 결과 확인
print(forecast_df[['trend', 'seasonality', 'residual', 'forecast']])
```

### 3. 자동 재학습 설정

#### 3.1. 매주 일요일 02:00 자동 재학습

```bash
python deploy_weekly_retrain.py
```

이 명령은 Prefect에 다음 스케줄을 배포합니다:
- **실행 주기**: 매주 일요일 02:00 (Asia/Seoul)
- **작업**: 전체 파이프라인 재학습
- **하이퍼파라미터 튜닝**: Random Search 50회 반복

#### 3.2. Prefect 서버 시작

```bash
# 터미널 1: Prefect 서버
prefect server start

# 터미널 2: Prefect agent
prefect agent start -q default
```

### 4. Grafana 대시보드

#### 4.1. InfluxDB 설정

환경 변수 설정:
```bash
export INFLUXDB_URL="http://localhost:8086"
export INFLUXDB_TOKEN="your_token"
export INFLUXDB_ORG="open-stef"
```

#### 4.2. 전송되는 데이터

1. **예측 결과** (`power_demand_forecast`)
   - `trend`: 추세 성분
   - `seasonality`: 계절성 성분
   - `residual`: 잔차 성분
   - `forecast`: 최종 예측값

2. **평가 지표** (`forecast_metrics`)
   - `mae`: Mean Absolute Error
   - `mse`: Mean Squared Error
   - `rmse`: Root Mean Squared Error
   - `r2`: R² Score
   - `mape`: Mean Absolute Percentage Error
   - `smape`: Symmetric MAPE

3. **Horizon별 지표** (`horizon_metrics`)
   - 각 예측 시점(1h~24h)별 평가지표

4. **컴포넌트 통계** (`component_metrics`)
   - 각 컴포넌트(trend, seasonality, residual)의 통계량

## 파일 구조

```
open-stef/
├── src/
│   ├── models/
│   │   ├── seq2seq_lstm.py          # Seq2Seq LSTM 모델
│   │   ├── train_pipeline.py        # 학습 파이프라인
│   │   └── inference_pipeline.py    # 추론 파이프라인
│   └── utils/
│       └── grafana_client.py        # Grafana 연동
├── flows/
│   └── weekly_retrain_hybrid.py     # 주간 재학습 플로우
├── models/
│   └── production/                  # 학습된 모델
│       ├── trend_model.pkl
│       ├── fourier_model.pkl
│       ├── lstm_model.pth
│       ├── residual_scaler.pkl
│       └── config.json
├── inference_demo.py                # 추론 데모 스크립트
├── deploy_weekly_retrain.py         # 자동 재학습 배포
└── README_HYBRID_FORECASTING.md     # 이 문서
```

## 모델 성능

### 예상 성능 지표

테스트 데이터 기준:
- **MAE**: ~500-800 MW
- **RMSE**: ~800-1200 MW
- **MAPE**: ~2-4%
- **R²**: ~0.95-0.98

### 컴포넌트별 기여도

1. **Trend**: 장기 추세 (전체 변동의 ~70%)
2. **Seasonality**: 주기적 패턴 (전체 변동의 ~20%)
3. **Residual**: 단기 변동 (전체 변동의 ~10%)

## 하이퍼파라미터

### LSTM 모델

Random Search로 탐색되는 파라미터:

```python
{
    'hidden_size': [64, 128, 256],
    'num_layers': [2, 3],
    'dropout': [0.1, 0.2, 0.3],
    'bidirectional': [False, True],
    'use_attention': [False, True],
    'batch_size': [32, 64],
    'learning_rate': [0.0005, 0.001, 0.005],
    'optimizer': ['adamw'],
    'weight_decay': [0.0, 1e-4],
    'grad_clip': [0.5, 1.0],
    'teacher_forcing_ratio': [0.0, 0.3, 0.5, 0.7],
    'scheduler': ['cosine', 'reduce'],
    'early_stopping_patience': [10, 15]
}
```

### Fourier 모델

Grid Search로 탐색되는 파라미터:

```python
{
    'Kd': [1, 2, 3],           # 일간 조화항 수
    'Kw': [1,2,3,4,5,6,7,8],   # 주간 조화항 수
    'Ky': [1, 2, 3, 4]         # 연간 조화항 수
}
```

## 문제 해결

### 1. CUDA 메모리 부족

```python
# inference_pipeline.py에서 device를 'cpu'로 변경
forecaster.forecast(..., device='cpu')
```

### 2. InfluxDB 연결 오류

```bash
# InfluxDB 상태 확인
curl http://localhost:8086/health

# 환경 변수 확인
echo $INFLUXDB_TOKEN
```

### 3. 모델 로드 실패

```bash
# 모델 파일 존재 확인
ls -la models/production/

# 재학습 실행
python flows/weekly_retrain_hybrid.py
```

## MLflow 추적

### MLflow UI 시작

```bash
mlflow ui --host 0.0.0.0 --port 5000
```

브라우저에서 `http://localhost:5000` 접속

### 기록되는 내용

1. **Parameters**
   - 모델 하이퍼파라미터
   - 데이터 분할 기준
   - 학습 설정

2. **Metrics**
   - 학습/검증 손실
   - 테스트 평가지표
   - Horizon별 성능

3. **Artifacts**
   - 학습된 모델 파일
   - 설정 파일
   - 예측 결과

## 성능 개선 팁

### 1. LSTM 모델

- **더 많은 반복**: `n_lstm_iter` 증가 (50 → 100)
- **더 긴 학습**: `lstm_epochs` 증가 (100 → 200)
- **더 큰 모델**: `hidden_size` 증가 (256 → 512)

### 2. Fourier 모델

- **더 많은 조화항**: Kd, Kw, Ky 범위 확장
- **외부 변수 추가**: 날씨, 경제 지표 등

### 3. 데이터

- **더 많은 학습 데이터**: 최소 2-3년 권장
- **데이터 품질**: 결측치 및 이상치 처리
- **Feature Engineering**: 추가 외부 변수 생성

## 참고 자료

- [Seq2Seq 논문](https://arxiv.org/abs/1409.3215)
- [Time Series Forecasting with LSTM](https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/)
- [Fourier Analysis for Time Series](https://otexts.com/fpp2/complexseasonality.html)

## 라이센스

MIT License

## 문의

문제가 발생하면 이슈를 등록해주세요.

