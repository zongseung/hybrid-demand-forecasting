# 하이브리드 모델 빠른 시작 가이드

이 가이드는 하이브리드 전력 수요 예측 모델을 빠르게 시작하는 방법을 안내합니다.

## 1. 환경 설정 (1분)

### 1.1. 패키지 설치 (uv 사용 - 10-100배 빠름!)

```bash
cd /mnt/nvme/open-stef

# Option 1: uv로 설치 (권장, 초고속 🚀)
make install
# 또는
uv sync

# Option 2: pip로 설치 (느림)
pip install -r requirements.txt
```

**속도 비교**: uv는 pip보다 **10-100배 빠릅니다!**

### 1.2. 환경 변수 설정

`.env` 파일 생성:

```bash
# InfluxDB 설정 (Grafana 연동)
INFLUXDB_URL=http://localhost:8086
INFLUXDB_TOKEN=your_influxdb_token
INFLUXDB_ORG=open-stef

# MLflow 설정
MLFLOW_TRACKING_URI=http://localhost:5000
```

### 1.3. 서비스 시작 (Docker Compose)

```bash
# 모든 서비스 한 번에 시작 (간편!)
make up

# 또는 수동으로
docker-compose up -d
```

**자동으로 시작되는 서비스**:
- ✅ PostgreSQL (TimescaleDB) - 포트 **15432** (외부) / 5432 (내부)
- ✅ Prefect Server - 포트 **14200** (외부) / 4200 (내부)
- ✅ Prefect Agent - 자동 실행
- ✅ MLflow - 포트 **15000** (외부) / 5000 (내부)
- ✅ InfluxDB - 포트 **18086** (외부) / 8086 (내부)
- ✅ Grafana - 포트 **13000** (외부) / 3000 (내부)
- ✅ FastAPI - 포트 **18000** (외부) / 8000 (내부)

> **참고**: 포트 충돌을 피하기 위해 외부 포트를 변경했습니다. 컨테이너 간 통신은 내부 포트를 사용합니다.

**서비스 상태 확인**:
```bash
make status
```

## 2. 모델 학습 (1-2시간)

### 2.1. 간단한 테스트 학습 (빠른 확인)

```bash
# Makefile 사용 (간편!)
make train

# 또는 직접 실행
uv run python flows/weekly_retrain_hybrid.py \
    --csv_path /mnt/nvme/tilting/power_demand_final.csv \
    --n_lstm_iter 5 \
    --lstm_epochs 10
```

**예상 소요 시간**: 약 15-30분
- Trend 학습: 1분
- Fourier 학습: **10초 미만** (고정 파라미터)
- LSTM 학습: 10-20분 (5회 반복, 각 10 에폭)

### 2.2. 실제 프로덕션 학습

```bash
# Makefile 사용 (간편!)
make train-prod

# 또는 직접 실행
uv run python flows/weekly_retrain_hybrid.py \
    --csv_path /mnt/nvme/tilting/power_demand_final.csv \
    --n_lstm_iter 50 \
    --lstm_epochs 100
```

**예상 소요 시간**: 약 1-2시간
- Trend 학습: 1분
- Fourier 학습: **10초 미만** (Grid Search 제거로 대폭 단축!)
- LSTM 학습: 1-2시간 (50회 반복, 각 최대 100 에폭)

### 2.3. 학습 진행 상황 확인

**MLflow UI**:
```
http://localhost:5000
```
- 실험 이름: `power-demand-hybrid-weekly`
- Run 이름: `weekly_retrain_YYYYMMDD`

**터미널 출력**:
```
================================================================================
TRAINING TREND MODEL (Log-Linear Regression)
================================================================================

✓ Trend model trained
  - R²: 0.9845

================================================================================
TRAINING FOURIER SEASONALITY MODEL (Grid Search)
================================================================================

Testing 96 combinations...
  Progress: 10/96 combinations tested
  ...

✓ Best Fourier model found
  - Kd (daily): 3
  - Kw (weekly): 7
  - Ky (yearly): 3

================================================================================
TRAINING LSTM RESIDUAL MODEL (Random Search)
================================================================================

Trial 1/50
================================================================================
Model parameters: 564,610

Epoch   Train Loss     Val Loss     Best Val     Status
-----------------------------------------------------------------
    1     0.943381     0.988513     0.988513     ✓ Best
    2     0.723702     0.949090     0.949090     ✓ Best
    ...
```

## 3. 모델 추론 (1분)

### 3.1. 기본 추론

```bash
# Makefile 사용 (간편!)
make inference

# 또는 직접 실행
uv run python inference_demo.py
```

이 스크립트는:
1. 학습된 모델을 로드합니다
2. 최근 168시간(7일) 데이터를 읽습니다
3. 미래 24시간을 예측합니다
4. 결과를 시각화하고 CSV로 저장합니다
5. Grafana로 결과를 전송합니다

### 3.2. 출력 예시

```
================================================================================
POWER DEMAND FORECASTING - INFERENCE DEMO
================================================================================

1️⃣ Initializing forecaster...

2️⃣ Loading trained models...
Loading models from: models/production
✓ Loaded trend model
✓ Loaded Fourier model (Kd=3, Kw=7, Ky=3)
✓ Loaded LSTM model
✓ Loaded residual scaler

✅ All models loaded successfully

3️⃣ Loading historical data...
📂 Loading historical data from: /mnt/nvme/tilting/power_demand_final.csv
✓ Loaded 51144 historical records
  - Date range: 2019-01-01 00:00:00 to 2024-12-31 23:00:00

4️⃣ Making 24-hour forecast...
📊 Forecasting next 24 hours...
  ├─ Forecasting trend...
  ├─ Forecasting seasonality...
  ├─ Forecasting residual (LSTM)...
  └─ Combining components...

✅ Forecast complete!
  - Trend range: [48500.23, 48650.45]
  - Seasonality range: [-1200.34, 1450.67]
  - Residual range: [-150.23, 180.45]
  - Final forecast range: [47800.12, 50100.89]

5️⃣ Plotting forecast...
✓ Plot saved to: forecast_demo.png

6️⃣ Evaluating forecast...
==================================================
  MAE       :   542.3456
  MSE       : 450123.7890
  RMSE      :   670.9123
  R2        :     0.9678
  MAPE      :     1.2345
  SMAPE     :     1.1234
==================================================

7️⃣ Sending results to Grafana...
✓ Grafana client initialized
📤 Sending forecast to Grafana...
✅ Sent 24 forecast points to Grafana
📤 Sending metrics to Grafana...
✅ Sent metrics to Grafana

✅ Forecast saved to: forecast_20241120_143025.csv

================================================================================
DEMO COMPLETE!
================================================================================
```

## 4. Grafana 대시보드 확인 (5분)

### 4.1. Grafana 접속

```
http://localhost:3000
```

기본 로그인:
- Username: `admin`
- Password: `admin`

### 4.2. 데이터 소스 추가

1. **Configuration** → **Data Sources** → **Add data source**
2. **InfluxDB** 선택
3. 설정:
   - Query Language: **Flux**
   - URL: `http://influxdb:8086`
   - Organization: `open-stef`
   - Token: (환경 변수에서 설정한 토큰)
   - Default Bucket: `power_demand`
4. **Save & Test**

### 4.3. 대시보드 생성

#### 패널 1: 24시간 예측 vs 실제

```flux
from(bucket: "power_demand")
  |> range(start: -7d)
  |> filter(fn: (r) => r._measurement == "power_demand_forecast")
  |> filter(fn: (r) => r._field == "forecast")
```

#### 패널 2: 예측 컴포넌트

```flux
from(bucket: "power_demand")
  |> range(start: -7d)
  |> filter(fn: (r) => r._measurement == "power_demand_forecast")
  |> filter(fn: (r) => r._field == "trend" or r._field == "seasonality" or r._field == "residual")
```

#### 패널 3: 평가 지표

```flux
from(bucket: "power_demand")
  |> range(start: -1h)
  |> filter(fn: (r) => r._measurement == "forecast_metrics")
```

## 5. 자동 재학습 설정 (2분)

### 5.1. 배포

```bash
# Makefile 사용 (간편!)
make deploy

# 또는 직접 실행
uv run python deploy_weekly_retrain.py
```

출력:
```
================================================================================
✅ DEPLOYMENT SUCCESSFUL
================================================================================
Deployment ID: abc123...
Schedule: Every Sunday at 2:00 AM (Asia/Seoul)
Flow: weekly_retrain_flow
Work Queue: default
================================================================================
```

### 5.2. 배포 확인

**Prefect UI**:
```
http://localhost:4200
```

**배포 목록 확인**:
```bash
prefect deployment ls
```

**수동 실행 (테스트)**:
```bash
prefect deployment run "Weekly Hybrid Model Retraining/weekly-hybrid-retrain-sunday"
```

## 6. 예측 결과 파일

학습 및 추론 후 생성되는 파일:

```
models/production/
├── trend_model.pkl          # Trend 모델
├── fourier_model.pkl        # Fourier 모델
├── lstm_model.pth           # LSTM 모델
├── residual_scaler.pkl      # Residual 스케일러
└── config.json              # 모델 설정

forecast_YYYYMMDD_HHMMSS.csv # 예측 결과
overall_metrics.csv          # 전체 평가지표
horizon_metrics.csv          # Horizon별 평가지표
forecast_demo.png            # 예측 시각화
```

## 7. 문제 해결

### 모델이 없다는 오류

```
❌ Models not found. Please train models first
```

**해결**:
```bash
python flows/weekly_retrain_hybrid.py
```

### CUDA 메모리 부족

```
RuntimeError: CUDA out of memory
```

**해결 1**: 배치 크기 줄이기
```python
# flows/weekly_retrain_hybrid.py 수정
# n_lstm_iter를 줄이거나, 배치 크기를 줄입니다
```

**해결 2**: CPU 사용
```python
# inference_demo.py에서
forecast_df = forecaster.forecast_with_timestamps(
    historical_data=historical_window,
    exog_features_future=exog_future,
    device='cpu'  # cuda → cpu
)
```

### InfluxDB 연결 오류

```
❌ Error sending forecast: Connection refused
```

**해결**:
```bash
# InfluxDB 상태 확인
docker ps | grep influxdb

# InfluxDB 시작
docker-compose up -d influxdb

# 환경 변수 확인
echo $INFLUXDB_TOKEN
```

### Prefect 연결 오류

```
Unable to connect to Prefect server
```

**해결**:
```bash
# Prefect 서버 상태 확인
curl http://localhost:4200/api/health

# Prefect 서버 재시작
prefect server start
```

## 8. 성능 최적화

### 학습 속도 향상

```bash
# GPU 사용 확인
python -c "import torch; print(torch.cuda.is_available())"

# 멀티 GPU 사용
CUDA_VISIBLE_DEVICES=0,1 python flows/weekly_retrain_hybrid.py
```

### 메모리 사용량 감소

```python
# train_pipeline.py에서 배치 크기 조정
param_distributions = {
    'batch_size': [16, 32],  # 64 → 32 또는 16
    ...
}
```

### 추론 속도 향상

```python
# LSTM 모델을 TorchScript로 컴파일
model = torch.jit.script(forecaster.lstm_model)
```

## 9. 다음 단계

✅ 학습 완료
✅ 추론 테스트
✅ Grafana 대시보드 구성
✅ 자동 재학습 설정

이제 다음을 수행할 수 있습니다:

1. **실시간 예측 API 구축**: FastAPI로 REST API 서비스 구축
2. **알림 시스템**: 예측 정확도가 떨어질 때 알림
3. **A/B 테스트**: 다양한 모델 비교
4. **앙상블 모델**: 여러 모델의 예측 결합

## 10. 추가 리소스

- [전체 문서](README_HYBRID_FORECASTING.md)
- [MLflow 문서](https://mlflow.org/docs/latest/index.html)
- [Prefect 문서](https://docs.prefect.io/)
- [InfluxDB 문서](https://docs.influxdata.com/)
- [Grafana 문서](https://grafana.com/docs/)

## 질문이나 문제가 있나요?

이슈를 등록하거나 문서를 참조하세요.

