# 모델 통제 아키텍처

## 📊 현재 시스템 구조

### **Prefect가 모델 통제를 담당합니다**

현재 시스템에서 모델의 생명주기(생성, 배포, 관리)는 **Prefect**가 제어하고, **MLflow**는 실험 추적 및 메트릭 로깅만 담당합니다.

## 🔄 역할 분담

### 1. **Prefect (워크플로우 오케스트레이션 + 모델 배포 제어)**

#### 담당 업무:
- ✅ **스케줄링**: 언제 모델을 학습/배포할지 결정
- ✅ **워크플로우 실행**: 데이터 수집 → 학습 → 검증 → 배포 파이프라인 제어
- ✅ **모델 배포 결정**: 새 모델이 프로덕션 모델보다 좋은지 비교
- ✅ **모델 저장/로드**: 파일 시스템 기반 (`models/production/`, `models/temp/`)
- ✅ **배포 상태 관리**: PostgreSQL `model_deployments` 테이블에 기록

#### 주요 코드 위치:
```python
# flows/weekly_retrain_v2.py
@flow(name="weekly_model_retrain_v2")
def weekly_model_retrain_v2_flow(...):
    # 1. 데이터 수집
    df = fetch_training_data_task(lookback_days)
    
    # 2. 모델 학습
    new_forecaster = retrain_models_task(train_df, best_params)
    
    # 3. 성능 검증
    new_metrics = validate_new_models_task(new_forecaster, val_df)
    
    # 4. 배포 결정 (Prefect가 제어)
    should_deploy, metrics = compare_with_production_task(new_metrics)
    
    if should_deploy:
        # 5. 모델 배포 (Prefect가 실행)
        deploy_models_task(model_version, best_params)
        # → models/temp/ → models/production/ 복사
        # → PostgreSQL에 배포 기록
```

#### 모델 저장 위치:
- **임시 모델**: `models/temp/` (학습 중)
- **프로덕션 모델**: `models/production/` (배포된 모델)
- **로드 경로**: `settings.PRODUCTION_MODEL_PATH`

### 2. **MLflow (실험 추적 및 메트릭 로깅)**

#### 담당 업무:
- ✅ **파라미터 로깅**: 하이퍼파라미터, 모델 버전 등
- ✅ **메트릭 로깅**: MAPE, RMSE, MAE, R² 등
- ✅ **실험 추적**: 각 실행(run)의 기록
- ❌ **모델 저장/로드**: 현재 사용하지 않음
- ❌ **모델 배포 제어**: 관여하지 않음

#### 주요 코드 위치:
```python
# flows/weekly_retrain_v2.py
with MLflowTracker(run_name=run_name, tags={...}) as tracker:
    # 파라미터 로깅
    tracker.log_params({
        "fourier_order": 10,
        "lstm_hidden_units": 128,
        "model_version": model_version
    })
    
    # 메트릭 로깅
    tracker.log_metrics({
        "mape": 5.2,
        "rmse": 1200.5,
        "mae": 800.3
    })
    
    # ⚠️ 모델 자체는 MLflow에 저장하지 않음
    # 모델은 파일 시스템에 저장됨
```

### 3. **PostgreSQL (배포 상태 추적)**

#### 담당 업무:
- ✅ **배포 이력 관리**: `model_deployments` 테이블
- ✅ **모델 메트릭 저장**: `model_metrics` 테이블
- ✅ **배포 상태 추적**: active/archived 상태 관리

## 📋 모델 생명주기

```
1. [Prefect] 주간 재학습 스케줄 (매주 일요일 03:00)
   ↓
2. [Prefect] 데이터 수집 및 전처리
   ↓
3. [Prefect] 모델 학습 (HybridForecasterV2.fit())
   ↓
4. [Prefect] 모델 저장 → models/temp/
   ↓
5. [Prefect] 검증 데이터로 성능 평가
   ↓
6. [MLflow] 메트릭 로깅 (MAPE, RMSE 등)
   ↓
7. [Prefect] 프로덕션 모델과 비교
   ↓
8. [Prefect] 배포 결정 (should_deploy)
   ↓
9. [Prefect] 배포 실행 (models/temp/ → models/production/)
   ↓
10. [PostgreSQL] 배포 기록 저장 (model_deployments)
   ↓
11. [Prefect] 일일 예측에서 프로덕션 모델 로드
```

## 🔍 코드 상세 분석

### 모델 배포 제어 (Prefect)

```python
# flows/weekly_retrain_v2.py:188-233
@task(name="deploy_models")
def deploy_models_task(model_version: str, best_params: dict = None):
    """Deploy new models to production"""
    # 1. 파일 시스템에서 복사
    temp_path = settings.TEMP_MODEL_PATH      # models/temp/
    prod_path = settings.PRODUCTION_MODEL_PATH # models/production/
    
    # 2. 모델 파일 복사
    shutil.copy2(src, dst)  # temp → production
    
    # 3. PostgreSQL에 배포 기록
    execute_query(
        "INSERT INTO model_deployments ...",
        (model_version, 'hybrid_forecaster_v2', prod_path, ...)
    )
    
    # 4. 이전 배포 비활성화
    execute_query(
        "UPDATE model_deployments SET status = 'archived' ..."
    )
```

### 모델 로드 (Prefect Flow)

```python
# flows/daily_forecast.py:64-93
@task(name="run_forecast")
def run_forecast_task(df: pd.DataFrame, n_steps: int = 168):
    # 프로덕션 모델 로드 (파일 시스템에서)
    forecaster = HybridForecaster.load_models(
        settings.PRODUCTION_MODEL_PATH  # models/production/
    )
    
    # 예측 실행
    results = forecaster.predict(n_steps=n_steps)
    return results
```

### MLflow 추적 (로깅만)

```python
# flows/weekly_retrain_v2.py:272-316
with MLflowTracker(...) as tracker:
    # 파라미터 로깅
    tracker.log_params(best_params)
    
    # 메트릭 로깅
    tracker.log_metrics(new_metrics)
    
    # ⚠️ 모델은 MLflow에 저장하지 않음
    # tracker.log_model(...) 호출 없음
```

## 🎯 결론

### **Prefect가 모델 통제를 담당합니다**

1. **모델 배포 결정**: Prefect의 `compare_with_production_task()`가 결정
2. **모델 저장/로드**: Prefect가 파일 시스템 기반으로 관리
3. **배포 실행**: Prefect의 `deploy_models_task()`가 실행
4. **스케줄링**: Prefect가 워크플로우 스케줄 관리

### **MLflow는 추적만 담당합니다**

1. **메트릭 로깅**: 성능 지표 기록
2. **파라미터 로깅**: 하이퍼파라미터 기록
3. **실험 추적**: 각 실행의 기록
4. **모델 저장/배포**: 현재 사용하지 않음

## 💡 개선 제안

현재는 MLflow에 모델을 저장하지 않습니다. 만약 MLflow Model Registry를 사용하고 싶다면:

```python
# flows/weekly_retrain_v2.py에 추가
if should_deploy:
    # MLflow에 모델 저장
    tracker.log_model(
        new_forecaster,
        artifact_path="model",
        model_type="pytorch"
    )
    
    # MLflow Model Registry에 등록
    mlflow.register_model(
        model_uri=f"runs:/{tracker.run.info.run_id}/model",
        name="demand_forecasting_model"
    )
```

하지만 현재는 **Prefect + 파일 시스템** 방식이 더 단순하고 효과적입니다.

