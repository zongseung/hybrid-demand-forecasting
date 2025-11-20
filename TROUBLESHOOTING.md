# 🔧 문제 해결 가이드 (Troubleshooting)

## Prefect UI 접속 문제

### 증상: `ERR_CONNECTION_REFUSED`

브라우저에서 `http://localhost:14200` 접속 시 "사이트에 연결할 수 없음" 에러

### 해결 방법

#### 1️⃣ 서비스 상태 확인

```bash
# 모든 서비스가 실행 중인지 확인
docker compose ps

# Prefect 서버 로그 확인
docker compose logs prefect-server --tail=20

# 포트가 열려있는지 확인
netstat -tlnp | grep 14200
```

#### 2️⃣ 로컬 환경 (Local Machine)

**Option A: 브라우저 캐시 삭제**
- Chrome: `Ctrl+Shift+Delete` → 캐시 삭제
- Firefox: `Ctrl+Shift+Delete` → 캐시 삭제
- 시크릿/프라이빗 모드로 시도

**Option B: 다른 URL 시도**
```bash
# localhost 대신 127.0.0.1 사용
http://127.0.0.1:14200

# IPv6 사용
http://[::1]:14200
```

**Option C: curl로 테스트**
```bash
# 서버가 응답하는지 확인
curl http://localhost:14200

# API 엔드포인트 확인
curl http://localhost:14200/api/health
```

#### 3️⃣ 원격 서버 (Remote Server)

원격 서버에서 Docker를 실행하고 로컬 브라우저로 접속하려는 경우:

**SSH 터널링 설정**

```bash
# 모든 UI 서비스 포트를 터널링
ssh -L 14200:localhost:14200 \
    -L 18000:localhost:18000 \
    -L 13000:localhost:13000 \
    -L 15000:localhost:15000 \
    user@remote-server

# 터널링 후 로컬 브라우저에서 접속
http://localhost:14200  # Prefect UI
http://localhost:18000  # FastAPI
http://localhost:13000  # Grafana
http://localhost:15000  # MLflow
```

**VS Code Remote SSH를 사용하는 경우**
- VS Code가 자동으로 포트 포워딩 설정
- "Ports" 탭에서 포트 추가: 14200, 18000, 13000, 15000

#### 4️⃣ 방화벽 확인

```bash
# 방화벽 상태 확인
sudo ufw status

# 필요시 포트 허용
sudo ufw allow 14200
sudo ufw allow 18000
sudo ufw allow 13000
sudo ufw allow 15000
```

#### 5️⃣ Docker 네트워크 재시작

```bash
# 서비스 재시작
docker compose restart prefect-server

# 또는 전체 재시작
docker compose down
docker compose up -d
```

---

## FastAPI 서비스 문제

### 증상: ModuleNotFoundError

```
ModuleNotFoundError: No module named 'src.models.hybrid_forecaster'
```

### 해결 방법

```bash
# Docker 이미지 재빌드
docker compose build fastapi

# 서비스 재시작
docker compose up -d fastapi

# 로그 확인
docker compose logs fastapi --tail=50
```

---

## MLflow 서비스 문제

### 증상: No module named 'psycopg2'

```
ModuleNotFoundError: No module named 'psycopg2'
```

### 해결 방법

```bash
# MLflow 이미지 재빌드 (psycopg2 포함)
docker compose build mlflow

# 서비스 재시작
docker compose up -d mlflow

# 로그 확인
docker compose logs mlflow --tail=50
```

---

## 포트 충돌 문제

### 증상: Port is already allocated

```
Error: failed to create endpoint: driver failed programming external connectivity
Bind for 0.0.0.0:XXXX failed: port is already allocated
```

### 해결 방법

**현재 사용 중인 포트:**
- PostgreSQL: 15432
- Prefect UI: 14200
- FastAPI: 18000
- MLflow: 15000
- InfluxDB: 18086
- Grafana: 13000

**포트 충돌 해결:**

```bash
# 1. 충돌하는 프로세스 확인
sudo lsof -i :PORT_NUMBER
# 또는
sudo netstat -tlnp | grep PORT_NUMBER

# 2. docker-compose.yml에서 포트 변경
# 예: 14200 -> 다른 포트로 변경

# 3. 서비스 재시작
docker compose down
docker compose up -d
```

---

## 데이터베이스 연결 문제

### 증상: Database connection failed

### 해결 방법

```bash
# PostgreSQL 상태 확인
docker compose logs postgres --tail=20

# 데이터베이스 접속 테스트
docker compose exec postgres psql -U postgres -d demand_forecasting -c "SELECT 1"

# 데이터베이스 리셋 (주의: 모든 데이터 삭제됨)
make db-reset
```

---

## 일반적인 문제 해결 단계

### 1. 로그 확인
```bash
# 모든 서비스 로그
docker compose logs -f

# 특정 서비스 로그
docker compose logs -f SERVICE_NAME

# 최근 N줄만 보기
docker compose logs --tail=50 SERVICE_NAME
```

### 2. 서비스 재시작
```bash
# 특정 서비스 재시작
docker compose restart SERVICE_NAME

# 모든 서비스 재시작
make restart
```

### 3. 완전 재시작
```bash
# 컨테이너 중지 및 제거
docker compose down

# 볼륨까지 제거 (주의: 데이터 삭제)
docker compose down -v

# 이미지 재빌드 및 시작
docker compose build
docker compose up -d
```

### 4. 시스템 정리
```bash
# 미사용 컨테이너/이미지 정리
make clean

# 모든 것 정리 (주의: 모든 데이터 삭제)
make clean-all
```

---

## 유용한 명령어

```bash
# 서비스 상태 확인
make status

# 모든 서비스 로그
make logs

# Prefect UI 열기
make prefect-ui

# Grafana UI 열기
make grafana-ui

# MLflow UI 열기
make mlflow-ui

# 데이터베이스 접속
make db-shell
```

---

## 추가 도움이 필요하신가요?

1. GitHub Issues: [프로젝트 URL]
2. Documentation: `README_HYBRID_FORECASTING.md`
3. Architecture Guide: `ARCHITECTURE_HYBRID.md`
4. Quick Start: `QUICKSTART_HYBRID.md`

