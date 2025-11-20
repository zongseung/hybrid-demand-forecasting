.PHONY: help install build up down restart logs clean test train inference deploy

help: ## Show this help message
	@echo "Open-STEF Makefile Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies with uv (ultra-fast!)
	@echo "📦 Installing dependencies with uv..."
	uv sync

build: ## Build Docker images
	@echo "🏗️  Building Docker images..."
	docker-compose build

up: ## Start all services
	@echo "🚀 Starting all services..."
	docker-compose up -d
	@echo ""
	@echo "✅ Services started!"
	@echo ""
	@echo "📊 Access points:"
	@echo "  - Prefect UI:  http://localhost:14200"
	@echo "  - FastAPI:     http://localhost:18000"
	@echo "  - MLflow:      http://localhost:15000"
	@echo "  - Grafana:     http://localhost:13000 (admin/admin)"
	@echo "  - InfluxDB:    http://localhost:18086"
	@echo "  - PostgreSQL:  localhost:15432"

down: ## Stop all services
	@echo "🛑 Stopping all services..."
	docker-compose down

restart: down up ## Restart all services

logs: ## Show logs from all services
	docker-compose logs -f

logs-agent: ## Show logs from Prefect agent
	docker-compose logs -f prefect-agent

logs-api: ## Show logs from FastAPI
	docker-compose logs -f fastapi

clean: ## Remove all containers, volumes, and images
	@echo "🧹 Cleaning up..."
	docker-compose down -v
	docker system prune -f

clean-all: clean ## Remove everything including images
	docker-compose down -v --rmi all
	docker system prune -af

test: ## Run tests
	@echo "🧪 Running tests..."
	uv run pytest tests/ -v

train: ## Run training pipeline (quick test)
	@echo "🎓 Training models (quick test)..."
	uv run python flows/weekly_retrain_hybrid.py \
		--n_lstm_iter 5 \
		--lstm_epochs 10

train-prod: ## Run training pipeline (production)
	@echo "🎓 Training models (production)..."
	uv run python flows/weekly_retrain_hybrid.py \
		--n_lstm_iter 50 \
		--lstm_epochs 100

inference: ## Run inference demo
	@echo "🔮 Running inference..."
	uv run python inference_demo.py

deploy: ## Deploy weekly retraining to Prefect
	@echo "📅 Deploying weekly retraining schedule..."
	uv run python deploy_weekly_retrain.py

status: ## Check service status
	@echo "📊 Service Status:"
	@docker-compose ps

mlflow-ui: ## Open MLflow UI in browser
	@echo "🔬 Opening MLflow UI..."
	@open http://localhost:15000 || xdg-open http://localhost:15000 || echo "Please open http://localhost:15000 in your browser"

grafana-ui: ## Open Grafana UI in browser
	@echo "📊 Opening Grafana UI..."
	@open http://localhost:13000 || xdg-open http://localhost:13000 || echo "Please open http://localhost:13000 in your browser"

prefect-ui: ## Open Prefect UI in browser
	@echo "⚡ Opening Prefect UI..."
	@open http://localhost:14200 || xdg-open http://localhost:14200 || echo "Please open http://localhost:14200 in your browser"

# Development helpers
dev-install: ## Install development dependencies
	@echo "📦 Installing dev dependencies..."
	uv sync --all-extras

format: ## Format code with black
	@echo "✨ Formatting code..."
	uv run black .

lint: ## Lint code with flake8
	@echo "🔍 Linting code..."
	uv run flake8 src/ flows/ api/

check: format lint ## Format and lint code

# Database helpers
db-reset: ## Reset database
	@echo "🗄️  Resetting database..."
	docker-compose down postgres
	docker volume rm open-stef_postgres_data
	docker-compose up -d postgres

db-shell: ## Open PostgreSQL shell
	@echo "🗄️  Opening database shell..."
	docker-compose exec postgres psql -U postgres -d demand_forecasting

# Quick start
quickstart: install build up ## Quick start (install + build + up)
	@echo ""
	@echo "🎉 Quick start complete!"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Train models:   make train"
	@echo "  2. Run inference:  make inference"
	@echo "  3. Deploy:         make deploy"
	@echo ""
	@echo "📖 Documentation: README_HYBRID_FORECASTING.md"
