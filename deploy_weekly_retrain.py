"""
Deploy Weekly Retraining Flow to Prefect
Schedules the flow to run every Sunday at 2:00 AM
"""

from prefect.client.schemas.schedules import CronSchedule
from flows.weekly_retrain_hybrid import weekly_retrain_flow


def deploy_weekly_retrain():
    """
    Deploy weekly retraining flow with cron schedule
    Runs every Sunday at 2:00 AM (Asia/Seoul)
    """

    deployment = weekly_retrain_flow.deploy(
        name="weekly-hybrid-retrain-sunday",
        work_pool_name="weekly-docker",
        image="open-stef-fastapi:latest",
        build=False,
        push=False,
        schedule=CronSchedule(
            cron="0 2 * * 0",  # Every Sunday at 2:00 AM
            timezone="Asia/Seoul"
        ),
        parameters={
            "csv_path": "/mnt/nvme/tilting/power_demand_final.csv",
            "model_dir": "models/production",
            "window_size": 168,  # 7 days
            "horizon": 24,       # 24 hours
            "n_lstm_iter": 50,   # 50 random search iterations
            "lstm_epochs": 100   # 100 epochs max per trial
        },
        description=(
            "Weekly retraining of hybrid power demand forecasting model "
            "(trend + fourier + LSTM). Runs every Sunday at 2:00 AM KST."
        ),
        tags=["weekly", "retrain", "hybrid", "mlflow", "production"],
    )

    print("\n" + "=" * 80)
    print("✅ DEPLOYMENT SUCCESSFUL")
    print("=" * 80)
    print(f"Deployment ID: {deployment.id}")
    print(f"Schedule: Every Sunday at 2:00 AM (Asia/Seoul)")
    print(f"Flow: weekly_retrain_flow")
    print(f"Work Pool: weekly-process")
    print("Next steps: run `prefect worker start --pool weekly-process`")
    print("=" * 80)

    return deployment.id


if __name__ == "__main__":
    deploy_weekly_retrain()

