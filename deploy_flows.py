"""
Script to deploy Prefect flows with schedules
"""
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule
from flows.data_ingestion import hourly_data_ingestion_flow
from flows.daily_forecast import daily_forecast_flow
from flows.weekly_retrain import weekly_model_retrain_flow


def deploy_all_flows():
    """Deploy all flows with their schedules"""
    
    # 1. Hourly Data Ingestion (every hour at :00)
    hourly_deployment = Deployment.build_from_flow(
        flow=hourly_data_ingestion_flow,
        name="hourly-data-ingestion",
        work_queue_name="data-queue",
        schedule=CronSchedule(cron="0 * * * *"),  # Every hour
        tags=["data-ingestion", "production"],
        description="Fetch demand, weather, and calendar data every hour"
    )
    hourly_deployment.apply()
    print("✅ Deployed: hourly-data-ingestion (Cron: 0 * * * *)")
    
    # 2. Daily Forecast (every day at 2:00 AM)
    daily_deployment = Deployment.build_from_flow(
        flow=daily_forecast_flow,
        name="daily-forecast",
        work_queue_name="model-queue",
        schedule=CronSchedule(cron="0 2 * * *"),  # Daily at 2 AM
        tags=["forecast", "inference", "production"],
        description="Generate daily demand forecasts for next 7 days"
    )
    daily_deployment.apply()
    print("✅ Deployed: daily-forecast (Cron: 0 2 * * *)")
    
    # 3. Weekly Model Retrain (every Sunday at 3:00 AM)
    weekly_deployment = Deployment.build_from_flow(
        flow=weekly_model_retrain_flow,
        name="weekly-retrain",
        work_queue_name="training-queue",
        schedule=CronSchedule(cron="0 3 * * 0"),  # Sunday at 3 AM
        tags=["training", "retraining", "production"],
        description="Retrain all model components weekly and deploy if improved"
    )
    weekly_deployment.apply()
    print("✅ Deployed: weekly-retrain (Cron: 0 3 * * 0)")
    
    print("\n🎉 All flows deployed successfully!")
    print("\nTo start the Prefect agent, run:")
    print("  prefect agent start -q data-queue")
    print("  prefect agent start -q model-queue")
    print("  prefect agent start -q training-queue")
    print("\nOr start a default agent for all queues:")
    print("  prefect agent start -q default")


if __name__ == "__main__":
    deploy_all_flows()



