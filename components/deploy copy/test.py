import mlflow

mlflow.models.build_docker(
    model_uri=f"models:/lgbm_classifier/latest",
    name="lgbm_classifier",
    enable_mlserver=True,
)