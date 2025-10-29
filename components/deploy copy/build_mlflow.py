#!/usr/bin/env python3
"""
MLflow Docker 이미지 빌드 스크립트
사용법: python build_mlflow.py
"""
import mlflow
import os

# 환경변수 설정
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# 모델 ID 설정
MODEL_ID = os.getenv("MLFLOW_MODEL_ID", "lgbm_classifier")
MODEL_STAGE = os.getenv("MLFLOW_MODEL_STAGE", "latest")  # latest, production 등

model_uri = f"models:/{MODEL_ID}/{MODEL_STAGE}"
image_name = f"restaurant-predictor-{MODEL_ID}:{MODEL_STAGE}"

print(f"🚀 Building Docker image for model: {model_uri}")
print(f"   Image name: {image_name}")

# Docker 이미지 빌드 (MLServer 미사용 - Docker Compose 환경용)
mlflow.models.build_docker(
    model_uri=model_uri,
    name=image_name,
    enable_mlserver=False  # MLServer 사용 안 함 (Docker Compose 환경)
)

print(f"✅ Docker image built: {image_name}")
print(f"\n사용법:")
print(f"  docker run -p 5001:8080 {image_name}")

