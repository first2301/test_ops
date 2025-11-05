import bentoml
import numpy as np
import pandas as pd
import mlflow
import os
from typing import List, Dict, Any
import logging
import socket
import time

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 모델 설정
TARGET_NAMES = ['폐업', '영업/정상']
EXPECTED_FEATURES = [
    '구분', 
    '번지_숫자', 
    '시도_encoded', 
    '시군구_encoded', 
    '읍면동_encoded', 
    '구분_encoded'
]

# MLflow Tracking URI 설정 (동적 IP 해석)
def get_mlflow_uri():
    """MLflow 서버 URI를 동적으로 가져옵니다 (IP 해석 포함)"""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri and tracking_uri.startswith("http://mlflow-server"):
        try:
            # 호스트명을 IP로 해석
            mlflow_ip = socket.gethostbyname("mlflow-server")
            return f"http://{mlflow_ip}:5000"
        except socket.gaierror as e:
            logger.warning(f"DNS 해석 실패: {e}, 원래 URI 사용")
            return tracking_uri
    return tracking_uri or "http://mlflow-server:5000"

MLFLOW_TRACKING_URI = get_mlflow_uri()
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
logger.info(f"MLflow Tracking URI: {MLFLOW_TRACKING_URI}")

# 모델 URI 설정 (환경변수로 override 가능)
MODEL_URI = os.getenv(
    "MLFLOW_MODEL_URI",
    "models:/lgbm_classifier/latest"  # MLflow UI에서 확인한 모델명
)
logger.info(f"모델 URI: {MODEL_URI}")

@bentoml.service(
    traffic={"timeout": 30},
    resources={"cpu": "2", "memory": "4Gi"}
)
class Classifier:
    def __init__(self):
        """모델 초기화 및 로드 (재시도 로직 포함)"""
        max_retries = 5
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                logger.info(f"모델 로딩 시도 {attempt + 1}/{max_retries}...")
                logger.info(f"Loading model from: {MODEL_URI}")
                self.model = mlflow.pyfunc.load_model(MODEL_URI)
                logger.info("✅ Model loaded successfully")
                
                # 기대하는 입력 형태 확인
                if hasattr(self.model, 'metadata') and hasattr(self.model.metadata, 'signature'):
                    logger.info(f"Model signature: {self.model.metadata.signature}")
                return
                
            except Exception as e:
                logger.warning(f"모델 로딩 실패 (시도 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    logger.info(f"{retry_delay}초 후 재시도...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"❌ 모델 로딩 실패 (최대 재시도 횟수 초과)")
                    raise

    @bentoml.api
    def predict(
        self, 
        data: List[Dict[str, Any]]
    ) -> List[str]:
        """
        음식점 폐업 예측
        
        Args:
            data: [{"구분": "한식", "번지_숫자": 3, "시도_encoded": 3, ...}, ...]
            
        Returns:
            예측 클래스 리스트: ["폐업", "영업/정상", ...]
        """
        try:
            # Dict 리스트를 DataFrame으로 변환
            df = pd.DataFrame(data)
            
            # 필수 피처 확인
            missing_features = set(EXPECTED_FEATURES) - set(df.columns)
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
            
            # 예측
            preds = self.model.predict(df[EXPECTED_FEATURES])
            
            # 클래스명으로 변환
            results = [TARGET_NAMES[int(pred)] for pred in preds]
            logger.info(f"✅ Predicted {len(results)} samples")
            return results
            
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}")
            raise bentoml.exceptions.BentoServiceException(f"Prediction failed: {str(e)}")

    @bentoml.api
    def predict_proba(
        self, 
        data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        음식점 폐업 예측 확률 반환
        
        Args:
            data: [{"구분": "한식", "번지_숫자": 3, ...}, ...]
            
        Returns:
            {
                "predictions": ["폐업", "영업/정상", ...],
                "probabilities": [[0.3, 0.7], [0.8, 0.2], ...]
            }
        """
        try:
            df = pd.DataFrame(data)
            
            # 필수 피처 확인
            missing_features = set(EXPECTED_FEATURES) - set(df.columns)
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
            
            # 예측 및 확률
            preds = self.model.predict(df[EXPECTED_FEATURES])
            probas = self.model.predict_proba(df[EXPECTED_FEATURES])
            
            # 클래스명 변환
            predictions = [TARGET_NAMES[int(pred)] for pred in preds]
            
            # 확률 반환 [폐업 확률, 영업 확률]
            probabilities = [[float(proba[0]), float(proba[1])] for proba in probas]
            
            return {
                "predictions": predictions,
                "probabilities": probabilities
            }
            
        except Exception as e:
            logger.error(f"❌ Prediction probability error: {e}")
            raise bentoml.exceptions.BentoServiceException(f"Prediction failed: {str(e)}")

    @bentoml.api
    def health(self) -> Dict[str, Any]:
        """헬스 체크"""
        return {
            "status": "healthy",
            "model_uri": MODEL_URI,
            "mlflow_uri": MLFLOW_TRACKING_URI,
            "features": EXPECTED_FEATURES,
            "target_names": TARGET_NAMES
        }