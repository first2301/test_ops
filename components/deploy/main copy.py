from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
import joblib
import pandas as pd
from typing import List, Any, Optional
import os

app = FastAPI()

# 모델 경로 및 상수 정의
MODEL_PATH = './model.pkl'
# 실제 학습 시 숫자형 컬럼만 사용 (구분 제외)
EXPECTED_FEATURES = ['번지_숫자', '시도_encoded', '시군구_encoded', '읍면동_encoded', '구분_encoded']
TARGET_NAMES = ['폐업', '영업/정상']
# 모델 로딩 (X_test는 선택적)
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    print(f"⚠️ 모델 로딩 실패: {e}")

# Pydantic 모델 정의
class InputData(BaseModel):
    """단일 예측 레코드 - 모델은 숫자형 컬럼만 사용 (학습 시 numeric_cols만 사용)"""
    번지_숫자: int = Field(..., ge=0, description="번지에서 추출한 숫자 값", example=3)
    시도_encoded: int = Field(..., ge=0, description="시도 인코딩 값", example=3)
    시군구_encoded: int = Field(..., ge=0, description="시군구 인코딩 값", example=32)
    읍면동_encoded: int = Field(..., ge=0, description="읍면동 인코딩 값", example=10870)
    구분_encoded: int = Field(..., ge=0, description="음식점 구분 인코딩 값", example=21)
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "번지_숫자": 3,
                "시도_encoded": 3,
                "시군구_encoded": 32,
                "읍면동_encoded": 10870,
                "구분_encoded": 21
            }
        }
    )


class PredictRequest(BaseModel):
    """외부 입력 데이터 예측 요청"""
    data: List[InputData] = Field(..., min_items=1, description="예측할 데이터 리스트")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "data": [
                    {
                        "번지_숫자": 3,
                        "시도_encoded": 3,
                        "시군구_encoded": 32,
                        "읍면동_encoded": 10870,
                        "구분_encoded": 21
                    },
                    {
                        "번지_숫자": 11,
                        "시도_encoded": 17,
                        "시군구_encoded": 190,
                        "읍면동_encoded": 14085,
                        "구분_encoded": 3
                    }
                ]
            }
        }
    )


class PredictResponse(BaseModel):
    """예측 응답"""
    predictions: List[int]
    predictions_label: List[str]


# 엔드포인트 정의
@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="음식점 폐업 예측",
    description="""
    외부에서 들어오는 값으로 음식점 폐업 여부를 예측합니다.
    
    **입력 필수 필드:**
    - `번지_숫자`: 번지에서 추출한 숫자 값 (0 이상)
    - `시도_encoded`: 시도 인코딩 값 (0 이상)
    - `시군구_encoded`: 시군구 인코딩 값 (0 이상)
    - `읍면동_encoded`: 읍면동 인코딩 값 (0 이상)
    - `구분_encoded`: 음식점 구분 인코딩 값 (0 이상)
    
    **응답:**
    - `predictions`: 예측 결과 (0: 폐업, 1: 영업/정상)
    - `predictions_label`: 예측 결과 한글 라벨
    """,
    response_description="예측 결과가 포함된 응답"
)
def predict(request: PredictRequest):
    """
    외부에서 들어오는 값으로 예측 수행
    
    **예시 요청:**
    ```json
    {
        "data": [
            {
                "번지_숫자": 3,
                "시도_encoded": 3,
                "시군구_encoded": 32,
                "읍면동_encoded": 10870,
                "구분_encoded": 21
            }
        ]
    }
    ```
    
    **예시 응답:**
    ```json
    {
        "predictions": [1],
        "predictions_label": ["영업/정상"]
    }
    ```
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please check model file.")
    
    try:
        # List[InputData]를 DataFrame으로 변환
        data_dicts = [item.model_dump() for item in request.data]
        input_df = pd.DataFrame(data_dicts)
        
        # 필수 컬럼 검증
        missing_features = set(EXPECTED_FEATURES) - set(input_df.columns)
        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required features: {list(missing_features)}"
            )
        
        # 필요한 컬럼만 선택 (순서 보장)
        input_df = input_df[EXPECTED_FEATURES]
        
        # 학습 시와 동일하게 float64로 변환 (모델이 기대하는 타입)
        input_df = input_df.astype('float64')
        
        # 예측 수행
        preds = model.predict(input_df)
        preds_list = preds.tolist() if hasattr(preds, 'tolist') else list(preds)
        
        # 클래스명으로 변환
        preds_label = [TARGET_NAMES[int(pred)] for pred in preds_list]
        
        return PredictResponse(
            predictions=preds_list,
            predictions_label=preds_label
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

