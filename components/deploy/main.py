from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
import joblib
import pandas as pd
from typing import List, Any, Optional
import os
import re

app = FastAPI()

# 모델 경로 및 상수 정의
MODEL_PATH = './model.pkl'
ENCODER_PATH = './encoders.pkl'
# 실제 학습 시 숫자형 컬럼만 사용
EXPECTED_FEATURES = ['번지_숫자', '시도_encoded', '시군구_encoded', '읍면동_encoded', '구분_encoded']
TARGET_NAMES = ['폐업', '영업/정상']

# 번지 숫자 추출 함수
def extract_lot_number(lot_str: str) -> int:
    """번지에서 숫자를 추출합니다 (예: '3', '11', '169-3' -> 3, 11, 169)"""
    if pd.isna(lot_str) or lot_str == '' or lot_str == 'nan':
        return 0
    # 첫 번째 숫자 추출
    match = re.search(r'\d+', str(lot_str))
    return int(match.group()) if match else 0

# 모델 및 인코더 로딩
try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ 모델 로딩 성공: {MODEL_PATH}")
except Exception as e:
    model = None
    print(f"⚠️ 모델 로딩 실패: {e}")

try:
    encoders = joblib.load(ENCODER_PATH)
    print(f"✅ 인코더 로딩 성공: {ENCODER_PATH}")
    print(f"   인코더 키: {list(encoders.keys())}")
except Exception as e:
    encoders = None
    print(f"⚠️ 인코더 로딩 실패: {e}")

# Pydantic 모델 정의
class InputData(BaseModel):
    """단일 예측 레코드 - 원본 문자열 값을 받아서 인코딩"""
    번지: str = Field(..., description="번지 주소 (예: '3', '11-1', '169-3')", example="3")
    시도: str = Field(..., description="시도명 (예: '서울특별시', '부산광역시')", example="서울특별시")
    시군구: str = Field(..., description="시군구명 (예: '강남구', '해운대구')", example="강남구")
    읍면동: str = Field(..., description="읍면동명 (예: '역삼동', '삼성동')", example="역삼동")
    구분: str = Field(..., description="음식점 구분 (예: '한식', '중식', '일식')", example="한식")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "번지": "3",
                "시도": "서울특별시",
                "시군구": "강남구",
                "읍면동": "역삼동",
                "구분": "한식"
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
                        "번지": "3",
                        "시도": "서울특별시",
                        "시군구": "강남구",
                        "읍면동": "역삼동",
                        "구분": "한식"
                    },
                    {
                        "번지": "11-1",
                        "시도": "부산광역시",
                        "시군구": "해운대구",
                        "읍면동": "우동",
                        "구분": "일식"
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
    원본 주소 및 음식점 정보를 입력받아 폐업 여부를 예측합니다.
    
    **입력 필수 필드:**
    - `번지`: 번지 주소 문자열 (예: '3', '11-1', '169-3')
    - `시도`: 시도명 (예: '서울특별시', '부산광역시')
    - `시군구`: 시군구명 (예: '강남구', '해운대구')
    - `읍면동`: 읍면동명 (예: '역삼동', '삼성동')
    - `구분`: 음식점 구분 (예: '한식', '중식', '일식')
    
    **응답:**
    - `predictions`: 예측 결과 (0: 폐업, 1: 영업/정상)
    - `predictions_label`: 예측 결과 한글 라벨
    """,
    response_description="예측 결과가 포함된 응답"
)
def predict(request: PredictRequest):
    """
    외부에서 들어오는 원본 문자열 값으로 예측 수행
    
    **예시 요청:**
    ```json
    {
        "data": [
            {
                "번지": "3",
                "시도": "서울특별시",
                "시군구": "강남구",
                "읍면동": "역삼동",
                "구분": "한식"
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
    
    if encoders is None:
        raise HTTPException(status_code=500, detail="Encoders not loaded. Please check encoder file.")
    
    try:
        # List[InputData]를 DataFrame으로 변환
        data_dicts = [item.model_dump() for item in request.data]
        input_df = pd.DataFrame(data_dicts)
        
        # 번지에서 숫자 추출
        input_df['번지_숫자'] = input_df['번지'].apply(extract_lot_number)
        
        # 인코더를 사용하여 문자열 값 인코딩
        # NaN 값은 '기타'로 채우기 (학습 시와 동일하게)
        encode_cols = ['시도', '시군구', '읍면동', '구분']
        
        for col in encode_cols:
            if col not in input_df.columns:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required column: {col}"
                )
            
            # NaN 값을 '기타'로 채우기
            input_df[col] = input_df[col].fillna('기타')
            
            # 인코더로 변환 (없는 값은 -1 또는 경고 발생 가능)
            if col in encoders:
                encoder = encoders[col]
                try:
                    # 학습 시 본 적 없는 값이면 -1 반환하거나 최대값+1로 처리
                    input_df[f'{col}_encoded'] = input_df[col].apply(
                        lambda x: encoder.transform([x])[0] if x in encoder.classes_ else len(encoder.classes_)
                    )
                except Exception as e:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Encoding failed for column '{col}': {str(e)}. Value: {input_df[col].tolist()}"
                    )
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Encoder for '{col}' not found in encoders."
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


@app.get("/health")
def health():
    """헬스 체크"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "encoders_loaded": encoders is not None,
        "encoder_keys": list(encoders.keys()) if encoders else None
    }