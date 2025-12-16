"""
음식점 폐업 예측 서비스
BentoML을 사용하여 음식점 폐업 여부 예측 모델을 서비스화합니다.
"""
import bentoml
import os
import joblib
import pandas as pd
import re
from typing import List, Dict, Any
from pydantic import BaseModel, Field

# 환경변수로 모델 경로 설정 (기본값: 로컬 파일)
MODEL_PATH = os.getenv("MODEL_PATH", "./model.pkl")
ENCODER_PATH = os.getenv("ENCODER_PATH", "./encoders.pkl")
# BentoML 모델 태그 (환경변수로 override 가능)
BENTOML_MODEL_TAG = os.getenv("BENTOML_MODEL_TAG", "latest")

# 실제 학습 시 숫자형 컬럼만 사용
EXPECTED_FEATURES = ['번지_숫자', '시도_encoded', '시군구_encoded', '읍면동_encoded', '구분_encoded']
TARGET_NAMES = ['폐업', '영업/정상']


def extract_lot_number(lot_str: str) -> int:
    """
    번지에서 숫자를 추출합니다.
    
    Args:
        lot_str: 번지 주소 문자열 (예: '3', '11', '169-3')
        
    Returns:
        추출된 숫자 (예: '3' -> 3, '11-1' -> 11, '169-3' -> 169)
        숫자가 없거나 빈 값인 경우 0 반환
    """
    if pd.isna(lot_str) or lot_str == '' or lot_str == 'nan':
        return 0
    # 첫 번째 숫자 추출
    match = re.search(r'\d+', str(lot_str))
    return int(match.group()) if match else 0


# Pydantic 모델 정의 (입력 검증용)
class InputData(BaseModel):
    """단일 예측 레코드 - 원본 문자열 값을 받아서 인코딩"""
    번지: str = Field(..., description="번지 주소 (예: '3', '11-1', '169-3')")
    시도: str = Field(..., description="시도명 (예: '서울특별시', '부산광역시')")
    시군구: str = Field(..., description="시군구명 (예: '강남구', '해운대구')")
    읍면동: str = Field(..., description="읍면동명 (예: '역삼동', '삼성동')")
    구분: str = Field(..., description="음식점 구분 (예: '한식', '중식', '일식')")


@bentoml.service(
    image=bentoml.images.Image(python_version="3.11").python_packages(
        "pandas", "scikit-learn", "lightgbm"
    ),
)
class RestaurantClassifier:
    """
    음식점 폐업 예측 서비스 클래스
    
    학습된 모델과 인코더를 로드하여 음식점의 폐업 여부를 예측합니다.
    원본 주소 및 음식점 정보를 입력받아 전처리 후 예측을 수행합니다.
    """
    
    def __init__(self) -> None:
        """
        모델 및 인코더 초기화 및 로드
        
        로드 우선순위:
        1. BentoML 모델 저장소 (BENTOML_MODEL_TAG 환경변수 설정 시)
        2. 로컬 파일 (joblib으로 저장된 .pkl 파일)
        
        Raises:
            FileNotFoundError: 모델 파일을 찾을 수 없는 경우
            Exception: 모델 로드 중 오류 발생 시
        """
        # BentoML 모델 저장소에서 로드 시도
        if BENTOML_MODEL_TAG:
            try:
                model_ref = bentoml.models.get(BENTOML_MODEL_TAG)
                self.model = joblib.load(model_ref.path_of("model.pkl"))
                self.encoders = joblib.load(model_ref.path_of("encoders.pkl"))
                print(f"✅ BentoML 모델 저장소에서 로드 성공: {BENTOML_MODEL_TAG}")
                print(f"   인코더 키: {list(self.encoders.keys())}")
                return
            except Exception as e:
                print(f"⚠️ BentoML 모델 저장소 로드 실패: {e}, 파일 로드 시도...")
        
        # 로컬 파일에서 로드 (fallback)
        try:
            if os.path.exists(MODEL_PATH):
                self.model = joblib.load(MODEL_PATH)
                print(f"✅ 모델 로딩 성공: {MODEL_PATH}")
            else:
                raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
            
            if os.path.exists(ENCODER_PATH):
                self.encoders = joblib.load(ENCODER_PATH)
                print(f"✅ 인코더 로딩 성공: {ENCODER_PATH}")
                print(f"   인코더 키: {list(self.encoders.keys())}")
            else:
                raise FileNotFoundError(f"인코더 파일을 찾을 수 없습니다: {ENCODER_PATH}")
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            raise
    
    def _preprocess(self, data: List[Dict[str, str]]) -> pd.DataFrame:
        """
        입력 데이터 전처리
        
        Args:
            data: 예측할 데이터 리스트 (Dict 형태)
            
        Returns:
            전처리된 DataFrame (EXPECTED_FEATURES 컬럼 포함)
            
        Raises:
            ValueError: 필수 컬럼이 없거나 인코딩 실패 시
        """
        # Dict 리스트를 DataFrame으로 변환
        input_df = pd.DataFrame(data)
        
        # 필수 컬럼 확인
        required_cols = ['번지', '시도', '시군구', '읍면동', '구분']
        missing_cols = [col for col in required_cols if col not in input_df.columns]
        if missing_cols:
            raise ValueError(f"필수 컬럼이 없습니다: {missing_cols}")
        
        # 번지에서 숫자 추출
        input_df['번지_숫자'] = input_df['번지'].apply(extract_lot_number)
        
        # 인코더를 사용하여 문자열 값 인코딩
        # NaN 값은 '기타'로 채우기 (학습 시와 동일하게)
        encode_cols = ['시도', '시군구', '읍면동', '구분']
        
        for col in encode_cols:
            # NaN 값을 '기타'로 채우기
            input_df[col] = input_df[col].fillna('기타')
            
            # 인코더로 변환
            if col in self.encoders:
                encoder = self.encoders[col]
                try:
                    # 학습 시 본 적 없는 값이면 최대값+1로 처리
                    input_df[f'{col}_encoded'] = input_df[col].apply(
                        lambda x: encoder.transform([x])[0] if x in encoder.classes_ else len(encoder.classes_)
                    )
                except Exception as e:
                    raise ValueError(
                        f"컬럼 '{col}' 인코딩 실패: {str(e)}. 값: {input_df[col].tolist()}"
                    )
            else:
                raise ValueError(f"인코더 '{col}'를 찾을 수 없습니다.")
        
        # 필요한 컬럼만 선택 (순서 보장)
        input_df = input_df[EXPECTED_FEATURES]
        
        # 학습 시와 동일하게 float64로 변환 (모델이 기대하는 타입)
        input_df = input_df.astype('float64')
        
        return input_df
    
    @bentoml.api(batchable=False)
    def predict(
        self, 
        data: List[InputData]
    ) -> Dict[str, List]:
        """
        음식점 폐업 예측 수행
        
        원본 주소 및 음식점 정보를 입력받아 폐업 여부를 예측합니다.
        
        Args:
            data: 예측할 데이터 리스트 (InputData Pydantic 모델)
                - 번지: 번지 주소 문자열 (예: '3', '11-1', '169-3')
                - 시도: 시도명 (예: '서울특별시', '부산광역시')
                - 시군구: 시군구명 (예: '강남구', '해운대구')
                - 읍면동: 읍면동명 (예: '역삼동', '삼성동')
                - 구분: 음식점 구분 (예: '한식', '중식', '일식')
        
        Returns:
            예측 결과 딕셔너리:
            - predictions: 예측 결과 리스트 (0: 폐업, 1: 영업/정상)
            - predictions_label: 예측 결과 한글 라벨 리스트
            
        Raises:
            ValueError: 입력 데이터가 비어있거나 필수 필드가 없는 경우
            RuntimeError: 모델 또는 인코더가 로드되지 않은 경우
        """
        if not data:
            raise ValueError("입력 데이터 리스트가 비어있습니다.")
        
        if not hasattr(self, 'model') or self.model is None:
            raise RuntimeError("모델이 로드되지 않았습니다. 모델 파일을 확인해주세요.")
        
        if not hasattr(self, 'encoders') or self.encoders is None:
            raise RuntimeError("인코더가 로드되지 않았습니다. 인코더 파일을 확인해주세요.")
        
        try:
            # Pydantic 모델을 Dict로 변환
            data_dicts = [item.model_dump() for item in data]
            
            # 전처리 수행
            input_df = self._preprocess(data_dicts)
            
            # 예측 수행
            preds = self.model.predict(input_df)
            preds_list = preds.tolist() if hasattr(preds, 'tolist') else list(preds)
            
            # 클래스명으로 변환
            preds_label = [TARGET_NAMES[int(pred)] for pred in preds_list]
            
            return {
                "predictions": preds_list,
                "predictions_label": preds_label
            }
            
        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(f"예측 실패: {str(e)}")
