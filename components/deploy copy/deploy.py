# service.py
import bentoml
import numpy as np
import pandas as pd
from bentoml.io import PandasDataFrame

TARGET_NAMES = ['폐업', '영업/정상']
EXPECTED_FEATURES = ['구분','번지_숫자','시도_encoded','시군구_encoded','읍면동_encoded','구분_encoded']

# 1. 모델 Runner 생성
model_runner = bentoml.mlflow.get("lgbm_classifier:latest").to_runner()

# 2. 서비스 정의 (Runner 등록)
svc = bentoml.Service("lgbm_classifier", runners=[model_runner])

# 3. API 엔드포인트 정의
@svc.api(input=PandasDataFrame(), output=bentoml.io.NumpyNdarray())
async def predict(input_df: pd.DataFrame) -> np.ndarray:
    missing = set(EXPECTED_FEATURES) - set(input_df.columns)
    if missing:
        raise ValueError(f"Missing features: {missing}")
    # 4. 비동기 추론 수행
    result = await model_runner.predict.async_run(input_df[EXPECTED_FEATURES])
    # 5. 결과를 한글 클래스 이름으로 매핑하여 반환
    return np.array([TARGET_NAMES[int(p)] for p in result])