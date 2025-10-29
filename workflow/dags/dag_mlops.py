from airflow import DAG
from airflow.decorators import task
from datetime import datetime
import joblib
import pandas as pd

default_args = {
    'start_date': datetime(2023, 1, 1),
    'catchup': False
}

with DAG(
    dag_id='ml_sample_predict',
    schedule_interval=None,
    default_args=default_args,
    catchup=False,
    tags=['mlops']
) as dag:

    @task()
    def predict_samples():
        # X_test 임시 로드 (예시 - 실제 경로/방식 맞게 수정)
        X_test = pd.read_csv('./X_test.csv')
        
        # 모델 불러오기
        model = joblib.load('./model.pkl')
        
        # 샘플 예측
        sample_X = X_test.iloc[:5]
        preds = model.predict(sample_X)
        print("샘플 예측 결과:", preds)
    
    predict_samples()