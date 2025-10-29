from airflow import DAG
from airflow.decorators import task
from datetime import datetime, timedelta
import pandas as pd
from io import BytesIO
from minio import Minio
from minio.error import S3Error
import logging

logger = logging.getLogger(__name__)

default_args = {
    'owner': 'dataops',
    'start_date': datetime(2023, 1, 1),
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'catchup': False
}

MINIO_ENDPOINT = "minio:9000"
MINIO_ACCESS_KEY = "minio"
MINIO_SECRET_KEY = "minio123"
RAW_BUCKET = "raw"
PREPRO_BUCKET = "prepro"
RAW_OBJECT = "restaurant_2020.csv"
PREPRO_OBJECT = "restaurant_2020_prepro.csv"

def get_minio_client():
    """MinIO 클라이언트 생성 및 반환"""
    try:
        client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=False
        )
        logger.info(f"MinIO 클라이언트 초기화 완료: {MINIO_ENDPOINT}")
        return client
    except Exception as e:
        logger.error(f"MinIO 클라이언트 초기화 실패: {str(e)}")
        raise

with DAG(
    dag_id='dataops_preprocess_minio',
    default_args=default_args,
    description='MinIO에서 원본 데이터를 읽어 전처리 후 저장하는 데이터 파이프라인',
    schedule_interval=None,
    catchup=False,
    tags=['dataops', 'minio', 'preprocessing']
) as dag:

    @task()
    def fetch_raw_data():
        """MinIO에서 원본 데이터 읽기"""
        try:
            logger.info(f"원본 데이터 읽기 시작: {RAW_BUCKET}/{RAW_OBJECT}")
            client = get_minio_client()
            
            # 버킷 존재 확인
            if not client.bucket_exists(RAW_BUCKET):
                raise ValueError(f"버킷이 존재하지 않습니다: {RAW_BUCKET}")
            
            # 객체 존재 확인
            try:
                client.stat_object(RAW_BUCKET, RAW_OBJECT)
            except S3Error as e:
                if e.code == 'NoSuchKey':
                    raise FileNotFoundError(f"객체를 찾을 수 없습니다: {RAW_BUCKET}/{RAW_OBJECT}")
                raise
            
            # 데이터 읽기
            response = client.get_object(RAW_BUCKET, RAW_OBJECT)
            df = pd.read_csv(BytesIO(response.read()))
            response.close()
            response.release_conn()
            
            logger.info(f"데이터 읽기 완료: {len(df)}행, {len(df.columns)}열")
            logger.info(f"데이터 샘플:\n{df.head()}")
            
            # 데이터 메타데이터 반환 (XCom 전달)
            return {
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': df.columns.tolist()
            }
            
        except Exception as e:
            logger.error(f"원본 데이터 읽기 실패: {str(e)}")
            raise
    
    @task()
    def preprocess_data(data_meta):
        """데이터 전처리 및 MinIO에 저장"""
        try:
            logger.info(f"전처리 시작: {data_meta}")
            client = get_minio_client()
            
            # 원본 데이터 다시 읽기 (실제 전처리 수행)
            logger.info(f"전처리를 위해 데이터 재읽기: {RAW_BUCKET}/{RAW_OBJECT}")
            response = client.get_object(RAW_BUCKET, RAW_OBJECT)
            df = pd.read_csv(BytesIO(response.read()))
            response.close()
            response.release_conn()
            
            logger.info(f"전처리 전 데이터 정보:")
            logger.info(f"  - 행 수: {len(df)}")
            logger.info(f"  - 결측치 수: {df.isnull().sum().sum()}")
            
            # 전처리 수행: 결측치 0으로 대체
            df_prepro = df.fillna(0)
            
            logger.info(f"전처리 후 데이터 정보:")
            logger.info(f"  - 행 수: {len(df_prepro)}")
            logger.info(f"  - 결측치 수: {df_prepro.isnull().sum().sum()}")
            
            # 전처리 버킷 존재 확인, 없으면 생성
            if not client.bucket_exists(PREPRO_BUCKET):
                logger.info(f"버킷 생성: {PREPRO_BUCKET}")
                client.make_bucket(PREPRO_BUCKET)
            
            # MinIO에 저장
            out_buffer = BytesIO()
            df_prepro.to_csv(out_buffer, index=False)
            out_buffer.seek(0)
            
            client.put_object(
                PREPRO_BUCKET,
                PREPRO_OBJECT,
                data=out_buffer,
                length=out_buffer.getbuffer().nbytes,
                content_type='application/csv'
            )
            
            logger.info(f"전처리 데이터 저장 완료: {PREPRO_BUCKET}/{PREPRO_OBJECT}")
            
            return {
                'rows': len(df_prepro),
                'columns': len(df_prepro.columns),
                'output_path': f"{PREPRO_BUCKET}/{PREPRO_OBJECT}"
            }
            
        except Exception as e:
            logger.error(f"전처리 실패: {str(e)}")
            raise
    
    @task()
    def validate_output(output_meta):
        """전처리 결과 검증"""
        try:
            logger.info(f"결과 검증 시작: {output_meta}")
            client = get_minio_client()
            
            # 저장된 객체 확인
            stat = client.stat_object(PREPRO_BUCKET, PREPRO_OBJECT)
            logger.info(f"저장된 객체 정보:")
            logger.info(f"  - 크기: {stat.size} bytes")
            logger.info(f"  - 수정 시간: {stat.last_modified}")
            
            # 간단한 데이터 검증: 첫 몇 행 읽어서 확인
            response = client.get_object(PREPRO_BUCKET, PREPRO_OBJECT)
            df_check = pd.read_csv(BytesIO(response.read()), nrows=5)
            response.close()
            response.release_conn()
            
            logger.info(f"검증 샘플 데이터:\n{df_check}")
            logger.info("전처리 완료 및 검증 성공")
            
            return {
                'status': 'success',
                'output_path': output_meta['output_path'],
                'file_size': stat.size
            }
            
        except Exception as e:
            logger.error(f"검증 실패: {str(e)}")
            raise

    # Task Dependency 정의
    raw_meta = fetch_raw_data()
    prepro_meta = preprocess_data(raw_meta)
    validate_output(prepro_meta)
