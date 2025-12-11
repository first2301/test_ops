

- airflow
echo -e "AIRFLOW_UID=$(id -u)" > .env



# dvc
## 1. DVC 프로젝트 초기화 (아직 안 했다면)
- dvc init

## 2. MinIO 버킷을 'minio_remote'라는 이름의 원격 저장소로 추가
-d 플래그는 기본(default) 원격으로 설정합니다.
- dvc remote add -d minio_remote s3://<your-bucket-name>

## 3. MinIO 서버의 엔드포인트 URL 설정
- dvc remote modify minio_remote endpointurl http://<your-minio-server-url>:<port>

## 4. 인증 정보 (Access Key 및 Secret Key) 설정
- --local 플래그를 사용하여 인증 정보를 .dvc/config.local에 저장하여 Git 추적에서 제외합니다.
- dvc remote modify --local minio_remote access_key_id <your-access-key>
- dvc remote modify --local minio_remote secret_access_key <your-secret-key>

## 5. (선택) SSL을 사용하지 않는 경우 (예: 로컬 MinIO)
- dvc remote modify minio_remote use_ssl False
