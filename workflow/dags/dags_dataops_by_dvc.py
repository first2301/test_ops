from airflow.sdk import DAG, task
import pendulum
from datetime import timedelta
import subprocess
import os


default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

with DAG(
    dag_id='dags_dataops_by_dvc',
    description='DVC를 사용하여 데이터 파이프라인 관리',
    start_date=pendulum.datetime(2025, 11, 5),
    schedule="@daily",
    tags=['dvc', 'dataops']
) as dag:

    @task.virtualenv(
        task_id="dvc_pull_data",
        requirements=['dvc']
    )
    def dvc_pull_data():
        """DVC 원격 저장소에서 데이터 가져오기"""
        import subprocess
        import os
        
        # 프로젝트 루트 디렉토리로 이동 (필요시)
        project_root = os.getenv('AIRFLOW_HOME', '/opt/airflow')
        os.chdir(project_root)
        
        try:
            result = subprocess.run(
                ['dvc', 'pull'],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"DVC pull 성공: {result.stdout}")
            return {"status": "success", "message": result.stdout}
        except subprocess.CalledProcessError as e:
            print(f"DVC pull 실패: {e.stderr}")
            raise


    @task.virtualenv(
        task_id="dvc_add_data",
        requirements=['dvc']
    )
    def dvc_add_data():
        """변경된 데이터 파일을 DVC에 추가"""
        import subprocess
        import os
        
        project_root = os.getenv('AIRFLOW_HOME', '/opt/airflow')
        os.chdir(project_root)
        
        data_files = [
            "test/test2.txt",
            "test/test3.txt",
            "test/test.txt"
        ]
        
        results = []
        failed_files = []
        missing_files = []
        
        for file_path in data_files:
            if os.path.exists(file_path):
                try:
                    result = subprocess.run(
                        ['dvc', 'add', file_path],
                        capture_output=True,
                        text=True,
                        check=True
                    )
                    print(f"DVC add 성공 ({file_path}): {result.stdout}")
                    results.append({"file": file_path, "status": "success"})
                except subprocess.CalledProcessError as e:
                    print(f"DVC add 실패 ({file_path}): {e.stderr}")
                    results.append({"file": file_path, "status": "failed", "error": e.stderr})
                    failed_files.append(file_path)
            else:
                print(f"파일이 존재하지 않습니다: {file_path}")
                results.append({"file": file_path, "status": "not_found"})
                missing_files.append(file_path)
        
        # DVC add 명령어가 실패한 경우 exception을 raise하여 task를 실패로 표시
        # (다른 함수들과 일관성 유지: dvc_pull_data, dvc_push_data는 실패 시 raise)
        if failed_files:
            error_msg = f"DVC add 실패: {len(failed_files)}개 파일 처리 실패 - {', '.join(failed_files)}"
            print(error_msg)
            raise Exception(error_msg)
        
        # 모든 파일이 없는 경우도 실패로 처리 (데이터가 전혀 없는 경우)
        if missing_files and not results:
            error_msg = f"DVC add 실패: 모든 파일이 존재하지 않음 - {', '.join(missing_files)}"
            print(error_msg)
            raise Exception(error_msg)
        
        # 일부 파일만 없는 경우는 경고만 출력하고 계속 진행
        if missing_files:
            print(f"경고: {len(missing_files)}개 파일이 존재하지 않지만 계속 진행합니다: {', '.join(missing_files)}")
        
        return {"results": results}


    @task.virtualenv(
        task_id="dvc_push_data",
        requirements=['dvc']
    )
    def dvc_push_data():
        """DVC 원격 저장소에 데이터 업로드"""
        import subprocess
        import os
        
        project_root = os.getenv('AIRFLOW_HOME', '/opt/airflow')
        os.chdir(project_root)
        
        try:
            result = subprocess.run(
                ['dvc', 'push'],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"DVC push 성공: {result.stdout}")
            return {"status": "success", "message": result.stdout}
        except subprocess.CalledProcessError as e:
            print(f"DVC push 실패: {e.stderr}")
            raise


    # Task 의존성 설정: pull -> add -> push 순서로 실행
    pull_result = dvc_pull_data()
    add_result = dvc_add_data()
    push_result = dvc_push_data()
    
    pull_result >> add_result >> push_result