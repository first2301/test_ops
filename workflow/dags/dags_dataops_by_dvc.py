from airflow.sdk import DAG, task
import pendulum
from datetime import timedelta


default_args = {
    'owner': 'airflow',
    'retries': 2,  # 백업은 재시도 중요
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='dags_dataops_backup',
    description='백업 파이프라인: 변경된 데이터만 자동 백업',
    start_date=pendulum.datetime(2025, 11, 5),
    schedule="@daily",  # 매일 실행
    tags=['dvc', 'backup', 'dataops']
) as dag:

    @task.virtualenv(
        task_id="backup_data",
        requirements=['dvc']
    )
    def backup_data():
        """변경된 데이터 자동 감지 및 백업"""
        import subprocess
        import os
        from datetime import datetime
        
        project_root = os.getenv('AIRFLOW_HOME', '/opt/airflow')
        os.chdir(project_root)
        
        # 백업 대상 디렉토리 (실제 데이터 경로)
        backup_targets = [
            "data/data_raw",      # 원본 데이터
            "data/data_prepro",   # 전처리 데이터
        ]
        
        backup_stats = {
            "timestamp": datetime.now().isoformat(),
            "backed_up": [],
            "skipped": [],
            "failed": []
        }
        
        for target in backup_targets:
            if not os.path.exists(target):
                backup_stats["skipped"].append(f"{target} (not found)")
                print(f"백업 대상 없음: {target}")
                continue
            
            try:
                # 1. DVC add (변경사항만 자동 감지)
                add_result = subprocess.run(
                    ['dvc', 'add', target],
                    capture_output=True,
                    text=True,
                    check=False  # 변경사항 없으면 에러 발생 가능
                )
                
                if add_result.returncode == 0:
                    # 변경사항이 있어서 추가됨 → push 수행
                    push_result = subprocess.run(
                        ['dvc', 'push'],
                        capture_output=True,
                        text=True,
                        check=True
                    )
                    backup_stats["backed_up"].append(target)
                    print(f"✓ 백업 완료: {target}")
                else:
                    # 변경사항 없음 (정상 케이스)
                    backup_stats["skipped"].append(f"{target} (no changes)")
                    print(f"- 변경사항 없음: {target}")
                    
            except subprocess.CalledProcessError as e:
                backup_stats["failed"].append(f"{target}: {str(e)}")
                print(f"✗ 백업 실패: {target}")
                # 백업 실패는 경고만 (전체 파이프라인 중단 방지)
        
        # 백업 결과 요약
        print(f"\n=== 백업 결과 ===")
        print(f"백업 완료: {len(backup_stats['backed_up'])}개")
        print(f"변경사항 없음: {len(backup_stats['skipped'])}개")
        print(f"실패: {len(backup_stats['failed'])}개")
        
        # 실패가 있으면 경고만 출력 (파이프라인은 성공으로 처리)
        if backup_stats["failed"]:
            print(f"경고: {len(backup_stats['failed'])}개 백업 실패")
        
        return backup_stats


    # 단일 태스크로 간단하게 구성
    backup_result = backup_data()