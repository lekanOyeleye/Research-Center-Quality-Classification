from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from datetime import datetime, timedelta

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5) 
}

dag = DAG(
    'research_center_training_k8s',
    default_args=default_args,
    description='Train Research Centre KMeans Model in a K8s Pod',
    schedule_interval=None, # Manual Trigger
    catchup=False
)

train_model = KubernetesPodOperator(
    namespace='airflow',
    image="research-center-mlops:latest", # We will build this
    cmds=["python3", "train.py"],
    labels={'app': "research-center-training"},
    name='training-pod',
    task_id="train_model_in_k8s",
    get_logs=True,
    dag=dag,
    is_delete_operator_pod=True,
    image_pull_policy="Never", # use from Kind cache
    env_vars={
        "MLFLOW_TRACKING_URI" : "http://mlflow.mlflow.svc.cluster.local"
    }
    
)