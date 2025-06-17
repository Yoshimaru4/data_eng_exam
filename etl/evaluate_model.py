import numpy as np
from minio import Minio
from io import BytesIO
import joblib
import json
import yaml
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

with open('/opt/airflow/config/config.yaml') as f:
    config = yaml.safe_load(f)

def load_from_minio(path):
    client = Minio(
        config['minio']['endpoint'],
        access_key=config['minio']['access_key'],
        secret_key=config['minio']['secret_key'],
        secure=config['minio']['secure']
    )
    response = client.get_object(config['minio']['bucket'], path)
    data = BytesIO(response.read())
    response.close()
    response.release_conn()
    return data

def main():
    # Загрузка данных и модели
    X_test = np.load(load_from_minio("processed/X_test.npy"))
    y_test = np.load(load_from_minio("processed/y_test.npy"))
    model = joblib.load(load_from_minio("models/logistic_regression_model.joblib"))
    
    # Предсказание и оценка
    y_pred = model.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }
    
    # Сохранение метрик
    client = Minio(
        config['minio']['endpoint'],
        access_key=config['minio']['access_key'],
        secret_key=config['minio']['secret_key'],
        secure=config['minio']['secure']
    )
    
    # В MinIO
    json_data = json.dumps(metrics).encode('utf-8')
    client.put_object(
        config['minio']['bucket'],
        "results/metrics.json",
        data=BytesIO(json_data),
        length=len(json_data),
        content_type='application/json'
    )
    
    # Локально
    with open('/opt/airflow/results/metrics.json', 'w') as f:
        json.dump(metrics, f)
    
    logger.info(f"Model evaluation completed. Metrics: {metrics}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Evaluation error: {str(e)}")
        raise