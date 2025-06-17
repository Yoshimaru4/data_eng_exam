import numpy as np
from minio import Minio
from io import BytesIO
import joblib
import yaml
import logging
from sklearn.linear_model import LogisticRegression

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
    # Загрузка данных
    X_train = np.load(load_from_minio("processed/X_train.npy"))
    y_train = np.load(load_from_minio("processed/y_train.npy"))
    
    # Обучение модели
    model = LogisticRegression(
        max_iter=10000,
        penalty='l2',
        solver='saga',
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Сохранение модели
    buffer = BytesIO()
    joblib.dump(model, buffer)
    buffer.seek(0)
    
    client = Minio(
        config['minio']['endpoint'],
        access_key=config['minio']['access_key'],
        secret_key=config['minio']['secret_key'],
        secure=config['minio']['secure']
    )
    client.put_object(
        config['minio']['bucket'],
        "models/logistic_regression_model.joblib",
        data=buffer,
        length=buffer.getbuffer().nbytes,
        content_type='application/octet-stream'
    )
    logger.info("Model trained and saved successfully")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        raise