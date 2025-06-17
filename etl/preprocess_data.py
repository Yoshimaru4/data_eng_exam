import pandas as pd
import numpy as np
from minio import Minio
from io import BytesIO, StringIO
import yaml
import logging
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

with open('/opt/airflow/config/config.yaml') as f:
    config = yaml.safe_load(f)

def main():
    # Подключение к MinIO
    client = Minio(
        config['minio']['endpoint'],
        access_key=config['minio']['access_key'],
        secret_key=config['minio']['secret_key'],
        secure=config['minio']['secure']
    )
    
    # Загрузка данных
    response = client.get_object(config['minio']['bucket'], "raw/breast_cancer.csv")
    df = pd.read_csv(response)
    response.close()
    response.release_conn()
    
    # Предобработка
    df.drop('id', axis=1, inplace=True)
    le = LabelEncoder()
    df['diagnosis'] = le.fit_transform(df['diagnosis'])
    
    # Разделение данных
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Масштабирование
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Сохранение в MinIO
    for name, data in zip(
        ['X_train', 'X_test', 'y_train', 'y_test', 'scaler'],
        [X_train, X_test, y_train, y_test, scaler]
    ):
        buffer = BytesIO()
        if name.startswith('X_') or name.startswith('y_'):
            np.save(buffer, data)
            content_type = 'application/npy'
        else:
            joblib.dump(data, buffer)
            content_type = 'application/octet-stream'
        
        buffer.seek(0)
        client.put_object(
            config['minio']['bucket'],
            f"processed/{name}.npy" if name.startswith(('X_', 'y_')) else f"models/{name}.joblib",
            data=buffer,
            length=buffer.getbuffer().nbytes,
            content_type=content_type
        )
    
    logger.info("Data preprocessing completed successfully")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Preprocessing error: {str(e)}")
        raise