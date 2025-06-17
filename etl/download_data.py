import pandas as pd
import os
from minio import Minio
from minio.error import S3Error
import yaml
from io import BytesIO
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Загрузка конфигурации
with open('/opt/airflow/config/config.yaml') as f:
    config = yaml.safe_load(f)

def main():
    # Загрузка данных
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
    column_names = [
        'id', 'diagnosis', 
        'mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area',
        'mean_smoothness', 'mean_compactness', 'mean_concavity', 'mean_concave_points',
        'mean_symmetry', 'mean_fractal_dimension',
        'se_radius', 'se_texture', 'se_perimeter', 'se_area',
        'se_smoothness', 'se_compactness', 'se_concavity', 'se_concave_points',
        'se_symmetry', 'se_fractal_dimension',
        'worst_radius', 'worst_texture', 'worst_perimeter', 'worst_area',
        'worst_smoothness', 'worst_compactness', 'worst_concavity', 'worst_concave_points',
        'worst_symmetry', 'worst_fractal_dimension'
    ]
    df = pd.read_csv(url, header=None, names=column_names)
    
    # Подключение к MinIO
    client = Minio(
        config['minio']['endpoint'],
        access_key=config['minio']['access_key'],
        secret_key=config['minio']['secret_key'],
        secure=config['minio']['secure']
    )
    
    # Проверка и создание бакета
    if not client.bucket_exists(config['minio']['bucket']):
        client.make_bucket(config['minio']['bucket'])
    
    # Сохранение в MinIO
    csv_data = df.to_csv(index=False).encode('utf-8')
    client.put_object(
        config['minio']['bucket'],
        "raw/breast_cancer.csv",
        data=BytesIO(csv_data),
        length=len(csv_data),
        content_type='application/csv'
    )
    logger.info("Data successfully uploaded to MinIO")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error in download_data: {str(e)}")
        raise