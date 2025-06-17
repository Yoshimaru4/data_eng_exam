FROM apache/airflow:2.7.1-python3.10

USER root
RUN apt-get update && \
    apt-get install -y wget && \
    rm -rf /var/lib/apt/lists/*

USER airflow
RUN pip install --no-cache-dir \
    minio==7.1.16 \
    pandas==2.1.0 \
    scikit-learn==1.3.0 \
    pyyaml==6.0.1 \
    joblib==1.3.2

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt