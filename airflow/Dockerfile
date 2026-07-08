FROM apache/airflow:2.8.4-python3.11

USER root

# Client Docker pour lancer spark-gold depuis le scheduler Airflow
RUN apt-get update \
    && apt-get install -y --no-install-recommends docker.io \
    && rm -rf /var/lib/apt/lists/* \
    && usermod -aG docker airflow || true

USER airflow

COPY airflow/requirements-airflow.txt /tmp/requirements-airflow.txt
COPY requirements.txt /opt/airflow/project/requirements.txt

RUN pip install --no-cache-dir -r /tmp/requirements-airflow.txt \
    && pip install --no-cache-dir -r /opt/airflow/project/requirements.txt

ENV PYTHONPATH=/opt/airflow/project \
    AIRFLOW__CORE__LOAD_EXAMPLES=False \
    AIRFLOW__CORE__DAGS_FOLDER=/opt/airflow/dags \
    AIRFLOW__WEBSERVER__EXPOSE_CONFIG=True

WORKDIR /opt/airflow
