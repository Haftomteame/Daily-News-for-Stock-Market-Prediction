FROM apache/airflow:2.8.4-python3.11

USER root

# Client Docker recent (API >= 1.44) + plugin compose pour lancer spark-gold
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl gnupg \
    && install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc \
    && chmod a+r /etc/apt/keyrings/docker.asc \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian bookworm stable" > /etc/apt/sources.list.d/docker.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends docker-ce-cli docker-compose-plugin \
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
