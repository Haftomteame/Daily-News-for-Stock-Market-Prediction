FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        postgresql-client \
        python3 \
        python3-pip \
    && curl -L https://github.com/dolthub/dolt/releases/latest/download/install.sh | bash \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages \
        "pandas>=2.0.0,<3.0" \
        "sqlalchemy>=2.0.0" \
        "psycopg2-binary>=2.9.9"

WORKDIR /data
