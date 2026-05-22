FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PYTHONUNBUFFERED=1
ENV PIP_DEFAULT_TIMEOUT=120
ENV PIP_RETRIES=10

WORKDIR /workspace

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src ./src
COPY config ./config
