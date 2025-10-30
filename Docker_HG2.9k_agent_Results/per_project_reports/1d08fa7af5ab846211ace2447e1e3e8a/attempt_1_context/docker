# Auto-generated Dockerfile for project: 1d08fa7af5ab846211ace2447e1e3e8a
FROM python:3.10-slim

WORKDIR /app

COPY snippet.py /app/snippet.py
COPY requirements.txt /app/requirements.txt

# Optional system packages:
# RUN apt-get update && apt-get install -y --no-install-recommends build-essential gcc

RUN if [ -s /app/requirements.txt ]; then pip install --no-cache-dir -r /app/requirements.txt; fi

CMD ["python", "snippet.py"]
