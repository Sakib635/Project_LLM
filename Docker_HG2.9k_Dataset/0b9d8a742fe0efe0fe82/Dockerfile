# Auto-generated Dockerfile for project: 0b9d8a742fe0efe0fe82
FROM python:3.11-slim

WORKDIR /app

COPY snippet.py /app/snippet.py
COPY requirements.txt /app/requirements.txt

# Optional system packages:
# RUN apt-get update && apt-get install -y --no-install-recommends build-essential gcc

RUN if [ -s /app/requirements.txt ]; then pip install --no-cache-dir -r /app/requirements.txt; fi

CMD ["python", "snippet.py"]
