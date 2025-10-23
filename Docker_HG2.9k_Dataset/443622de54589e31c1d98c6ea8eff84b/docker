# Auto-generated Dockerfile for project: 443622de54589e31c1d98c6ea8eff84b
FROM python:3.9-slim

WORKDIR /app

COPY snippet.py /app/snippet.py
COPY requirements.txt /app/requirements.txt

# Optional system packages:
# RUN apt-get update && apt-get install -y --no-install-recommends build-essential gcc

RUN if [ -s /app/requirements.txt ]; then pip install --no-cache-dir -r /app/requirements.txt; fi

CMD ["python", "snippet.py"]
