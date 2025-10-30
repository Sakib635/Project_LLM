# Auto-generated Dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY . /app
RUN if [ -s /app/requirements.txt ]; then pip install --no-cache-dir -r /app/requirements.txt; fi
CMD ["python", "snippet.py"]
