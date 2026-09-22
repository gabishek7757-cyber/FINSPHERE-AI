FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default port for FastAPI (8000) or Streamlit (8501)
EXPOSE 8000 8501

CMD ["uvicorn", "src.api.app_api:app", "--host", "0.0.0.0", "--port", "8000"]
