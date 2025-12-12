FROM python:3.10-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl build-essential libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy code
COPY api /app/api
COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy model artifacts (downloaded by GitHub before build)
COPY model /app/model

EXPOSE 7860

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "7860"]
