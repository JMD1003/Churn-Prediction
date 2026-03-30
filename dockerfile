FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY . .

COPY src/serving/models/m-6388e90892ae420a8f7c5c976b4003a7/artifacts/model.ubj /app/model/model.ubj

COPY artifacts/feature_columns.json /app/model/feature_columns.json

COPY artifacts/preprocessing.pkl /app/model/preprocessing.pkl

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    MODEL_DIR=/app/model

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]