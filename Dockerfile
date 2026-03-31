FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy ML source (build context = the ml/ directory itself)
COPY . ./ml/

# Create artifact directories
RUN mkdir -p ml/artifacts/data ml/artifacts/models ml/artifacts/metrics

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "ml.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
