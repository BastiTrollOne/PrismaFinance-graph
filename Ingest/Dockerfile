# Ingest/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# CORRECCIÓN: Usamos 'libgl1' en lugar de 'libgl1-mesa-glx' para compatibilidad con Debian nuevo
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000 9999

CMD ["sh", "-c", "uvicorn ocr_server:app --port 9999 --host 0.0.0.0 & uvicorn testfapi:app --port 8000 --host 0.0.0.0"]