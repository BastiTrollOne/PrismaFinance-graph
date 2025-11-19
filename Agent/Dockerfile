FROM python:3.11-slim

# ESTA LÍNEA ES NUEVA: Obliga a Python a mostrar los prints inmediatamente
ENV PYTHONUNBUFFERED=1 

WORKDIR /app
# ... (el resto del archivo déjalo igual)

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Exponemos el puerto 8081
EXPOSE 8081

# Arrancar la API Unificada
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8081"]