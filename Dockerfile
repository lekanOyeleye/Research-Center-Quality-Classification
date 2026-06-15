FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN useradd --create-home appuser

COPY requirements.txt .

RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc libc-dev \
 && pip install --no-cache-dir -r requirements.txt \
 && apt-get remove -y gcc libc-dev \
 && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/*

# Copy application
COPY app.py .

# Set ownership
RUN chown -R appuser /app

USER appuser

EXPOSE 5001


CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5001", "--workers", "4"]