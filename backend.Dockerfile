FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security hardening
RUN useradd -m appuser && chown -R appuser /app
USER appuser
ENV PATH="/home/appuser/.local/bin:${PATH}"

COPY --chown=appuser:appuser requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=appuser:appuser api.py .

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
