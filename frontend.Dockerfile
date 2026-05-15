FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m appuser && chown -R appuser /app
USER appuser
ENV PATH="/home/appuser/.local/bin:${PATH}"

COPY --chown=appuser:appuser requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=appuser:appuser . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
