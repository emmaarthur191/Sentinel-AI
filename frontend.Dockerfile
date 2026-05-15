FROM python:3.10-slim

WORKDIR /app

# Create non-root user for security hardening
RUN useradd -m appuser && chown -R appuser /app
USER appuser
ENV PATH="/home/appuser/.local/bin:${PATH}"

COPY --chown=appuser:appuser requirements-app.txt .
RUN pip install --no-cache-dir --user -r requirements-app.txt

COPY --chown=appuser:appuser app.py .

CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
