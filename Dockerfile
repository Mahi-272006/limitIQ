FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set PYTHONPATH so src imports work
ENV PYTHONPATH=/app/src

# Copy source code
COPY src/ ./src/
COPY api/ ./api/
COPY dashboard/ ./dashboard/
COPY tests/ ./tests/
COPY data/ ./data/
COPY models/ ./models/
COPY .env ./
COPY .env.example ./
COPY entrypoint.sh ./

RUN chmod +x entrypoint.sh

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import requests; r=requests.get('http://localhost:8000/health'); exit(0 if r.status_code==200 else 1)" || exit 1

ENTRYPOINT ["./entrypoint.sh"]
CMD ["gunicorn", "api.main:app", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "120"]
