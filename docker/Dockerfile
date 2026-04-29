FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy minimal dependency manifest first for layer caching
COPY pyproject.toml ./
RUN pip install --no-cache-dir \
    "torch>=2.2" "transformers>=4.40" "lightgbm>=4.3" "scikit-learn>=1.4" \
    "pandas>=2.2" "numpy>=1.26" "tldextract>=5.1" "beautifulsoup4>=4.12" \
    "fastapi>=0.110" "uvicorn[standard]>=0.29" "onnxruntime>=1.17" \
    "pydantic>=2.6" "pillow>=10.2"

COPY src/ ./src/
COPY models/ ./models/
ENV PYTHONPATH=/app/src

EXPOSE 8000
CMD ["uvicorn", "phishguard.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
