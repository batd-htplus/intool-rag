ARG PYTHON_VERSION=3.11-slim
FROM python:${PYTHON_VERSION}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    curl \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -U pip setuptools wheel

RUN pip install --no-cache-dir \
    "numpy<2.0.0" \
    torch==2.2.0+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html

COPY rag/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt && \
    pip check || true

RUN find /usr/local/lib/python3.11/site-packages/onnxruntime \
    -name "*.so" -type f \
    -exec patchelf --clear-execstack {} \; \
    2>/dev/null || true

COPY rag /app/rag
ENV PYTHONPATH=/app

RUN groupadd -r raguser && useradd -r -g raguser -m -d /home/raguser raguser && \
    mkdir -p /storage/uploads /home/raguser/.cache && \
    chown -R raguser:raguser /app /storage /home/raguser

USER raguser
ENV HOME=/home/raguser

EXPOSE 8001
CMD ["python", "-m", "uvicorn", "rag.main:app", "--host", "0.0.0.0", "--port", "8001"]
