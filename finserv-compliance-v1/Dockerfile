# ============================================================
# FinServ Compliance Assistant — API Service
# ============================================================
# Multi-stage build:
#   Stage 1 (builder): install Python deps into a venv
#   Stage 2 (runtime): copy venv + app code; non-root user
# ============================================================

# ---- Stage 1: builder ----------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps for spaCy, pdfminer, sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libpoppler-cpp-dev \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create isolated venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download spaCy English model (required by Presidio PII redactor)
RUN python -m spacy download en_core_web_lg

# ---- Stage 2: runtime ----------------------------------------
FROM python:3.11-slim AS runtime

# Runtime system libs only
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copy application source
COPY config/     ./config/
COPY src/        ./src/
COPY docs/       ./docs/
COPY sample_docs/ ./sample_docs/

# Ensure Python finds local packages
ENV PYTHONPATH="/app"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Drop to non-root
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "2", "--log-level", "info"]
