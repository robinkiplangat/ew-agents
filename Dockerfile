# syntax=docker/dockerfile:1.4

#################################
# Multi-stage build for optimization
#################################

# Build stage - install dependencies and compile
FROM python:3.12-slim as builder

# Set build arguments
ARG DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create virtual environment for dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

#################################
# Production stage - minimal runtime
#################################

FROM python:3.12-slim as production

# Security and performance environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set Google Cloud environment variables
ENV GOOGLE_CLOUD_PROJECT=ew-agents-v02 \
    GOOGLE_CLOUD_LOCATION=europe-west1 \
    GOOGLE_APPLICATION_CREDENTIALS=/app/credentials/service-key.json

# Install minimal runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create application user with specific UID for consistency
ARG UID=10001
RUN groupadd --gid ${UID} appgroup && \
    useradd --uid ${UID} --gid appgroup --no-create-home --shell /bin/false appuser

# Create directories with proper permissions
RUN mkdir -p /app/credentials /app/data /app/logs && \
    chown -R appuser:appgroup /app

# Set working directory
WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application files with proper ownership
COPY --chown=appuser:appgroup ew_agents/ ./ew_agents/
COPY --chown=appuser:appgroup main.py ./
COPY --chown=appuser:appgroup .env.production ./.env

# Copy service account key securely
COPY --chown=appuser:appgroup ew-agent-service-key.json ./credentials/service-key.json

# Create ADK-compatible directory structure
RUN mkdir -p /app/agents && \
    ln -s /app/ew_agents /app/agents/ew_agents && \
    chown -R appuser:appgroup /app/agents

# Add health check script
RUN echo '#!/usr/bin/env python3\n\
import requests\n\
import sys\n\
import os\n\
\n\
def health_check():\n\
    try:\n\
        port = os.environ.get("PORT", "8080")\n\
        response = requests.get(f"http://localhost:{port}/health", timeout=10)\n\
        if response.status_code == 200:\n\
            print("Health check passed")\n\
            sys.exit(0)\n\
        else:\n\
            print(f"Health check failed with status: {response.status_code}")\n\
            sys.exit(1)\n\
    except Exception as e:\n\
        print(f"Health check failed with error: {e}")\n\
        sys.exit(1)\n\
\n\
if __name__ == "__main__":\n\
    health_check()' > /app/healthcheck.py && \
    chmod +x /app/healthcheck.py && \
    chown appuser:appgroup /app/healthcheck.py

# Switch to non-privileged user
USER appuser

# Expose port
EXPOSE 8080

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python /app/healthcheck.py

# Add labels for better maintainability
LABEL org.opencontainers.image.title="ElectionWatch Misinformation Detection API" \
      org.opencontainers.image.description="AI-powered misinformation detection for African elections" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.vendor="ElectionWatch" \
      maintainer="ElectionWatch Team"

# Use exec form for better signal handling and add -u flag for unbuffered output
CMD ["python", "-u", "main.py"] 