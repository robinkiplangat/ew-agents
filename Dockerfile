# syntax=docker/dockerfile:1

# Use Python 3.12 slim image for smaller size and better security
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim as base

# Prevents Python from writing pyc files and keeps Python from buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Create a non-privileged user for security
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# Install system dependencies for Google Cloud and ML libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the agent source code
COPY ew_agents/ ./ew_agents/
COPY ew-agent-service-key.json .
COPY .env.production ./.env

# Create proper directory structure for ADK
RUN mkdir -p /app/agents && \
    ln -s /app/ew_agents /app/agents/ew_agents

# Change ownership to non-privileged user
RUN chown -R appuser:appuser /app

# Switch to non-privileged user
USER appuser

# Set environment variables for Google Cloud
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/ew-agent-service-key.json
ENV GOOGLE_CLOUD_PROJECT=ew-agents-v02
ENV GOOGLE_CLOUD_LOCATION=europe-west1

# Expose port that the application listens on
EXPOSE 8080

# Start the ElectionWatch API server
CMD ["python", "main.py"] 