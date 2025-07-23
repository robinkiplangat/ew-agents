FROM python:3.12-slim

# Install system dependencies for ADK + Vertex AI
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ew_agents/ ./ew_agents/
COPY main.py ./
COPY *.json ./

# Set environment for production
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    GOOGLE_CLOUD_PROJECT=ew-agents-v02 \
    GOOGLE_CLOUD_LOCATION=europe-west1

EXPOSE 8080

CMD ["python", "-u", "main.py"]
