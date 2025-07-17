# 🚀 ElectionWatch API Deployment Guide

Complete guide for deploying your ElectionWatch Simple Agent API to various platforms.

## 📋 Pre-Deployment Checklist

- ✅ `app/simple_agent_api.py` working locally
- ✅ All dependencies in `app/requirements_api.txt`
- ✅ Environment variables configured
- ✅ Database connections tested (if used)
- ✅ Files organized in `app/` folder structure

## 🌐 Deployment Options

### 1. 🐳 **Docker Deployment** (Recommended)

#### Create Dockerfile
```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt

# Copy application files
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "simple_agent_api.py"]
```

#### Build and Run
```bash
# From the app/ directory
cd app/

# Build image (note the context is parent directory)
docker build -f Dockerfile -t electionwatch-api ..

# Run locally
docker run -p 8000:8000 -e PORT=8000 electionwatch-api

# Run with environment file
docker run -p 8000:8000 --env-file .env electionwatch-api
```

#### Docker Compose (with databases)
```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - PORT=8000
      - MONGODB_ATLAS_URI=${MONGODB_ATLAS_URI}
      - NEO4J_URI=${NEO4J_URI}
    depends_on:
      - mongodb
      - neo4j
    restart: unless-stopped

  mongodb:
    image: mongo:7
    ports:
      - "27017:27017"
    volumes:
      - mongodb_data:/data/db
    restart: unless-stopped

  neo4j:
    image: neo4j:5
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      - NEO4J_AUTH=neo4j/password
    volumes:
      - neo4j_data:/data
    restart: unless-stopped

volumes:
  mongodb_data:
  neo4j_data:
```

### 2. ☁️ **Google Cloud Run** (Serverless)

#### Setup
```bash
# Install gcloud CLI
# Set project
gcloud config set project YOUR-PROJECT-ID

# Enable required APIs
gcloud services enable run.googleapis.com
gcloud services enable artifactregistry.googleapis.com
```

#### Deploy
```bash
# Build and deploy in one command
gcloud run deploy electionwatch-api \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8000 \
  --memory 2Gi \
  --cpu 1 \
  --min-instances 0 \
  --max-instances 10 \
  --set-env-vars PORT=8000

# Custom domain (optional)
gcloud run domain-mappings create \
  --service electionwatch-api \
  --domain api.yourdomain.com \
  --region us-central1
```

#### Cloud Run YAML
```yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: electionwatch-api
  annotations:
    run.googleapis.com/ingress: all
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/minScale: "0"
        autoscaling.knative.dev/maxScale: "10"
        run.googleapis.com/memory: "2Gi"
        run.googleapis.com/cpu: "1"
    spec:
      containerConcurrency: 100
      containers:
      - image: gcr.io/YOUR-PROJECT/electionwatch-api
        ports:
        - containerPort: 8000
        env:
        - name: PORT
          value: "8000"
        resources:
          limits:
            memory: "2Gi"
            cpu: "1"
```

### 3. 🔷 **Azure Container Instances**

```bash
# Create resource group
az group create --name electionwatch-rg --location eastus

# Deploy container
az container create \
  --resource-group electionwatch-rg \
  --name electionwatch-api \
  --image YOUR-REGISTRY/electionwatch-api:latest \
  --dns-name-label electionwatch-api \
  --ports 8000 \
  --environment-variables PORT=8000 \
  --cpu 1 \
  --memory 2

# Get URL
az container show \
  --resource-group electionwatch-rg \
  --name electionwatch-api \
  --query ipAddress.fqdn
```

### 4. 🟠 **AWS Lambda** (Serverless)

#### Install Mangum adapter
```bash
pip install mangum
```

#### Lambda handler
```python
# lambda_handler.py
from mangum import Mangum
from simple_agent_api import app

handler = Mangum(app)
```

#### Deploy with Serverless Framework
```yaml
# serverless.yml
service: electionwatch-api

provider:
  name: aws
  runtime: python3.12
  region: us-east-1
  timeout: 30
  memorySize: 1024

functions:
  api:
    handler: lambda_handler.handler
    events:
      - httpApi: '*'

plugins:
  - serverless-python-requirements
```

```bash
npm install -g serverless
serverless deploy
```

### 5. 🟪 **Heroku** (Platform as a Service)

#### Procfile
```
web: python simple_agent_api.py
```

#### Deploy
```bash
# Install Heroku CLI
# Login
heroku login

# Create app
heroku create your-app-name

# Set config vars
heroku config:set PORT=8000
heroku config:set MONGODB_ATLAS_URI=your-mongo-uri

# Deploy
git add .
git commit -m "Deploy to Heroku"
git push heroku main

# Scale
heroku ps:scale web=1
```

### 6. 🔵 **DigitalOcean App Platform**

#### app.yaml
```yaml
name: electionwatch-api
services:
- name: api
  source_dir: /
  github:
    repo: your-username/your-repo
    branch: main
  run_command: python simple_agent_api.py
  environment_slug: python
  instance_count: 1
  instance_size_slug: basic-xxs
  envs:
  - key: PORT
    value: "8000"
  http_port: 8000
  health_check:
    http_path: /health
```

### 7. 🏠 **VPS/Dedicated Server**

#### Setup (Ubuntu)
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install python3.12 python3.12-venv nginx certbot -y

# Create app user
sudo useradd -m -s /bin/bash electionwatch
sudo su - electionwatch

# Setup application
git clone your-repo
cd your-repo
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements_api.txt

# Test locally
python simple_agent_api.py
```

#### Systemd Service
```ini
# /etc/systemd/system/electionwatch-api.service
[Unit]
Description=ElectionWatch API
After=network.target

[Service]
User=electionwatch
Group=electionwatch
WorkingDirectory=/home/electionwatch/your-repo
Environment=PATH=/home/electionwatch/your-repo/venv/bin
ExecStart=/home/electionwatch/your-repo/venv/bin/python simple_agent_api.py
Restart=always

[Install]
WantedBy=multi-user.target
```

#### Nginx Configuration
```nginx
# /etc/nginx/sites-available/electionwatch-api
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /health {
        proxy_pass http://127.0.0.1:8000/health;
        access_log off;
    }
}
```

#### Start Services
```bash
# Enable and start service
sudo systemctl enable electionwatch-api
sudo systemctl start electionwatch-api

# Setup nginx
sudo ln -s /etc/nginx/sites-available/electionwatch-api /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx

# Setup SSL
sudo certbot --nginx -d your-domain.com
```

## 🔧 **Configuration & Environment**

### Environment Variables
```bash
# Required
PORT=8000

# Optional (for enhanced features)
GOOGLE_CLOUD_PROJECT=your-project-id
MONGODB_ATLAS_URI=mongodb+srv://...
NEO4J_URI=bolt://localhost:7687

# Production settings
ENVIRONMENT=production
LOG_LEVEL=INFO
```

### .env file
```bash
# .env
PORT=8000
ENVIRONMENT=production
LOG_LEVEL=INFO
MONGODB_ATLAS_URI=your-mongo-connection-string
NEO4J_URI=your-neo4j-connection-string
```

## 🌍 **Domain & SSL Setup**

### Custom Domain
1. **Purchase domain** (Namecheap, GoDaddy, etc.)
2. **Point DNS** to your deployment IP/URL
3. **Setup SSL** (Let's Encrypt, Cloudflare, etc.)

### DNS Configuration
```
Type    Name    Value
A       @       YOUR-SERVER-IP
A       api     YOUR-SERVER-IP
CNAME   www     your-domain.com
```

## 📊 **Monitoring & Scaling**

### Health Monitoring
```bash
# Simple uptime monitoring
curl -f https://your-domain.com/health || echo "API is down!"

# With status page
# Use: UptimeRobot, Pingdom, or StatusPage.io
```

### Load Balancing (High Traffic)
```yaml
# docker-compose with load balancer
version: '3.8'
services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - api1
      - api2

  api1:
    build: .
    environment:
      - PORT=8000

  api2:
    build: .
    environment:
      - PORT=8000
```

### Auto-Scaling
- **Cloud Run**: Auto-scales 0-10 instances
- **AWS Lambda**: Auto-scales based on requests
- **Kubernetes**: HPA (Horizontal Pod Autoscaler)

## 🚀 **Quick Start Recommendations**

### For Development/Testing
```bash
# Local development
cd app/
python simple_agent_api.py

# Docker local
cd app/
docker build -f Dockerfile -t electionwatch-api ..
docker run -p 8000:8000 electionwatch-api
```

### For Production (Recommended)
1. **Small Scale**: Heroku or DigitalOcean App Platform
2. **Medium Scale**: Google Cloud Run or AWS Lambda
3. **Large Scale**: Kubernetes on GKE/EKS/AKS
4. **Custom**: VPS with Docker + Nginx + SSL

### Cost Comparison (Monthly)
- **Heroku**: $7+ (Basic Dyno)
- **Cloud Run**: $0-50+ (Pay per use)
- **VPS**: $5-20+ (DigitalOcean, Linode)
- **AWS Lambda**: $0-30+ (Pay per request)

## 🛡️ **Security Checklist**

- ✅ **HTTPS** enabled (SSL certificate)
- ✅ **Environment variables** for secrets
- ✅ **Rate limiting** (optional middleware)
- ✅ **CORS** configured properly
- ✅ **Firewall** rules configured
- ✅ **Regular updates** scheduled
- ✅ **Monitoring** and alerting setup

Your API is now ready for production deployment! 🎉 