#!/bin/bash

# ElectionWatch API Deployment Script for Google Cloud Run
# Usage: ./deploy.sh [PROJECT_ID] [REGION]

set -e

# Configuration
PROJECT_ID=${1:-"your-project-id"}
REGION=${2:-"europe-west1"}
SERVICE_NAME="electionwatch-api"
IMAGE_NAME="gcr.io/$PROJECT_ID/$SERVICE_NAME"

echo "🚀 Deploying ElectionWatch API to Google Cloud Run"
echo "📋 Project: $PROJECT_ID"
echo "📍 Region: $REGION"
echo "🏷️  Image: $IMAGE_NAME"
echo ""

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI is not installed. Please install it first:"
    echo "   https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install it first:"
    echo "   https://docs.docker.com/get-docker/"
    exit 1
fi

# Set the project
echo "🔧 Setting up Google Cloud project..."
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "🔌 Enabling required APIs..."
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable artifactregistry.googleapis.com

# Build the Docker image
echo "🏗️  Building Docker image..."
docker build -f Dockerfile -t $IMAGE_NAME ..

# Push the image to Google Container Registry
echo "📤 Pushing image to registry..."
docker push $IMAGE_NAME

# Deploy to Cloud Run
echo "🚀 Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
    --source .. \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --port 8000 \
    --memory 2Gi \
    --cpu 1 \
    --min-instances 0 \
    --max-instances 10 \
    --set-env-vars PORT=8000,ENVIRONMENT=production,LOG_LEVEL=INFO

# Get the service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region $REGION --format 'value(status.url)')

echo ""
echo "✅ Deployment completed successfully!"
echo "🌐 Service URL: $SERVICE_URL"
echo "📚 API Documentation: $SERVICE_URL/docs"
echo "❤️  Health Check: $SERVICE_URL/health"
echo ""
echo "🔧 To update environment variables, use:"
echo "   gcloud run services update $SERVICE_NAME --region $REGION --set-env-vars KEY=VALUE"
echo ""
echo "📊 To view logs, use:"
echo "   gcloud logs tail /projects/$PROJECT_ID/logs/run.googleapis.com%2Frequests"
echo ""

# Test the deployment
echo "🧪 Testing deployment..."
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$SERVICE_URL/health")
if [ $HTTP_STATUS -eq 200 ]; then
    echo "✅ Health check passed!"
else
    echo "⚠️  Health check failed (HTTP $HTTP_STATUS). Check the logs."
fi

echo "🎉 Deployment process complete!" 