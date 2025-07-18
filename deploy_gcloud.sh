#!/bin/bash

# ElectionWatch Agents - gCloud Deployment Script
# Deploys using custom Dockerfile with proper memory configuration

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ID="ew-agents-v02"
REGION="europe-west1"
SERVICE_NAME="electionwatch-misinformation-api"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"
MEMORY="2Gi"        # 2GiB to fix memory issues
CPU="1"
MAX_INSTANCES="10"
MIN_INSTANCES="1"

echo -e "${BLUE}🛡️  ElectionWatch Misinformation Detection API - Cloud Deployment${NC}"
echo -e "${BLUE}=========================================================${NC}"
echo ""

# Step 1: Verify prerequisites
echo -e "${BLUE}📋 Step 1: Verifying prerequisites...${NC}"

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}❌ gcloud CLI is not installed. Please install it first.${NC}"
    exit 1
fi

# Note: Using Google Cloud Build - no local Docker required
echo -e "${GREEN}ℹ️  Using Google Cloud Build (no local Docker needed)${NC}"

# Verify required files exist
required_files=("Dockerfile" "requirements.txt" "ew_agents/agent.py" "ew-agent-service-key.json" ".env.production")
for file in "${required_files[@]}"; do
    if [[ ! -f "$file" ]]; then
        echo -e "${RED}❌ Required file not found: $file${NC}"
        exit 1
    fi
done

echo -e "${GREEN}✅ All prerequisites verified${NC}"
echo ""

# Step 2: Set up gcloud configuration
echo -e "${BLUE}⚙️  Step 2: Configuring gcloud...${NC}"

gcloud config set project $PROJECT_ID
gcloud config set run/region $REGION

# Enable required APIs
echo -e "${YELLOW}📡 Enabling required Google Cloud APIs...${NC}"
gcloud services enable \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    containerregistry.googleapis.com

echo -e "${GREEN}✅ gcloud configuration complete${NC}"
echo ""

# Step 3: Build the container image
echo -e "${BLUE}🔨 Step 3: Building container image...${NC}"

echo -e "${YELLOW}📦 Building Docker image: $IMAGE_NAME${NC}"
gcloud builds submit --tag $IMAGE_NAME .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Container image built successfully${NC}"
else
    echo -e "${RED}❌ Container build failed${NC}"
    exit 1
fi
echo ""

# Step 4: Deploy to Cloud Run
echo -e "${BLUE}🚀 Step 4: Deploying to Cloud Run...${NC}"

echo -e "${YELLOW}🌐 Deploying service: $SERVICE_NAME${NC}"
echo -e "${YELLOW}   Memory: $MEMORY${NC}"
echo -e "${YELLOW}   CPU: $CPU${NC}"
echo -e "${YELLOW}   Min instances: $MIN_INSTANCES${NC}"
echo -e "${YELLOW}   Max instances: $MAX_INSTANCES${NC}"

gcloud run deploy $SERVICE_NAME \
    --image $IMAGE_NAME \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --memory $MEMORY \
    --cpu $CPU \
    --min-instances $MIN_INSTANCES \
    --max-instances $MAX_INSTANCES \
    --timeout 3600 \
    --concurrency 80 \
    --port 8080 \
    --set-env-vars="GOOGLE_CLOUD_PROJECT=${PROJECT_ID},GOOGLE_CLOUD_LOCATION=${REGION}" \
    --service-account="ew-agent-service@${PROJECT_ID}.iam.gserviceaccount.com"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Service deployed successfully${NC}"
else
    echo -e "${RED}❌ Service deployment failed${NC}"
    exit 1
fi
echo ""

# Step 5: Get service information
echo -e "${BLUE}ℹ️  Step 5: Service information...${NC}"

SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)")

echo -e "${GREEN}🎉 Deployment completed successfully!${NC}"
echo ""
echo -e "${GREEN}📍 Service Details:${NC}"
echo -e "   Service Name: $SERVICE_NAME"
echo -e "   Region: $REGION"
echo -e "   Memory: $MEMORY (Fixed 512MiB → 2GiB)"
echo -e "   CPU: $CPU"
echo -e "   Image: $IMAGE_NAME"
echo ""
echo -e "${GREEN}🌐 Service URLs:${NC}"
echo -e "   Web UI: ${SERVICE_URL}/dev-ui/"
echo -e "   API Docs: ${SERVICE_URL}/docs"
echo -e "   Health Check: ${SERVICE_URL}/health"
echo -e "   List Apps: ${SERVICE_URL}/list-apps"
echo ""
echo -e "${BLUE}🧪 Quick Test Commands:${NC}"
echo -e "   curl \"${SERVICE_URL}/list-apps\""
echo -e "   curl \"${SERVICE_URL}/health\""
echo ""
echo -e "${GREEN}✨ ElectionWatch Misinformation Detection API is now live and ready!${NC}" 