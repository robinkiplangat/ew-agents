#!/bin/bash

# ElectionWatch Optimized Cloud Run Deployment
# Combines Cloud Run flexibility with Agent Engine benefits

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
PROJECT_ID="ew-agents-v02"
REGION="europe-west1"
SERVICE_NAME="electionwatch-agents-api"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"
MEMORY="2Gi"        # Optimized for quota limits
CPU="1"             # Within quota limits
MAX_INSTANCES="5"   # Max allowed per quota
MIN_INSTANCES="1"   # Keep warm instances

echo -e "${BLUE}🚀 ElectionWatch Optimized Deployment${NC}"
echo -e "${BLUE}=====================================${NC}"
echo ""

# Step 1: Verify ADK compatibility
echo -e "${BLUE}📋 Step 1: Verifying ADK setup...${NC}"

if [[ ! -f "ew_agents/vertex_ai_integration.py" ]]; then
    echo -e "${RED}❌ Vertex AI integration not found${NC}"
    exit 1
fi

if [[ ! -f "ew_agents/coordinator_integration.py" ]]; then
    echo -e "${RED}❌ Enhanced coordinator not found${NC}"
    exit 1
fi

echo -e "${GREEN}✅ ADK components verified${NC}"

# Step 2: Enable required APIs (enhanced for Agent Engine compatibility)
echo -e "${BLUE}⚙️  Step 2: Enabling Google Cloud APIs...${NC}"

gcloud config set project $PROJECT_ID
gcloud config set run/region $REGION

gcloud services enable \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    containerregistry.googleapis.com \
    aiplatform.googleapis.com

echo -e "${GREEN}✅ APIs enabled${NC}"

# Step 3: Build optimized container
echo -e "${BLUE}🔨 Step 3: Building optimized container...${NC}"

# Create optimized Dockerfile if needed
cat > Dockerfile.optimized << 'EOF'
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
EOF

echo -e "${YELLOW}📦 Building with optimized Dockerfile...${NC}"
mv Dockerfile.optimized Dockerfile
gcloud builds submit --tag $IMAGE_NAME .

echo -e "${GREEN}✅ Container built successfully${NC}"

# Step 4: Deploy with Agent Engine-like capabilities
echo -e "${BLUE}🚀 Step 4: Deploying with enhanced configuration...${NC}"

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
    --concurrency 100 \
    --port 8080 \
    --set-env-vars="GOOGLE_CLOUD_PROJECT=${PROJECT_ID},GOOGLE_CLOUD_LOCATION=${REGION},VERTEX_AI_ENABLED=true,ADK_VERTEX_INTEGRATION=enabled,MONGODB_URI=mongodb+srv://ew_ml:moHsc5i6gYFrLsvL@ewcluster1.fpkzpxg.mongodb.net/knowledge?retryWrites=true&w=majority,ADK_AGENTS_ENABLED=true,ENHANCED_COORDINATOR_ENABLED=true,REASONING_ENGINE_COMPATIBLE=true" \
    --service-account="ew-agent-service@${PROJECT_ID}.iam.gserviceaccount.com" \
    --execution-environment gen2 \
    --cpu-boost

echo -e "${GREEN}✅ Service deployed successfully${NC}"

# Step 5: Get service information and test
echo -e "${BLUE}ℹ️  Step 5: Service validation...${NC}"

SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)")

echo -e "${GREEN}🎉 Deployment completed successfully!${NC}"
echo ""
echo -e "${GREEN}📍 Service Details:${NC}"
echo -e "   Service Name: $SERVICE_NAME"
echo -e "   URL: $SERVICE_URL"
echo -e "   Memory: $MEMORY"
echo -e "   CPU: $CPU"
echo -e "   Scaling: $MIN_INSTANCES-$MAX_INSTANCES instances"
echo ""

# Test endpoints
echo -e "${BLUE}🧪 Testing deployment...${NC}"

echo -e "${YELLOW}📋 Testing health check...${NC}"
curl -s "${SERVICE_URL}/health/comprehensive" | head -5

echo -e "${YELLOW}📋 Testing Vertex AI integration...${NC}"
curl -s "${SERVICE_URL}/vertex-ai/health" | head -5

echo -e "${YELLOW}📋 Testing deployment info...${NC}"
curl -s "${SERVICE_URL}/deployment/info" | head -10

echo ""
echo -e "${GREEN}🌐 Available Endpoints:${NC}"
echo -e "   Health Check: ${SERVICE_URL}/health/comprehensive"
echo -e "   Standard Analysis: ${SERVICE_URL}/analyze"
echo -e "   Vertex AI Analysis: ${SERVICE_URL}/analyze/vertex-ai"
echo -e "   API Documentation: ${SERVICE_URL}/docs"
echo -e "   Deployment Info: ${SERVICE_URL}/deployment/info"
echo ""
echo -e "${GREEN}✨ ElectionWatch agents are live with Agent Engine capabilities!${NC}"

# Cleanup temporary files
rm -f Dockerfile.optimized

echo ""
echo -e "${BLUE}🎯 Next Steps:${NC}"
echo -e "   1. Test the enhanced endpoints"
echo -e "   2. Monitor performance metrics"
echo -e "   3. Scale based on traffic patterns"
echo -e "   4. Consider adding load balancing for production" 