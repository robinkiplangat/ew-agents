#!/bin/bash

# ElectionWatch Optimized Cloud Run Deployment
# Combines Cloud Run flexibility with Agent Engine benefits and Reports System

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

echo -e "${BLUE}🚀 ElectionWatch Optimized Deployment with Reports System${NC}"
echo -e "${BLUE}=====================================================${NC}"
echo ""

# Step 1: Verify ADK compatibility and new features
echo -e "${BLUE}📋 Step 1: Verifying system components...${NC}"

if [[ ! -f "ew_agents/vertex_ai_integration.py" ]]; then
    echo -e "${RED}❌ Vertex AI integration not found${NC}"
    exit 1
fi

if [[ ! -f "ew_agents/coordinator_integration.py" ]]; then
    echo -e "${RED}❌ Enhanced coordinator not found${NC}"
    exit 1
fi

if [[ ! -f "ew_agents/mongodb_storage.py" ]]; then
    echo -e "${RED}❌ MongoDB storage not found${NC}"
    exit 1
fi

echo -e "${GREEN}✅ All system components verified${NC}"

# Step 2: Enable required APIs (enhanced for Agent Engine compatibility)
echo -e "${BLUE}⚙️  Step 2: Enabling Google Cloud APIs...${NC}"

gcloud config set project $PROJECT_ID
gcloud config set run/region $REGION

gcloud services enable \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    containerregistry.googleapis.com \
    aiplatform.googleapis.com \
    secretmanager.googleapis.com

echo -e "${GREEN}✅ APIs enabled${NC}"

# Step 3: Setup Secrets Manager
echo -e "${BLUE}🔐 Step 3: Setting up Secrets Manager...${NC}"

# Setup MongoDB URI secret
echo -e "${YELLOW}📝 Setting up MongoDB URI secret...${NC}"
if [[ -f ".env" ]]; then
    MONGODB_ATLAS_URI=$(grep "MONGODB_ATLAS_URI" .env | cut -d '=' -f2)
    if [[ -n "$MONGODB_ATLAS_URI" ]]; then
        echo -e "${YELLOW}📝 Found MONGODB_ATLAS_URI in .env file${NC}"
        
        # Create secret if it doesn't exist
        if ! gcloud secrets describe "mongodb-atlas-uri" --project=$PROJECT_ID >/dev/null 2>&1; then
            echo -e "${YELLOW}🔐 Creating mongodb-atlas-uri secret...${NC}"
            echo "$MONGODB_ATLAS_URI" | gcloud secrets create "mongodb-atlas-uri" \
                --data-file=- \
                --project=$PROJECT_ID \
                --replication-policy="automatic"
        else
            echo -e "${YELLOW}🔄 Updating existing mongodb-atlas-uri secret...${NC}"
            echo "$MONGODB_ATLAS_URI" | gcloud secrets versions add "mongodb-atlas-uri" \
                --data-file=- \
                --project=$PROJECT_ID
        fi
        
        # Grant access to the Cloud Run service account
        SERVICE_ACCOUNT="ew-agent-service@${PROJECT_ID}.iam.gserviceaccount.com"
        gcloud secrets add-iam-policy-binding "mongodb-atlas-uri" \
            --member="serviceAccount:${SERVICE_ACCOUNT}" \
            --role="roles/secretmanager.secretAccessor" \
            --project=$PROJECT_ID
        
        echo -e "${GREEN}✅ MongoDB URI secret configured${NC}"
    else
        echo -e "${YELLOW}⚠️  MONGODB_ATLAS_URI not found in .env file${NC}"
        echo -e "${YELLOW}   You can set it manually in Secret Manager later${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  .env file not found${NC}"
    echo -e "${YELLOW}   You can set MONGODB_ATLAS_URI manually in Secret Manager later${NC}"
fi

# Setup OPEN_ROUTER_API_KEY secret
echo -e "${YELLOW}📝 Setting up OPEN_ROUTER_API_KEY secret...${NC}"
if [[ -f ".env" ]]; then
    OPEN_ROUTER_API_KEY=$(grep "OPEN_ROUTER_API_KEY" .env | cut -d '=' -f2)
    if [[ -n "$OPEN_ROUTER_API_KEY" ]]; then
        echo -e "${YELLOW}📝 Found OPEN_ROUTER_API_KEY in .env file${NC}"
        
        # Create secret if it doesn't exist
        if ! gcloud secrets describe "open-router-api-key" --project=$PROJECT_ID >/dev/null 2>&1; then
            echo -e "${YELLOW}🔐 Creating OPEN_ROUTER_API_KEY secret...${NC}"
            echo "$OPEN_ROUTER_API_KEY" | gcloud secrets create "open-router-api-key" \
                --data-file=- \
                --project=$PROJECT_ID \
                --replication-policy="automatic"
        else
            echo -e "${YELLOW}🔄 Updating existing OPEN_ROUTER_API_KEY secret...${NC}"
            echo "$OPEN_ROUTER_API_KEY" | gcloud secrets versions add "open-router-api-key" \
                --data-file=- \
                --project=$PROJECT_ID
        fi
        
        # Grant access to the Cloud Run service account
        SERVICE_ACCOUNT="ew-agent-service@${PROJECT_ID}.iam.gserviceaccount.com"
        gcloud secrets add-iam-policy-binding "open-router-api-key" \
            --member="serviceAccount:${SERVICE_ACCOUNT}" \
            --role="roles/secretmanager.secretAccessor" \
            --project=$PROJECT_ID
        
        echo -e "${GREEN}✅ OPEN_ROUTER_API_KEY secret configured${NC}"
    else
        echo -e "${YELLOW}⚠️  OPEN_ROUTER_API_KEY not found in .env file${NC}"
        echo -e "${YELLOW}   Reports system will use fallback formatting${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  .env file not found${NC}"
    echo -e "${YELLOW}   Reports system will use fallback formatting${NC}"
fi

# Step 4: Build optimized container
echo -e "${BLUE}🔨 Step 4: Building optimized container...${NC}"

# Create optimized Dockerfile if needed
cat > Dockerfile.optimized << 'EOF'
FROM python:3.12-slim

# Install system dependencies for ADK + Vertex AI + Reports
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

CMD ["python3", "-u", "main.py"]
EOF

echo -e "${YELLOW}📦 Building with optimized Dockerfile...${NC}"
mv Dockerfile.optimized Dockerfile
gcloud builds submit --tag $IMAGE_NAME .

echo -e "${GREEN}✅ Container built successfully${NC}"

# Step 5: Deploy with enhanced configuration including Reports System
echo -e "${BLUE}🚀 Step 5: Deploying with enhanced configuration...${NC}"

# First, clear existing environment variables to avoid type conflicts
echo -e "${YELLOW}🧹 Clearing existing environment variables...${NC}"
gcloud run services update $SERVICE_NAME \
    --region $REGION \
    --clear-env-vars

# Then deploy with new environment variables including Reports System
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
    --set-env-vars="GOOGLE_CLOUD_PROJECT=${PROJECT_ID}" \
    --set-env-vars="GOOGLE_CLOUD_LOCATION=${REGION}" \
    --set-env-vars="GOOGLE_GENAI_USE_VERTEXAI=TRUE" \
    --set-env-vars="VERTEX_AI_ENABLED=true" \
    --set-env-vars="ADK_VERTEX_INTEGRATION=enabled" \

    --set-env-vars="ADK_AGENTS_ENABLED=true" \
    --set-env-vars="ENHANCED_COORDINATOR_ENABLED=true" \
    --set-env-vars="REASONING_ENGINE_COMPATIBLE=true" \
    --set-env-vars="REPORTS_SYSTEM_ENABLED=true" \
    --set-env-vars="AI_REPORT_GENERATION_ENABLED=true" \
    --set-env-vars="CLOUD_RUN_MODE=true" \
    --service-account="ew-agent-service@${PROJECT_ID}.iam.gserviceaccount.com" \
    --execution-environment gen2 \
    --cpu-boost

echo -e "${GREEN}✅ Service deployed successfully${NC}"

# Step 6: Configure secret access
echo -e "${BLUE}🔐 Step 6: Configuring secret access...${NC}"

# Configure MongoDB URI secret access
gcloud run services update $SERVICE_NAME \
    --region $REGION \
    --update-secrets="MONGODB_ATLAS_URI=mongodb-atlas-uri:latest"

# Configure OpenRouter API key secret access (if available)
if [[ -n "$OPEN_ROUTER_API_KEY" ]]; then
    gcloud run services update $SERVICE_NAME \
        --region $REGION \
        --update-secrets="OPEN_ROUTER_API_KEY=open-router-api-key:latest"
fi

echo -e "${GREEN}✅ Secret access configured${NC}"

# Step 7: Get service information and test
echo -e "${BLUE}ℹ️  Step 7: Service validation...${NC}"

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
curl -s "${SERVICE_URL}/health" | head -5

echo -e "${YELLOW}📋 Testing reports system...${NC}"
curl -s "${SERVICE_URL}/api/reports/available" | head -5

echo -e "${YELLOW}📋 Testing storage connection...${NC}"
curl -s "${SERVICE_URL}/storage/stats" | head -5

echo ""
echo -e "${GREEN}🌐 Available Endpoints:${NC}"
echo -e "   Health Check: ${SERVICE_URL}/health"
echo -e "   Development UI: ${SERVICE_URL}/dev-ui"
echo -e "   Reports Interface: ${SERVICE_URL}/view_reports"
echo -e "   Available Reports: ${SERVICE_URL}/api/reports/available"
echo -e "   Storage Stats: ${SERVICE_URL}/storage/stats"
echo -e "   Analysis: ${SERVICE_URL}/AnalysePosts"
echo ""
echo -e "${GREEN}✨ ElectionWatch with Reports System is live!${NC}"

# Cleanup temporary files
rm -f Dockerfile.optimized

echo ""
echo -e "${BLUE}🎯 Next Steps:${NC}"
echo -e "   1. Test the reports system endpoints"
echo -e "   2. Generate and download sample reports"
echo -e "   3. Monitor performance metrics"
echo -e "   4. Scale based on traffic patterns"
echo -e "   5. Consider adding load balancing for production"
echo ""
echo -e "${BLUE}📊 Reports System Features:${NC}"
echo -e "   ✅ HTML report generation"
echo -e "   ✅ PDF export with enhanced styling"
echo -e "   ✅ Web interface for report management"
echo -e "   ✅ LLM integration for formatting"
echo -e "   ✅ MongoDB integration for persistence" 