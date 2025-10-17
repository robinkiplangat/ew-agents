#!/bin/bash

# ElectionWatch Optimized Cloud Run Deployment
# Delegates to unified build_and_deploy.sh; keeps secrets setup as optional add-on

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration (can be overridden via env)
PROJECT_ID="${PROJECT_ID:-ew-agents-v02}"
REGION="${REGION:-us-central1}"
SERVICE_NAME="${SERVICE_NAME:-EMERY-agents-api}"
MEMORY="${MEMORY:-2Gi}"
CPU="${CPU:-1}"
MAX_INSTANCES="${MAX_INSTANCES:-5}"
MIN_INSTANCES="${MIN_INSTANCES:-1}"
CONCURRENCY="${CONCURRENCY:-100}"
TIMEOUT="${TIMEOUT:-3600}"
PORT="${PORT:-8080}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-ew-agent-service@${PROJECT_ID}.iam.gserviceaccount.com}"

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

echo -e "${BLUE}⚙️  Step 2: Deploying core service via unified script...${NC}"
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
UNIFIED_SCRIPT="${SCRIPT_DIR}/build_and_deploy.sh"
if [[ ! -x "${UNIFIED_SCRIPT}" ]]; then
  echo -e "${RED}❌ ${UNIFIED_SCRIPT} not found or not executable${NC}"
  exit 1
fi

"${UNIFIED_SCRIPT}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --service "${SERVICE_NAME}" \
  --memory "${MEMORY}" \
  --cpu "${CPU}" \
  --min-instances "${MIN_INSTANCES}" \
  --max-instances "${MAX_INSTANCES}" \
  --concurrency "${CONCURRENCY}" \
  --timeout "${TIMEOUT}" \
  --port "${PORT}" \
  --sa "${SERVICE_ACCOUNT}"

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

# Optional advanced features below (Secrets). Core deploy above already done.

echo -e "${GREEN}✅ Core service deployed (via unified script)${NC}"

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

true

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