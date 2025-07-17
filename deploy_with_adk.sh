#!/bin/bash

# ElectionWatch ADK Cloud Run Deployment Script
# Uses Google Agent Development Kit to deploy the coordinator agent

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ID="${1:-ew-agents-v01}"
REGION="${2:-europe-west1}"
SERVICE_NAME="${3:-ew-agent-service}"
APP_NAME="${4:-ew-agent-app}"

echo -e "${BLUE}🚀 ElectionWatch ADK Cloud Run Deployment${NC}"
echo -e "${BLUE}=========================================${NC}"
echo -e "📋 Project ID: ${GREEN}$PROJECT_ID${NC}"
echo -e "📍 Region: ${GREEN}$REGION${NC}"
echo -e "🏷️  Service Name: ${GREEN}$SERVICE_NAME${NC}"
echo -e "📱 App Name: ${GREEN}$APP_NAME${NC}"
echo ""

# Check if setup was completed
if [ ! -f ".env.production" ]; then
    echo -e "${RED}❌ .env.production not found${NC}"
    echo -e "Please run the setup script first:"
    echo -e "   ${BLUE}./setup_cloud_deployment.sh $PROJECT_ID $REGION${NC}"
    exit 1
fi

# Source environment variables
echo -e "${YELLOW}⚙️  Loading environment configuration...${NC}"
source .env.production
echo -e "${GREEN}✅ Environment loaded${NC}"

# Verify service account key exists
if [ ! -f "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo -e "${RED}❌ Service account key not found: $GOOGLE_APPLICATION_CREDENTIALS${NC}"
    echo -e "Please run the setup script to create the service account"
    exit 1
fi

# Set environment variables for ADK
export GOOGLE_CLOUD_PROJECT="$PROJECT_ID"
export GOOGLE_CLOUD_LOCATION="$REGION"

echo -e "${YELLOW}🔑 Setting up authentication...${NC}"

# Authenticate using service account key
gcloud auth activate-service-account --key-file="$GOOGLE_APPLICATION_CREDENTIALS"
gcloud config set project "$PROJECT_ID"

echo -e "${GREEN}✅ Authentication configured${NC}"

# Check if ADK is installed
echo -e "${YELLOW}🔧 Checking ADK installation...${NC}"
if ! command -v adk &> /dev/null; then
    echo -e "${RED}❌ ADK (Agent Development Kit) is not installed${NC}"
    echo -e "Please install ADK using:"
    echo -e "   ${BLUE}pip install google-adk${NC}"
    exit 1
fi

echo -e "${GREEN}✅ ADK is available${NC}"

# Install dependencies first
echo -e "${YELLOW}📦 Installing updated dependencies...${NC}"
pip install -r requirements.txt
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Dependencies installed successfully${NC}"
else
    echo -e "${RED}❌ Failed to install dependencies${NC}"
    echo -e "Dependency conflict may still exist. Trying with --force-reinstall..."
    pip install --force-reinstall google-cloud-aiplatform==1.95.1
    pip install --force-reinstall google-adk==1.5.0
fi

# Test agent loading
echo -e "${YELLOW}🧪 Testing agent loading...${NC}"
python -c "
import sys
sys.path.append('.')
try:
    from ew_agents.election_watch_agents import coordinator_agent
    from main import coordinator_agent as main_coordinator
    print('✅ Agent import successful')
except Exception as e:
    print(f'❌ Agent import failed: {e}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Agent loading test failed${NC}"
    exit 1
fi

# Deploy using ADK
echo -e "${YELLOW}🚀 Deploying with ADK...${NC}"

# Create temporary agent configuration
cat > agent_config.yaml << EOF
name: $SERVICE_NAME
app_name: $APP_NAME
project_id: $PROJECT_ID
region: $REGION
service_account: $SERVICE_ACCOUNT_EMAIL

# Cloud Run Configuration
cloud_run:
  memory: "2Gi"
  cpu: "1"
  max_instances: 10
  min_instances: 0
  concurrency: 100
  timeout: 300
  
# Environment variables
environment:
  GOOGLE_CLOUD_PROJECT: $PROJECT_ID
  GOOGLE_CLOUD_LOCATION: $REGION
  ENVIRONMENT: production
  LOG_LEVEL: INFO
  AGENT_MODE: production
EOF

echo -e "${BLUE}Running ADK deployment command...${NC}"

# Run the ADK deployment command
adk deploy cloud_run \
  --project="$PROJECT_ID" \
  --region="$REGION" \
  --service_name="$SERVICE_NAME" \
  --app_name="$APP_NAME" \
  --with_ui \
  .
DEPLOYMENT_STATUS=$?

if [ $DEPLOYMENT_STATUS -eq 0 ]; then
    echo -e "${GREEN}✅ ADK deployment completed successfully!${NC}"
    
    # Get the deployed service URL
    SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)")
    
    echo ""
    echo -e "${BLUE}🎉 Deployment Summary${NC}"
    echo -e "${BLUE}===================${NC}"
    echo -e "${GREEN}✅ Service deployed:${NC} $SERVICE_NAME"
    echo -e "${GREEN}✅ Service URL:${NC} $SERVICE_URL"
    echo -e "${GREEN}✅ Project:${NC} $PROJECT_ID"
    echo -e "${GREEN}✅ Region:${NC} $REGION"
    echo ""
    
    echo -e "${YELLOW}🔗 Access Your Service:${NC}"
    echo -e "• Service URL: ${BLUE}$SERVICE_URL${NC}"
    echo -e "• Health Check: ${BLUE}$SERVICE_URL/health${NC}"
    echo -e "• API Docs: ${BLUE}$SERVICE_URL/docs${NC}"
    echo ""
    
    echo -e "${YELLOW}🧪 Test Your Deployment:${NC}"
    echo -e "curl -X GET \"$SERVICE_URL/health\""
    echo ""
    echo -e "curl -X POST \"$SERVICE_URL/analyze\" \\"
    echo -e "  -H \"Content-Type: application/json\" \\"
    echo -e "  -d '{\"content\": \"Test analysis\", \"source_platform\": \"test\"}'"
    echo ""
    
    # Test the deployment
    echo -e "${YELLOW}🧪 Testing deployment...${NC}"
    sleep 10  # Wait for service to be ready
    
    if curl -s "$SERVICE_URL/health" > /dev/null; then
        echo -e "${GREEN}✅ Service is responding${NC}"
    else
        echo -e "${YELLOW}⚠️  Service may still be starting up${NC}"
    fi
    
else
    echo -e "${RED}❌ ADK deployment failed${NC}"
    echo -e "Check the logs above for error details"
    exit 1
fi

# Cleanup temporary files
rm -f agent_config.yaml

echo -e "${GREEN}🎉 Deployment process completed!${NC}" 