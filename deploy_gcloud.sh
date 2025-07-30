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

# Configuration file path
CONFIG_FILE="deploy_config.env"

# Function to load configuration from file
load_config_from_file() {
    if [[ -f "$CONFIG_FILE" ]]; then
        echo -e "${BLUE}📁 Loading configuration from $CONFIG_FILE${NC}"
        # Source the config file, but only export variables that are set
        set -a  # automatically export all variables
        source "$CONFIG_FILE"
        set +a  # stop automatically exporting
        echo -e "${GREEN}✅ Configuration loaded from $CONFIG_FILE${NC}"
    else
        echo -e "${YELLOW}⚠️  Configuration file $CONFIG_FILE not found, using environment variables and defaults${NC}"
    fi
}

# Function to validate required configuration
validate_config() {
    local missing_vars=()
    
    # Check required variables
    if [[ -z "$PROJECT_ID" ]]; then
        missing_vars+=("PROJECT_ID")
    fi
    
    if [[ -z "$REGION" ]]; then
        missing_vars+=("REGION")
    fi
    
    if [[ -z "$SERVICE_NAME" ]]; then
        missing_vars+=("SERVICE_NAME")
    fi
    
    if [[ ${#missing_vars[@]} -gt 0 ]]; then
        echo -e "${RED}❌ Missing required configuration variables:${NC}"
        for var in "${missing_vars[@]}"; do
            echo -e "${RED}   - $var${NC}"
        done
        echo ""
        echo -e "${YELLOW}💡 Set these variables in your environment or create a $CONFIG_FILE file${NC}"
        echo -e "${YELLOW}   Example $CONFIG_FILE content:${NC}"
        echo -e "${BLUE}   PROJECT_ID=your-project-id${NC}"
        echo -e "${BLUE}   REGION=your-region${NC}"
        echo -e "${BLUE}   SERVICE_NAME=your-service-name${NC}"
        exit 1
    fi
}

# Load configuration
load_config_from_file

# Set configuration with environment variables and defaults
PROJECT_ID="${PROJECT_ID:-ew-agents-v02}"
REGION="${REGION:-europe-west1}"
SERVICE_NAME="${SERVICE_NAME:-electionwatch-misinformation-api}"
MEMORY="${MEMORY:-2Gi}"
CPU="${CPU:-1}"
MAX_INSTANCES="${MAX_INSTANCES:-10}"
MIN_INSTANCES="${MIN_INSTANCES:-1}"

# Validate configuration
validate_config

# Derived configuration
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo -e "${BLUE}🛡️  ElectionWatch Misinformation Detection API - Cloud Deployment${NC}"
echo -e "${BLUE}=========================================================${NC}"
echo ""

# Display configuration
echo -e "${BLUE}📋 Configuration:${NC}"
echo -e "   Project ID: $PROJECT_ID"
echo -e "   Region: $REGION"
echo -e "   Service Name: $SERVICE_NAME"
echo -e "   Image Name: $IMAGE_NAME"
echo -e "   Memory: $MEMORY"
echo -e "   CPU: $CPU"
echo -e "   Min Instances: $MIN_INSTANCES"
echo -e "   Max Instances: $MAX_INSTANCES"
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
    containerregistry.googleapis.com \
    aiplatform.googleapis.com \
    vertex-ai.googleapis.com \
    ml.googleapis.com \
    secretmanager.googleapis.com

echo -e "${GREEN}✅ gcloud configuration complete${NC}"
echo ""

# Step 2.5: Set up Secret Manager (if needed)
echo -e "${BLUE}🔐 Step 2.5: Setting up Secret Manager...${NC}"

# Check if MongoDB secret exists
if gcloud secrets describe mongodb-atlas-uri --project=$PROJECT_ID >/dev/null 2>&1; then
    echo -e "${GREEN}✅ MongoDB secret already exists in Secret Manager${NC}"
else
    echo -e "${YELLOW}⚠️ MongoDB secret not found in Secret Manager${NC}"
    echo -e "${YELLOW}💡 You can create it manually using:${NC}"
    echo -e "${BLUE}   gcloud secrets create mongodb-atlas-uri --project=$PROJECT_ID${NC}"
    echo -e "${BLUE}   echo 'your-mongodb-connection-string' | gcloud secrets versions add mongodb-atlas-uri --data-file=- --project=$PROJECT_ID${NC}"
    echo -e "${YELLOW}   Or set MONGODB_ATLAS_URI environment variable for local development${NC}"
fi
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

# First, clear existing environment variables to avoid type conflicts
echo -e "${YELLOW}🧹 Clearing existing environment variables...${NC}"
gcloud run services update $SERVICE_NAME \
    --region $REGION \
    --clear-env-vars

# Then deploy with new environment variables
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
    --set-env-vars="GOOGLE_CLOUD_PROJECT=${PROJECT_ID}" \
    --set-env-vars="GOOGLE_CLOUD_LOCATION=${REGION}" \
    --set-env-vars="VERTEX_AI_ENABLED=true" \
    --set-env-vars="ADK_VERTEX_INTEGRATION=enabled" \
    --set-env-vars="ADK_AGENTS_ENABLED=true" \
    --set-env-vars="ENHANCED_COORDINATOR_ENABLED=true" \
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