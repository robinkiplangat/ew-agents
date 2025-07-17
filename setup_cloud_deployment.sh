#!/bin/bash

# ElectionWatch Cloud Deployment Setup Script
# Creates service accounts, sets up IAM permissions, and enables required APIs

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ID="${1:-ew-agents-v02}"
REGION="${2:-europe-west1}"
SERVICE_ACCOUNT_NAME="ew-agent-service"
CLOUD_RUN_SERVICE="ew-agent-service"

echo -e "${BLUE}🚀 ElectionWatch Cloud Deployment Setup${NC}"
echo -e "${BLUE}=======================================${NC}"
echo -e "📋 Project ID: ${GREEN}$PROJECT_ID${NC}"
echo -e "📍 Region: ${GREEN}$REGION${NC}"
echo -e "🔑 Service Account: ${GREEN}$SERVICE_ACCOUNT_NAME${NC}"
echo ""

# Function to check if command succeeded
check_command() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ $1${NC}"
    else
        echo -e "${RED}❌ $1 failed${NC}"
        exit 1
    fi
}

# Check prerequisites
echo -e "${YELLOW}🔧 Checking prerequisites...${NC}"

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}❌ gcloud CLI is not installed${NC}"
    echo "Please install it from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if user is authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" &> /dev/null; then
    echo -e "${RED}❌ Not authenticated with gcloud${NC}"
    echo "Please run: gcloud auth login"
    exit 1
fi

echo -e "${GREEN}✅ Prerequisites check passed${NC}"
echo ""

# Set the project
echo -e "${YELLOW}🔧 Setting up Google Cloud project...${NC}"
gcloud config set project $PROJECT_ID
check_command "Project set to $PROJECT_ID"

# Enable required APIs
echo -e "${YELLOW}🔌 Enabling required APIs...${NC}"

APIs=(
    "run.googleapis.com"
    "cloudbuild.googleapis.com"
    "artifactregistry.googleapis.com"
    "aiplatform.googleapis.com"
    "secretmanager.googleapis.com"
    "cloudresourcemanager.googleapis.com"
    "iam.googleapis.com"
    "logging.googleapis.com"
    "monitoring.googleapis.com"
    "compute.googleapis.com"
)

for api in "${APIs[@]}"; do
    echo "Enabling $api..."
    gcloud services enable $api --quiet
    check_command "Enabled $api"
done

echo ""

# Create service account
echo -e "${YELLOW}👤 Creating service account...${NC}"

# Check if service account already exists
if gcloud iam service-accounts describe "${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com" &> /dev/null; then
    echo -e "${YELLOW}⚠️  Service account already exists${NC}"
else
    gcloud iam service-accounts create $SERVICE_ACCOUNT_NAME \
        --display-name="ElectionWatch Agent Service Account" \
        --description="Service account for ElectionWatch multi-agent system"
    check_command "Created service account $SERVICE_ACCOUNT_NAME"
fi

SERVICE_ACCOUNT_EMAIL="${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
echo -e "📧 Service Account Email: ${GREEN}$SERVICE_ACCOUNT_EMAIL${NC}"

# Assign IAM roles
echo -e "${YELLOW}🔐 Assigning IAM roles...${NC}"

IAM_ROLES=(
    "roles/run.admin"
    "roles/aiplatform.user"
    "roles/aiplatform.admin"
    "roles/storage.admin"
    "roles/secretmanager.accessor"
    "roles/logging.logWriter"
    "roles/monitoring.metricWriter"
    "roles/cloudsql.client"
    "roles/pubsub.editor"
    "roles/serviceusage.serviceUsageViewer"
    "roles/cloudtrace.agent"
    "roles/clouddebugger.agent"
)

for role in "${IAM_ROLES[@]}"; do
    echo "Assigning role: $role"
    gcloud projects add-iam-policy-binding $PROJECT_ID \
        --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
        --role="$role" \
        --quiet
    check_command "Assigned $role"
done

echo ""

# Create and download service account key
echo -e "${YELLOW}🗝️  Creating service account key...${NC}"

KEY_FILE="ew-agent-service-key.json"
if [ -f "$KEY_FILE" ]; then
    echo -e "${YELLOW}⚠️  Key file already exists, backing up...${NC}"
    mv "$KEY_FILE" "${KEY_FILE}.backup.$(date +%s)"
fi

gcloud iam service-accounts keys create $KEY_FILE \
    --iam-account=$SERVICE_ACCOUNT_EMAIL
check_command "Created service account key: $KEY_FILE"

echo ""

# Set up default credentials
echo -e "${YELLOW}🔑 Setting up application default credentials...${NC}"
export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/$KEY_FILE"
echo "export GOOGLE_APPLICATION_CREDENTIALS=\"$(pwd)/$KEY_FILE\"" >> ~/.bashrc
echo "export GOOGLE_CLOUD_PROJECT=\"$PROJECT_ID\"" >> ~/.bashrc
echo "export GOOGLE_CLOUD_LOCATION=\"$REGION\"" >> ~/.bashrc

check_command "Application default credentials configured"

# Create Cloud Storage bucket for artifacts
echo -e "${YELLOW}🪣 Creating Cloud Storage bucket...${NC}"

BUCKET_NAME="${PROJECT_ID}-ew-artifacts"
if gsutil ls -b gs://$BUCKET_NAME &> /dev/null; then
    echo -e "${YELLOW}⚠️  Bucket already exists: gs://$BUCKET_NAME${NC}"
else
    gsutil mb -p $PROJECT_ID -c STANDARD -l $REGION gs://$BUCKET_NAME
    check_command "Created bucket: gs://$BUCKET_NAME"
fi

# Set bucket permissions
gsutil iam ch serviceAccount:$SERVICE_ACCOUNT_EMAIL:objectAdmin gs://$BUCKET_NAME
check_command "Set bucket permissions"

echo ""

# Create environment configuration
echo -e "${YELLOW}⚙️  Creating environment configuration...${NC}"

cat > .env.production << EOF
# ElectionWatch Production Environment Configuration
GOOGLE_CLOUD_PROJECT=$PROJECT_ID
GOOGLE_CLOUD_LOCATION=$REGION
GOOGLE_APPLICATION_CREDENTIALS=$(pwd)/$KEY_FILE

# Service Configuration
PORT=8080
ENVIRONMENT=production
LOG_LEVEL=INFO

# Cloud Run Configuration
SERVICE_NAME=$CLOUD_RUN_SERVICE
SERVICE_ACCOUNT_EMAIL=$SERVICE_ACCOUNT_EMAIL

# Storage Configuration
BUCKET_NAME=$BUCKET_NAME

# Agent Configuration
AGENT_MODE=production
MAX_WORKERS=4
TIMEOUT_SECONDS=300
EOF

check_command "Created .env.production file"

# Create deployment configuration
cat > deployment_config.yaml << EOF
# ElectionWatch Deployment Configuration
project_id: $PROJECT_ID
region: $REGION
service_name: $CLOUD_RUN_SERVICE
service_account: $SERVICE_ACCOUNT_EMAIL
image_name: gcr.io/$PROJECT_ID/ew-agent-service

# Cloud Run Configuration
cloud_run:
  memory: "2Gi"
  cpu: "1"
  max_instances: 10
  min_instances: 0
  concurrency: 100
  timeout: 300

# Environment Variables
environment:
  GOOGLE_CLOUD_PROJECT: $PROJECT_ID
  GOOGLE_CLOUD_LOCATION: $REGION
  ENVIRONMENT: production
  LOG_LEVEL: INFO
  AGENT_MODE: production

# Monitoring
monitoring:
  enabled: true
  log_level: INFO
  health_check_path: /health
EOF

check_command "Created deployment_config.yaml"

echo ""

# Security recommendations
echo -e "${YELLOW}🔒 Security Setup${NC}"

echo "Creating Cloud Armor security policy..."
if gcloud compute security-policies describe ew-agent-security-policy --global &> /dev/null; then
    echo -e "${YELLOW}⚠️  Security policy already exists${NC}"
else
    gcloud compute security-policies create ew-agent-security-policy \
        --description="Security policy for ElectionWatch Agent API" \
        --global
    check_command "Created security policy"
    
    # Add rate limiting rule
    gcloud compute security-policies rules create 1000 \
        --security-policy=ew-agent-security-policy \
        --expression="true" \
        --action="rate-based-ban" \
        --rate-limit-threshold-count=100 \
        --rate-limit-threshold-interval-sec=60 \
        --ban-duration-sec=600 \
        --conform-action=allow \
        --exceed-action=deny-429 \
        --enforce-on-key=IP \
        --global
    check_command "Added rate limiting rule"
fi

echo ""

# Summary
echo -e "${BLUE}📋 Deployment Setup Complete!${NC}"
echo -e "${BLUE}=============================${NC}"
echo ""
echo -e "${GREEN}✅ Created service account:${NC} $SERVICE_ACCOUNT_EMAIL"
echo -e "${GREEN}✅ Assigned IAM roles:${NC} ${#IAM_ROLES[@]} roles"
echo -e "${GREEN}✅ Enabled APIs:${NC} ${#APIs[@]} APIs"
echo -e "${GREEN}✅ Created storage bucket:${NC} gs://$BUCKET_NAME"
echo -e "${GREEN}✅ Generated service key:${NC} $KEY_FILE"
echo -e "${GREEN}✅ Created configuration files${NC}"
echo ""

echo -e "${YELLOW}🚀 Next Steps:${NC}"
echo -e "1. Source the environment variables:"
echo -e "   ${BLUE}source .env.production${NC}"
echo ""
echo -e "2. Test your credentials:"
echo -e "   ${BLUE}gcloud auth application-default print-access-token${NC}"
echo ""
echo -e "3. Deploy your agent:"
echo -e "   ${BLUE}adk deploy cloud_run \\${NC}"
echo -e "   ${BLUE}  --project=$PROJECT_ID \\${NC}"
echo -e "   ${BLUE}  --region=$REGION \\${NC}"
echo -e "   ${BLUE}  --service_name=$CLOUD_RUN_SERVICE \\${NC}"
echo -e "   ${BLUE}  --app_name=ew-agent-app \\${NC}"
echo -e "   ${BLUE}  --with_ui \\${NC}"
echo -e "   ${BLUE}  main:coordinator_agent${NC}"
echo ""
echo -e "4. Or use the Cloud Run deployment:"
echo -e "   ${BLUE}./deploy.sh $PROJECT_ID $REGION${NC}"
echo ""

echo -e "${YELLOW}📁 Important Files Created:${NC}"
echo -e "• ${GREEN}$KEY_FILE${NC} - Service account credentials (keep secure!)"
echo -e "• ${GREEN}.env.production${NC} - Environment configuration"
echo -e "• ${GREEN}deployment_config.yaml${NC} - Deployment settings"
echo ""

echo -e "${RED}⚠️  Security Notes:${NC}"
echo -e "• Keep ${GREEN}$KEY_FILE${NC} secure and never commit to version control"
echo -e "• Consider using Workload Identity for production"
echo -e "• Review IAM permissions periodically"
echo -e "• Enable audit logging for compliance"
echo ""

echo -e "${GREEN}🎉 Setup completed successfully!${NC}" 