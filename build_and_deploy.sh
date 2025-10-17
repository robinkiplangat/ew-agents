#!/bin/bash

# ElectionWatch Unified Build and Deploy
# Builds Docker image and deploys to Cloud Run with configurable flags/env
# Usage examples:
#   ./build_and_deploy.sh
#   ./build_and_deploy.sh --project my-proj --region europe-west1 --service my-svc \
#     --memory 2Gi --cpu 1 --min-instances 1 --max-instances 5 --sa my-sa@my-proj.iam.gserviceaccount.com

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Defaults (override via flags or environment)
PROJECT_ID=${PROJECT_ID:-"ew-agents-v02"}
REGION=${REGION:-"us-central1"}
SERVICE_NAME=${SERVICE_NAME:-"10-08-agents-api"}
IMAGE_REPO_HOST=${IMAGE_REPO_HOST:-"gcr.io"}
BUILD_TAG=${BUILD_TAG:-"latest"}
MEMORY=${MEMORY:-"16Gi"}
CPU=${CPU:-"1"}
MIN_INSTANCES=${MIN_INSTANCES:-"1"}
MAX_INSTANCES=${MAX_INSTANCES:-"1"}
CONCURRENCY=${CONCURRENCY:-"100"}
TIMEOUT=${TIMEOUT:-"3600"}
PORT=${PORT:-"8080"}
SERVICE_ACCOUNT=${SERVICE_ACCOUNT:-""}
ENABLE_APIS=${ENABLE_APIS:-"true"}
SKIP_TEST=${SKIP_TEST:-"false"}

print_help() {
  cat <<EOF
ElectionWatch Unified Build and Deploy

Flags (or set as env vars):
  --project|-p              GCP project ID (default: ${PROJECT_ID})
  --region|-r               GCP region (default: ${REGION})
  --service|-s              Cloud Run service name (default: ${SERVICE_NAME})
  --memory                  Memory (default: ${MEMORY})
  --cpu                     vCPU (default: ${CPU})
  --min-instances           Minimum instances (default: ${MIN_INSTANCES})
  --max-instances           Maximum instances (default: ${MAX_INSTANCES})
  --concurrency             Request concurrency (default: ${CONCURRENCY})
  --timeout                 Request timeout seconds (default: ${TIMEOUT})
  --port                    Container port (default: ${PORT})
  --sa                      Service account email (optional)
  --image-repo-host         Image repo host gcr.io|us-docker.pkg.dev (default: ${IMAGE_REPO_HOST})
  --build-tag               Image tag (default: ${BUILD_TAG})
  --no-enable-apis          Do not enable APIs
  --skip-test               Skip /health test
  -h|--help                 Show help
EOF
}

# Parse flags
while [[ $# -gt 0 ]]; do
  case "$1" in
    --project|-p) PROJECT_ID="$2"; shift 2;;
    --region|-r) REGION="$2"; shift 2;;
    --service|-s) SERVICE_NAME="$2"; shift 2;;
    --memory) MEMORY="$2"; shift 2;;
    --cpu) CPU="$2"; shift 2;;
    --min-instances) MIN_INSTANCES="$2"; shift 2;;
    --max-instances) MAX_INSTANCES="$2"; shift 2;;
    --concurrency) CONCURRENCY="$2"; shift 2;;
    --timeout) TIMEOUT="$2"; shift 2;;
    --port) PORT="$2"; shift 2;;
    --sa) SERVICE_ACCOUNT="$2"; shift 2;;
    --image-repo-host) IMAGE_REPO_HOST="$2"; shift 2;;
    --build-tag) BUILD_TAG="$2"; shift 2;;
    --no-enable-apis) ENABLE_APIS="false"; shift;;
    --skip-test) SKIP_TEST="true"; shift;;
    -h|--help) print_help; exit 0;;
    *) echo -e "${RED}Unknown flag: $1${NC}"; print_help; exit 1;;
  esac
done

IMAGE_NAME="${IMAGE_REPO_HOST}/${PROJECT_ID}/${SERVICE_NAME}"

echo -e "${BLUE}🚀 ElectionWatch Build and Deploy${NC}"
echo -e "${BLUE}================================${NC}"
echo ""
echo -e "Configuration:"
echo -e "  Project: ${PROJECT_ID}"
echo -e "  Region: ${REGION}"
echo -e "  Service: ${SERVICE_NAME}"
echo -e "  Image: ${IMAGE_NAME}:${BUILD_TAG}"
echo -e "  Memory: ${MEMORY}, CPU: ${CPU}, Min: ${MIN_INSTANCES}, Max: ${MAX_INSTANCES}"
echo -e "  Concurrency: ${CONCURRENCY}, Timeout: ${TIMEOUT}s, Port: ${PORT}"
if [[ -n "${SERVICE_ACCOUNT}" ]]; then
  echo -e "  Service Account: ${SERVICE_ACCOUNT}"
fi
echo ""

# Step 1: Configure gcloud
echo -e "${BLUE}⚙️  Step 1: Configuring gcloud...${NC}"
gcloud config set project ${PROJECT_ID}
gcloud config set run/region ${REGION}
echo -e "${GREEN}✅ gcloud configured${NC}"

# Step 2: Enable required APIs (optional)
if [[ "${ENABLE_APIS}" == "true" ]]; then
  echo -e "${BLUE}🔧 Step 2: Enabling required APIs...${NC}"
  gcloud services enable \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    containerregistry.googleapis.com \
    aiplatform.googleapis.com \
    secretmanager.googleapis.com || true
  echo -e "${GREEN}✅ APIs enable attempted${NC}"
else
  echo -e "${YELLOW}⏭️  Skipping API enable step (--no-enable-apis)${NC}"
fi

# Step 2.5: Verify authentication and permissions
echo -e "${BLUE}🔐 Step 2.5: Verifying authentication...${NC}"
CURRENT_USER=$(gcloud auth list --filter=status:ACTIVE --format="value(account)" || true)
echo -e "${YELLOW}Authenticated as: ${CURRENT_USER:-unknown}${NC}"
echo -e "${YELLOW}Verifying project access...${NC}"
if gcloud projects describe ${PROJECT_ID} >/dev/null 2>&1; then
  echo -e "${GREEN}✅ Project access confirmed${NC}"
else
  echo -e "${RED}❌ No access to project ${PROJECT_ID}${NC}"
  echo -e "${YELLOW}Tip: gcloud auth login && gcloud auth application-default login${NC}"
  exit 1
fi

# Step 3: Build Docker image
echo -e "${BLUE}🔨 Step 3: Building Docker image...${NC}"
echo -e "${YELLOW}Building image: ${IMAGE_NAME}:${BUILD_TAG}${NC}"
gcloud builds submit \
  --tag ${IMAGE_NAME}:${BUILD_TAG} \
  --timeout=20m \
  --machine-type=e2-highcpu-8
echo -e "${GREEN}✅ Docker image built successfully${NC}"

# Step 4: Deploy to Cloud Run
echo -e "${BLUE}🚀 Step 4: Deploying to Cloud Run...${NC}"
DEPLOY_CMD=(
  gcloud run deploy ${SERVICE_NAME}
  --image ${IMAGE_NAME}:${BUILD_TAG}
  --platform managed
  --region ${REGION}
  --allow-unauthenticated
  --memory ${MEMORY}
  --cpu ${CPU}
  --min-instances ${MIN_INSTANCES}
  --max-instances ${MAX_INSTANCES}
  --timeout ${TIMEOUT}
  --concurrency ${CONCURRENCY}
  --port ${PORT}
  --set-env-vars "GOOGLE_CLOUD_PROJECT=${PROJECT_ID}"
  --set-env-vars "GOOGLE_CLOUD_LOCATION=${REGION}"
  --set-env-vars "VERTEX_AI_ENABLED=true"
  --set-env-vars "ADK_VERTEX_INTEGRATION=enabled"
  --set-env-vars "ADK_AGENTS_ENABLED=true"
)
if [[ -n "${SERVICE_ACCOUNT}" ]]; then
  DEPLOY_CMD+=(--service-account "${SERVICE_ACCOUNT}")
fi

"${DEPLOY_CMD[@]}"
echo -e "${GREEN}✅ Service deployed successfully${NC}"

# Step 5: Get service information
echo -e "${BLUE}ℹ️  Step 5: Service information...${NC}"
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --region=${REGION} --format="value(status.url)")

echo ""
echo -e "${GREEN}🎉 Build and Deploy completed successfully!${NC}"
echo ""
echo -e "${GREEN}📍 Service Details:${NC}"
echo -e "   Service Name: ${SERVICE_NAME}"
echo -e "   URL: ${SERVICE_URL}"
echo -e "   Image: ${IMAGE_NAME}:${BUILD_TAG}"
echo -e "   Memory: ${MEMORY}"
echo -e "   CPU: ${CPU}"
echo -e "   Scaling: ${MIN_INSTANCES}-${MAX_INSTANCES} instances"
echo ""

# Step 6: Test deployment (optional)
if [[ "${SKIP_TEST}" != "true" ]]; then
  echo -e "${BLUE}🧪 Step 6: Testing deployment...${NC}"
  echo -e "${YELLOW}Testing health endpoint...${NC}"
  if curl -s "${SERVICE_URL}/health" > /dev/null; then
    echo -e "${GREEN}✅ Health check passed${NC}"
  else
    echo -e "${YELLOW}⚠️  Health check failed or endpoint not ready${NC}"
  fi

  echo ""
  echo -e "${GREEN}🌐 Available Endpoints:${NC}"
  echo -e "   Health Check: ${SERVICE_URL}/health"
  echo -e "   API Docs: ${SERVICE_URL}/docs"
  echo -e "   Analysis: ${SERVICE_URL}/run_analysis"
fi

echo ""
echo -e "${GREEN}✨ ElectionWatch deployment complete${NC}"
