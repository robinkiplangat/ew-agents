#!/bin/bash

# ElectionWatch Full System Deployment Script
# Deploys agents to Vertex AI Agent Engine and starts FastAPI gateway

set -e

echo "🚀 ElectionWatch Full System Deployment"
echo "========================================"

# Configuration
PROJECT_ID="${1:-ew-agents-v01}"
LOCATION="${2:-us-central1}"
DEPLOY_AGENTS="${3:-true}"

echo "📊 Configuration:"
echo "   Project ID: $PROJECT_ID"
echo "   Location: $LOCATION"
echo "   Deploy Agents: $DEPLOY_AGENTS"
echo

# Set environment variables
export GOOGLE_CLOUD_PROJECT="$PROJECT_ID"
export GOOGLE_CLOUD_LOCATION="$LOCATION"

# Step 1: Deploy agents to Vertex AI Agent Engine (if requested)
if [ "$DEPLOY_AGENTS" = "true" ]; then
    echo "🤖 Step 1: Deploying agents to Vertex AI Agent Engine"
    echo "=================================================="
    
    cd ../deployment
    
    # Check if we have the deployment script
    if [ -f "deploy_agents.py" ]; then
        echo "📤 Deploying coordinator agent..."
        python deploy_agents.py --agent coordinator_agent
        
        echo "✅ Agent deployment completed"
    else
        echo "⚠️  Agent deployment script not found, skipping agent deployment"
        echo "   Agents will run in mock mode"
    fi
    
    cd ../app
else
    echo "⏭️  Skipping agent deployment (will use mock mode)"
fi

echo
echo "🌐 Step 2: Starting FastAPI Gateway"
echo "================================="

# Check if agent_client.py exists
if [ ! -f "agent_client.py" ]; then
    echo "❌ Error: agent_client.py not found"
    echo "   Please ensure the agent client is properly set up"
    exit 1
fi

# Install dependencies if needed
echo "📦 Installing dependencies..."
pip install -q fastapi uvicorn google-cloud-aiplatform

# Start the FastAPI gateway
echo "🚀 Starting ElectionWatch Agent Gateway..."
echo "   Mode: Gateway to deployed agents"
echo "   URL: http://localhost:8080"
echo "   Docs: http://localhost:8080/docs"
echo

# Set the port from environment or default to 8080
PORT="${PORT:-8080}"

echo "🔧 Environment Check:"
echo "   GOOGLE_CLOUD_PROJECT: $GOOGLE_CLOUD_PROJECT"
echo "   GOOGLE_CLOUD_LOCATION: $GOOGLE_CLOUD_LOCATION"
echo "   PORT: $PORT"
echo

# Start the server
echo "▶️  Starting server..."
python simple_agent_api.py

echo "🏁 Deployment completed!" 