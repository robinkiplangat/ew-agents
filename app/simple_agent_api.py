"""
ElectionWatch Simple Agent API Service

A simplified FastAPI application for running the ElectionWatch multi-agent system.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Add parent directory to Python path to access ew_agents
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import agent client for calling deployed agents
try:
    from agent_client import agent_client, AgentResponse
    AGENT_CLIENT_AVAILABLE = True
    logger.info("✅ Agent client initialized for gateway mode")
except ImportError as e:
    logger.error(f"❌ Failed to import agent client: {e}")
    AGENT_CLIENT_AVAILABLE = False
    agent_client = None

# Create FastAPI app
app = FastAPI(
    title="ElectionWatch Simple Agent API",
    description="Simplified multi-agent system for election monitoring",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class AnalysisRequest(BaseModel):
    content: str
    source_platform: str = "unknown"
    request_type: str = "auto_detect"

class AnalysisResponse(BaseModel):
    status: str
    content_preview: str
    source_platform: str
    analysis: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class FindingSubmission(BaseModel):
    content_text: str
    source_platform: str
    source_url: Optional[str] = None
    author_handle: Optional[str] = None

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "ElectionWatch Agent Gateway API",
        "version": "1.0.0",
        "mode": "gateway",
        "status": "running",
        "agent_client_available": AGENT_CLIENT_AVAILABLE,
        "endpoints": {
            "analyze": "/analyze",
            "submit_finding": "/submit_finding",
            "health": "/health",
            "agents": "/agents"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if AGENT_CLIENT_AVAILABLE else "degraded",
        "gateway_mode": True,
        "agent_client_available": AGENT_CLIENT_AVAILABLE,
        "components": await agent_client.get_agent_status() if AGENT_CLIENT_AVAILABLE and agent_client else {
            "error": "Agent client not available"
        }
    }

@app.get("/agents")
async def list_agents():
    """List available deployed agents"""
    if not AGENT_CLIENT_AVAILABLE:
        raise HTTPException(status_code=503, detail="Agent client not available")
    
    try:
        deployed_agents = await agent_client.list_deployed_agents() if agent_client else []
        
        agent_descriptions = {
            "coordinator_agent": "Central orchestrator for the ElectionWatch system",
            "data_eng_agent": "Data collection, cleaning, and database management",
            "osint_agent": "OSINT analysis and narrative classification",
            "lexicon_agent": "Multilingual lexicon management",
            "trend_analysis_agent": "Narrative trend analysis and early warnings"
        }
        
        return {
            "mode": "gateway",
            "agents": [
                {
                    "name": agent["name"],
                    "description": agent_descriptions.get(agent["name"], "ElectionWatch specialist agent"),
                    "status": agent["status"],
                    "endpoint": agent["endpoint"],
                    "available": agent["status"] in ["deployed", "mock"]
                }
                for agent in deployed_agents
            ]
        }
    except Exception as e:
        logger.error(f"Failed to list agents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list agents: {str(e)}")

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_content(request: AnalysisRequest):
    """
    Analyze content using the deployed multi-agent system
    """
    try:
        logger.info(f"📝 Analysis request: {request.content[:50]}...")
        
        if not AGENT_CLIENT_AVAILABLE:
            raise HTTPException(status_code=503, detail="Agent client not available")
        
        # Create analysis request for coordinator agent
        analysis_request = f"""
        Please analyze the following content:
        
        Content: {request.content}
        Source Platform: {request.source_platform}
        Request Type: {request.request_type}
        
        Provide a comprehensive analysis including:
        1. Content classification and themes
        2. Risk assessment
        3. OSINT analysis if applicable
        4. Lexicon analysis for multilingual terms
        5. Trend analysis insights
        6. Actionable recommendations
        """
        
        # Call the deployed coordinator agent
        if not agent_client:
            raise HTTPException(status_code=503, detail="Agent client not initialized")
        
        agent_response = await agent_client.call_coordinator_agent(analysis_request)
        
        if agent_response.status == "failed":
            raise HTTPException(status_code=500, detail=f"Agent analysis failed: {agent_response.error}")
        
        return AnalysisResponse(
            status=agent_response.status,
            content_preview=request.content[:100],
            source_platform=request.source_platform,
            analysis=agent_response.result
        )
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {str(e)}")
        return AnalysisResponse(
            status="failed",
            content_preview=request.content[:100],
            source_platform=request.source_platform,
            error=str(e)
        )

@app.post("/submit_finding")
async def submit_finding(finding: FindingSubmission):
    """
    Submit a new finding for analysis and storage
    """
    try:
        logger.info(f"📥 New finding submitted from {finding.source_platform}")
        
        # For now, accept and log the finding
        # In the future, this would trigger analysis and storage
        
        return {
            "status": "accepted",
            "message": "Finding submitted successfully",
            "finding_id": f"finding_{hash(finding.content_text) % 100000}",
            "content_preview": finding.content_text[:100],
            "source_platform": finding.source_platform,
            "next_steps": [
                "Content will be analyzed",
                "Results stored in database", 
                "Alerts generated if needed"
            ]
        }
        
    except Exception as e:
        logger.error(f"❌ Finding submission failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/status")
async def get_service_status():
    """Get detailed service status"""
    return {
        "service": "ElectionWatch Simple Agent API",
        "version": "1.0.0",
        "uptime": "running",
        "agents": await agent_client.get_agent_status() if AGENT_CLIENT_AVAILABLE and agent_client else {"error": "not_available"},
        "capabilities": {
            "content_analysis": AGENT_CLIENT_AVAILABLE,
            "finding_submission": True,
            "health_monitoring": True,
            "agent_coordination": AGENT_CLIENT_AVAILABLE,
            "gateway_mode": True
        },
        "endpoints_active": [
            "/",
            "/health", 
            "/agents",
            "/analyze",
            "/submit_finding",
            "/status"
        ]
    }

# Run the application
if __name__ == "__main__":
    print("🚀 Starting ElectionWatch Agent Gateway API...")
    print("🌐 Gateway Mode: Calling deployed agents via Agent Engine")
    print("📡 Service will be available at: http://localhost:8080")
    print("📚 API Documentation: http://localhost:8080/docs") 
    print("🔧 Agent Client Status: Available" if AGENT_CLIENT_AVAILABLE else "🔧 Agent Client Status: Not Available")
    
    uvicorn.run(
        "simple_agent_api:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    ) 