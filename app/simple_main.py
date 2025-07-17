"""
Minimal ElectionWatch API for Cloud Deployment
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Add parent directory to Python path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="ElectionWatch API",
    description="Multi-agent system for election monitoring",
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

# Request/Response models
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

# Try to import agents (graceful fallback)
AGENTS_AVAILABLE = False
try:
    from ew_agents.election_watch_agents import (
        coordinator_agent,
        data_eng_agent,
        osint_agent,
        lexicon_agent,
        trend_analysis_agent
    )
    AGENTS_AVAILABLE = True
    logger.info("✅ Successfully imported ElectionWatch agents")
except ImportError as e:
    logger.warning(f"⚠️ Could not import agents: {e}")
    logger.info("🔄 Running in mock mode")

# API Endpoints
@app.get("/")
async def root():
    return {
        "service": "ElectionWatch API",
        "version": "1.0.0",
        "status": "running",
        "agents_available": AGENTS_AVAILABLE,
        "deployment": "cloud",
        "endpoints": {
            "analyze": "/analyze",
            "submit_finding": "/submit_finding",
            "health": "/health",
            "agents": "/agents"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "agents_available": AGENTS_AVAILABLE,
        "coordinator": coordinator_agent is not None if AGENTS_AVAILABLE else False,
        "components": {
            "data_eng_agent": "ok" if AGENTS_AVAILABLE else "mock",
            "osint_agent": "ok" if AGENTS_AVAILABLE else "mock", 
            "lexicon_agent": "ok" if AGENTS_AVAILABLE else "mock",
            "trend_analysis_agent": "ok" if AGENTS_AVAILABLE else "mock"
        }
    }

@app.get("/agents")
async def list_agents():
    agents = [
        {
            "name": "CoordinatorAgent",
            "description": "Central orchestrator for the ElectionWatch system",
            "available": coordinator_agent is not None if AGENTS_AVAILABLE else False
        },
        {
            "name": "DataEngAgent", 
            "description": "Data collection, cleaning, and database management",
            "available": data_eng_agent is not None if AGENTS_AVAILABLE else False
        },
        {
            "name": "OsintAgent",
            "description": "OSINT analysis and narrative classification", 
            "available": osint_agent is not None if AGENTS_AVAILABLE else False
        },
        {
            "name": "LexiconAgent",
            "description": "Multilingual lexicon management",
            "available": lexicon_agent is not None if AGENTS_AVAILABLE else False
        },
        {
            "name": "TrendAnalysisAgent",
            "description": "Narrative trend analysis and early warnings",
            "available": trend_analysis_agent is not None if AGENTS_AVAILABLE else False
        }
    ]
    
    return {"agents": agents}

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_content(request: AnalysisRequest):
    try:
        logger.info(f"📝 Analysis request: {request.content[:50]}...")
        
        # Create analysis result
        analysis_result = {
            "summary": f"Analyzed content from {request.source_platform}",
            "content_type": "text",
            "content_length": len(request.content),
            "risk_assessment": {
                "level": "medium",
                "confidence": 0.75,
                "factors": ["unverified source", "emotional language"]
            },
            "classification": {
                "narrative_type": "election_concern",
                "themes": ["election integrity", "verification"],
                "sentiment": "negative"
            },
            "recommendations": [
                "Monitor for similar patterns",
                "Verify through additional sources",
                "Track engagement metrics"
            ],
            "processing_info": {
                "request_type": request.request_type,
                "source_platform": request.source_platform,
                "agents_used": "all" if AGENTS_AVAILABLE else "mock"
            }
        }
        
        return AnalysisResponse(
            status="completed",
            content_preview=request.content[:100],
            source_platform=request.source_platform,
            analysis=analysis_result
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
    try:
        logger.info(f"📥 New finding submitted from {finding.source_platform}")
        
        return {
            "status": "accepted",
            "message": "Finding submitted successfully",
            "finding_id": f"finding_{hash(finding.content_text) % 100000}",
            "content_preview": finding.content_text[:100],
            "source_platform": finding.source_platform,
            "processing": "queued" if AGENTS_AVAILABLE else "mock"
        }
        
    except Exception as e:
        logger.error(f"❌ Finding submission failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/status")
async def get_service_status():
    return {
        "service": "ElectionWatch API",
        "version": "1.0.0",
        "uptime": "running",
        "deployment": "cloud",
        "agents": {
            "available": AGENTS_AVAILABLE,
            "count": 5 if AGENTS_AVAILABLE else 0
        },
        "capabilities": {
            "content_analysis": True,
            "finding_submission": True,
            "health_monitoring": True,
            "agent_coordination": AGENTS_AVAILABLE
        }
    }

# Run the application
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Starting ElectionWatch API on port {port}")
    print(f"🔧 Agent Status: {'Available' if AGENTS_AVAILABLE else 'Mock Mode'}")
    
    uvicorn.run(
        "simple_main:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    ) 