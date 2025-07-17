"""
ElectionWatch Multi-Agent System Entry Point

This module serves dual purposes:
1. Exposes the coordinator_agent for interactive testing and development using `adk run`
2. Provides a FastAPI server for Cloud Run deployment

To run this agent interactively:
    cd ml/
    adk run main:coordinator_agent

For Cloud Run deployment:
    uvicorn main:app --host 0.0.0.0 --port 8080

For batch processing, use:
    python run_agents.py (for local file processing)
    python custom_fastapi_server.py (for FastAPI server with GCS integration)
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the coordinator agent
from ew_agents.election_watch_agents import coordinator_agent

# Export coordinator agent for adk run  
__all__ = ['coordinator_agent', 'app'] 

# Initialize FastAPI app
app = FastAPI(
    title="ElectionWatch Agent API",
    description="Multi-agent system for election monitoring and misinformation detection",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class TextAnalysisRequest(BaseModel):
    text: str
    analysis_type: str = "comprehensive"  # comprehensive, quick, trend_monitoring
    
class AnalysisResponse(BaseModel):
    status: str
    report_id: str
    timestamp: str
    analysis_result: Dict[str, Any]

class MockAgent:
    """Mock agent for testing when ADK is not available"""
    
    def run(self, prompt: str) -> str:
        return self.__call__(prompt)
    
    def __call__(self, prompt: str) -> str:
        return json.dumps({
            "status": "mock_analysis",
            "message": "This is a mock response. ADK agent not available.",
            "content_analyzed": prompt[:100] + "..." if len(prompt) > 100 else prompt,
            "timestamp": datetime.now().isoformat(),
            "mock_data": {
                "narrative_classification": {
                    "primary_theme": "Unknown - Mock Mode",
                    "confidence": 0.5
                },
                "risk_level": "low",
                "actors_identified": [],
                "lexicon_terms": []
            }
        })

# Global agent instance for API (fallback to mock if needed)
api_agent = None

def initialize_api_agent():
    """Initialize the agent for API use with proper error handling"""
    global api_agent
    try:
        api_agent = coordinator_agent
        logger.info("ElectionWatch coordinator agent initialized successfully for API")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize agent for API: {str(e)}")
        # Create a mock agent for testing if initialization fails
        api_agent = MockAgent()
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize the agent on startup"""
    success = initialize_api_agent()
    if success:
        logger.info("API started with full ADK agent support")
    else:
        logger.warning("API started in mock mode - ADK agent not available")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "ElectionWatch Agent API",
        "status": "running",
        "version": "1.0.0",
        "agent_mode": "mock" if isinstance(api_agent, MockAgent) else "adk"
    }

@app.get("/health")
async def health_check():
    """Health check for Cloud Run"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/analyze/text", response_model=AnalysisResponse)
async def analyze_text(request: TextAnalysisRequest):
    """Analyze text content using the ElectionWatch agents"""
    try:
        if not api_agent:
            raise HTTPException(status_code=500, detail="Agent not initialized")
        
        # Create analysis prompt based on request type
        if request.analysis_type == "comprehensive":
            prompt = f"""
            Perform a comprehensive analysis of the following text content for election monitoring:
            
            Text: {request.text}
            
            Please analyze for:
            1. Narrative classification and themes
            2. Actor identification and roles
            3. Lexicon analysis for misinformation terms
            4. Risk assessment and recommendations
            
            Provide a structured analysis report.
            """
        elif request.analysis_type == "quick":
            prompt = f"""
            Perform a quick analysis of this text for immediate threats or misinformation:
            
            Text: {request.text}
            
            Focus on rapid detection of harmful content.
            """
        else:  # trend_monitoring
            prompt = f"""
            Analyze this text for trending patterns and early warning indicators:
            
            Text: {request.text}
            
            Focus on trend analysis and pattern detection.
            """
        
        # Call the agent (try different methods based on agent type)
        try:
            if hasattr(api_agent, 'run'):
                result = api_agent.run(prompt)
            elif hasattr(api_agent, '__call__'):
                result = api_agent(prompt)
            elif hasattr(api_agent, 'process'):
                result = api_agent.process(prompt)
            else:
                result = {"error": "Unknown agent interface", "prompt": prompt}
        except Exception as e:
            logger.error(f"Agent call failed: {str(e)}")
            result = {"error": f"Agent execution failed: {str(e)}", "prompt": prompt}
        
        # Parse result if it's a string
        if isinstance(result, str):
            try:
                analysis_result = json.loads(result)
            except json.JSONDecodeError:
                analysis_result = {"raw_response": result}
        else:
            analysis_result = result
        
        # Generate response
        report_id = f"ew-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        return AnalysisResponse(
            status="success",
            report_id=report_id,
            timestamp=datetime.now().isoformat(),
            analysis_result=analysis_result
        )
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze/image")
async def analyze_image(
    file: UploadFile = File(...),
    analysis_type: str = "multimedia"
):
    """Analyze uploaded image content"""
    try:
        if not api_agent:
            raise HTTPException(status_code=500, detail="Agent not initialized")
        
        # Read image content
        content = await file.read()
        
        # For now, we'll create a prompt indicating image analysis
        # In a full implementation, you'd integrate with image processing tools
        prompt = f"""
        Analyze the uploaded image file: {file.filename}
        File size: {len(content)} bytes
        Content type: {file.content_type}
        
        Please analyze this image for election monitoring purposes, looking for:
        1. Text content extraction
        2. Visual narrative themes
        3. Actor identification
        4. Misinformation indicators
        
        Note: This is a placeholder for image analysis integration.
        """
        
        result = api_agent.run(prompt)
        
        # Parse result
        if isinstance(result, str):
            try:
                analysis_result = json.loads(result)
            except json.JSONDecodeError:
                analysis_result = {"raw_response": result}
        else:
            analysis_result = result
        
        report_id = f"ew-img-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        return AnalysisResponse(
            status="success",
            report_id=report_id,
            timestamp=datetime.now().isoformat(),
            analysis_result=analysis_result
        )
        
    except Exception as e:
        logger.error(f"Image analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Image analysis failed: {str(e)}")

@app.get("/agent/status")
async def agent_status():
    """Get agent status information"""
    return {
        "agent_initialized": api_agent is not None,
        "agent_type": "mock" if isinstance(api_agent, MockAgent) else "adk",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port) 