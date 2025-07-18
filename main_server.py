#!/usr/bin/env python3
"""
ElectionWatch Custom API Server
Combines ADK agent functionality with custom endpoints for post analysis and reporting.
"""

import os
import json
import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field
import pandas as pd

# Import ADK components
try:
    from google.adk.agents.api_server import create_api_server
    from google.adk.agents.llm_agent import LlmAgent
    from ew_agents.election_watch_agents import coordinator_agent
    ADK_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ADK not available: {e}")
    ADK_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API requests/responses
class AnalysisRequest(BaseModel):
    text: Optional[str] = None
    source: Optional[str] = "direct_input"
    metadata: Optional[Dict[str, Any]] = {}

class AnalysisResponse(BaseModel):
    analysis_id: str
    status: str
    results: Dict[str, Any]
    timestamp: datetime
    processing_time_seconds: float

class ReportSubmission(BaseModel):
    report_id: str
    analysis_results: Dict[str, Any]
    report_type: str = "election_monitoring"
    priority: str = "medium"
    metadata: Optional[Dict[str, Any]] = {}

class ReportResponse(BaseModel):
    submission_id: str
    status: str
    report_id: str
    timestamp: datetime

# Create the main FastAPI application
app = FastAPI(
    title="ElectionWatch Analysis API",
    description="Advanced election monitoring and social media analysis API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global storage for analysis results (in production, use proper database)
analysis_storage = {}
report_storage = {}

def get_coordinator_agent():
    """Get the coordinator agent instance"""
    if not ADK_AVAILABLE:
        raise HTTPException(status_code=503, detail="ADK agent system not available")
    return coordinator_agent

async def process_with_agent(data: Dict[str, Any], agent: LlmAgent) -> Dict[str, Any]:
    """Process data using the ElectionWatch coordinator agent"""
    try:
        # Create a session for the agent
        session_data = {
            "input": data,
            "timestamp": datetime.now().isoformat(),
            "source": "api_request"
        }
        
        # For now, return mock analysis - integrate with actual agent in production
        analysis_results = {
            "sentiment_analysis": {
                "overall_sentiment": "neutral",
                "confidence": 0.75,
                "emotional_indicators": ["concern", "engagement"]
            },
            "narrative_analysis": {
                "detected_narratives": ["electoral_process", "civic_engagement"],
                "misinformation_risk": "low",
                "narrative_confidence": 0.82
            },
            "threat_assessment": {
                "risk_level": "low",
                "threat_indicators": [],
                "recommended_actions": ["continue_monitoring"]
            },
            "demographic_analysis": {
                "target_demographics": ["general_public"],
                "geographic_indicators": [],
                "language_patterns": ["english"]
            }
        }
        
        return analysis_results
        
    except Exception as e:
        logger.error(f"Agent processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis processing failed: {str(e)}")

# Root endpoint
@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with service information"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>ElectionWatch Analysis API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; }
            .endpoint { background: #ecf0f1; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .method { font-weight: bold; color: #27ae60; }
            a { color: #3498db; text-decoration: none; }
            a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🗳️ ElectionWatch Analysis API</h1>
            <p>Advanced election monitoring and social media analysis system.</p>
            
            <h2>Priority Endpoints</h2>
            <div class="endpoint">
                <span class="method">POST</span> <strong>/AnalysePosts</strong><br>
                Analyze text, CSV files, images, or other content for election-related insights.
            </div>
            <div class="endpoint">
                <span class="method">POST</span> <strong>/submitReport</strong><br>
                Submit analysis reports for processing and storage.
            </div>
            
            <h2>System Endpoints</h2>
            <div class="endpoint">
                <span class="method">GET</span> <strong>/list-apps</strong><br>
                List available ADK agent applications.
            </div>
            <div class="endpoint">
                <span class="method">GET</span> <strong>/health</strong><br>
                Service health check.
            </div>
            
            <h2>Documentation</h2>
            <p>
                • <a href="/docs">Interactive API Documentation (Swagger UI)</a><br>
                • <a href="/redoc">ReDoc Documentation</a>
            </p>
            
            <p><em>Service Status: ✅ Online</em></p>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# Priority Endpoint 1: AnalysePosts
@app.post("/AnalysePosts", response_model=AnalysisResponse)
async def analyse_posts(
    text: Optional[str] = Form(None),
    files: List[UploadFile] = File(default=[]),
    source: str = Form("api_upload"),
    metadata: str = Form("{}")
):
    """
    Analyze posts from various sources (text, CSV, images, etc.) for election-related insights.
    
    - **text**: Direct text input for analysis
    - **files**: Upload CSV, images, or other files for analysis
    - **source**: Source of the data (api_upload, social_media, etc.)
    - **metadata**: Additional metadata as JSON string
    """
    start_time = datetime.now()
    analysis_id = f"analysis_{int(start_time.timestamp())}"
    
    try:
        # Parse metadata
        try:
            metadata_dict = json.loads(metadata) if metadata else {}
        except json.JSONDecodeError:
            metadata_dict = {}
        
        # Prepare data for analysis
        analysis_data = {
            "analysis_id": analysis_id,
            "text_input": text,
            "source": source,
            "metadata": metadata_dict,
            "files_processed": []
        }
        
        # Process uploaded files
        for file in files:
            if file.filename:
                # Create temporary file to store upload
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
                    content = await file.read()
                    tmp_file.write(content)
                    tmp_file_path = tmp_file.name
                
                file_info = {
                    "filename": file.filename,
                    "content_type": file.content_type,
                    "size_bytes": len(content)
                }
                
                # Process different file types
                if file.content_type == "text/csv":
                    try:
                        df = pd.read_csv(tmp_file_path)
                        file_info["csv_info"] = {
                            "rows": len(df),
                            "columns": list(df.columns),
                            "sample_data": df.head(3).to_dict('records') if len(df) > 0 else []
                        }
                        # Add CSV text content to analysis
                        if analysis_data["text_input"]:
                            analysis_data["text_input"] += f"\n\nCSV Data from {file.filename}:\n{df.to_string()}"
                        else:
                            analysis_data["text_input"] = f"CSV Data from {file.filename}:\n{df.to_string()}"
                    except Exception as e:
                        file_info["error"] = f"CSV processing error: {str(e)}"
                
                elif file.content_type.startswith("image/"):
                    file_info["image_info"] = {
                        "format": file.content_type,
                        "note": "Image analysis requires additional ML models - processed as metadata"
                    }
                
                elif file.content_type.startswith("text/"):
                    try:
                        with open(tmp_file_path, 'r', encoding='utf-8') as f:
                            file_content = f.read()
                        if analysis_data["text_input"]:
                            analysis_data["text_input"] += f"\n\nContent from {file.filename}:\n{file_content}"
                        else:
                            analysis_data["text_input"] = f"Content from {file.filename}:\n{file_content}"
                    except Exception as e:
                        file_info["error"] = f"Text file processing error: {str(e)}"
                
                analysis_data["files_processed"].append(file_info)
                
                # Clean up temporary file
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass
        
        # Process with ElectionWatch agent
        if ADK_AVAILABLE and analysis_data.get("text_input"):
            agent = get_coordinator_agent()
            analysis_results = await process_with_agent(analysis_data, agent)
        else:
            # Mock analysis when agent not available or no text input
            analysis_results = {
                "status": "processed",
                "summary": "Analysis completed with mock data (agent not available)",
                "text_length": len(analysis_data.get("text_input", "")),
                "files_count": len(analysis_data["files_processed"]),
                "mock_analysis": True
            }
        
        # Calculate processing time
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Store results
        response_data = {
            "analysis_id": analysis_id,
            "status": "completed",
            "results": {
                "input_summary": {
                    "text_length": len(analysis_data.get("text_input", "")),
                    "files_processed": len(analysis_data["files_processed"]),
                    "source": source
                },
                "analysis": analysis_results,
                "metadata": metadata_dict
            },
            "timestamp": end_time,
            "processing_time_seconds": processing_time
        }
        
        analysis_storage[analysis_id] = response_data
        
        return AnalysisResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        error_response = {
            "analysis_id": analysis_id,
            "status": "error",
            "results": {"error": str(e)},
            "timestamp": end_time,
            "processing_time_seconds": processing_time
        }
        
        return AnalysisResponse(**error_response)

# Priority Endpoint 2: submitReport
@app.post("/submitReport", response_model=ReportResponse)
async def submit_report(report: ReportSubmission):
    """
    Submit analysis reports for processing and storage.
    
    - **report_id**: Unique identifier for the report
    - **analysis_results**: Complete analysis results to be reported
    - **report_type**: Type of report (election_monitoring, threat_assessment, etc.)
    - **priority**: Priority level (low, medium, high, critical)
    - **metadata**: Additional report metadata
    """
    try:
        submission_id = f"report_{int(datetime.now().timestamp())}"
        timestamp = datetime.now()
        
        # Validate report data
        if not report.report_id:
            raise HTTPException(status_code=400, detail="report_id is required")
        
        if not report.analysis_results:
            raise HTTPException(status_code=400, detail="analysis_results cannot be empty")
        
        # Store the report submission
        report_data = {
            "submission_id": submission_id,
            "status": "submitted",
            "report_id": report.report_id,
            "timestamp": timestamp,
            "report_content": {
                "analysis_results": report.analysis_results,
                "report_type": report.report_type,
                "priority": report.priority,
                "metadata": report.metadata or {}
            }
        }
        
        report_storage[submission_id] = report_data
        
        # Log the submission
        logger.info(f"Report submitted: {report.report_id} (submission: {submission_id})")
        
        return ReportResponse(
            submission_id=submission_id,
            status="submitted",
            report_id=report.report_id,
            timestamp=timestamp
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Report submission error: {e}")
        raise HTTPException(status_code=500, detail=f"Report submission failed: {str(e)}")

# Additional utility endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "adk_available": ADK_AVAILABLE,
        "analysis_count": len(analysis_storage),
        "report_count": len(report_storage)
    }

@app.get("/list-apps")
async def list_apps():
    """List available ADK applications"""
    if ADK_AVAILABLE:
        return ["ew_agents"]
    else:
        return {"error": "ADK not available"}

@app.get("/analysis/{analysis_id}")
async def get_analysis(analysis_id: str):
    """Retrieve analysis results by ID"""
    if analysis_id not in analysis_storage:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return analysis_storage[analysis_id]

@app.get("/report/{submission_id}")
async def get_report(submission_id: str):
    """Retrieve report submission by ID"""
    if submission_id not in report_storage:
        raise HTTPException(status_code=404, detail="Report submission not found")
    return report_storage[submission_id]

# Mount ADK API if available
if ADK_AVAILABLE:
    try:
        # Mount the ADK API server as a sub-application
        adk_app = create_api_server(agents_dir=".")
        app.mount("/adk", adk_app)
        logger.info("ADK API mounted at /adk")
    except Exception as e:
        logger.warning(f"Failed to mount ADK API: {e}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    host = os.environ.get("HOST", "0.0.0.0")
    
    logger.info(f"Starting ElectionWatch API server on {host}:{port}")
    
    uvicorn.run(
        "main_server:app",
        host=host,
        port=port,
        log_level="info",
        access_log=True
    ) 