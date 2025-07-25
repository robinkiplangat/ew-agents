#!/usr/bin/env python3
"""
ElectionWatch Standard FastAPI + ADK Integration
==============================================================
"""

# Load environment variables first, before any other imports
from dotenv import load_dotenv
load_dotenv()

import os
import json
import asyncio
import uvicorn
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from google.adk.cli.fast_api import get_fast_api_app

# Get the directory where main.py is located
AGENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Configuration following ADK standards
SESSION_SERVICE_URI = "sqlite:///./sessions.db"
ALLOWED_ORIGINS = [
    "http://localhost", 
    "http://localhost:8080", 
    "http://localhost:3000",
    "*"  # Remove in production, add specific domains
]

# Enable web interface for testing and debugging
SERVE_WEB_INTERFACE = True

# MongoDB storage integration via ew_agents
from ew_agents.mongodb_storage import (
    storage,
    store_analysis,
    get_analysis,
    store_report,
    get_report,
    get_stats
)

# Storage function aliases for compatibility
async def store_analysis_result(analysis_id: str, analysis_data: Dict[str, Any]) -> bool:
    """Store analysis result using ew_agents storage."""
    return await store_analysis(analysis_id, analysis_data)

async def get_analysis_result(analysis_id: str) -> Optional[Dict[str, Any]]:
    """Get analysis result using ew_agents storage."""
    return await get_analysis(analysis_id)

async def store_report_submission(submission_id: str, report_data: Dict[str, Any]) -> bool:
    """Store report submission using ew_agents storage."""
    return await store_report(submission_id, report_data)

async def get_report_submission(submission_id: str) -> Optional[Dict[str, Any]]:
    """Get report submission using ew_agents storage."""
    return await get_report(submission_id)

async def get_storage_stats() -> Dict[str, Any]:
    """Get storage statistics using ew_agents storage."""
    return await get_stats()

# Pydantic models for request/response validation
class AnalysisMetadata(BaseModel):
    content_type: str = Field(..., description="Type of content being analyzed")
    region: str = Field(default="general", description="Geographic region")
    language: str = Field(default="auto", description="Content language")

class AnalysisRequest(BaseModel):
    analysis_type: str = Field(default="misinformation_detection", description="Type of analysis to perform")
    text_content: Optional[str] = Field(None, description="Text content to analyze")
    priority: str = Field(default="medium", description="Analysis priority level")
    metadata: Optional[AnalysisMetadata] = Field(default_factory=AnalysisMetadata)

class ReportMetadata(BaseModel):
    analyst: str = Field(default="system", description="Analyst identifier")
    region: str = Field(..., description="Geographic region")
    time_period: Optional[str] = Field(None, description="Analysis time period")
    sources_analyzed: int = Field(default=1, description="Number of sources analyzed")

class SupportingData(BaseModel):
    analysis_ids: List[str] = Field(default_factory=list, description="Related analysis IDs")
    confidence_scores: List[float] = Field(default_factory=list, description="Confidence scores")
    trend_indicators: List[str] = Field(default_factory=list, description="Trend indicators")

class ReportSubmission(BaseModel):
    report_id: str = Field(..., description="Unique report identifier")
    report_type: str = Field(..., description="Type of report")
    findings: str = Field(..., description="Analysis findings")
    threat_level: str = Field(..., description="Assessed threat level")
    metadata: ReportMetadata = Field(..., description="Report metadata")
    supporting_data: Optional[SupportingData] = Field(default_factory=SupportingData)

def create_app():
    """Create and configure the FastAPI application using ADK standards."""
    
    # Use ADK's built-in FastAPI app factory
    app = get_fast_api_app(
        agents_dir=AGENT_DIR,
        session_service_uri=SESSION_SERVICE_URI,
        allow_origins=ALLOWED_ORIGINS,
        web=SERVE_WEB_INTERFACE,
    )
    
    # ===== ELECTIONWATCH CUSTOM ENDPOINTS =====
    
    @app.post("/AnalysePosts")
    async def analyze_posts(
        text: Optional[str] = Form(None),
        files: List[UploadFile] = File(default=[]),
        analysis_type: str = Form("misinformation_detection"),
        priority: str = Form("medium"),
        source: str = Form("api_upload"),
        metadata: str = Form("{}")
    ):
        """
        Analyze posts from various sources for election-related insights.
        
        - **text**: Direct text input for analysis
        - **files**: Upload files (CSV, text, images) for analysis
        - **analysis_type**: Type of analysis to perform
        - **priority**: Analysis priority (low, medium, high, critical)
        - **source**: Source identifier
        - **metadata**: Additional metadata as JSON string
        """
        start_time = datetime.now()
        analysis_id = f"analysis_{int(start_time.timestamp())}"
        
        try:
            # Parse metadata safely
            try:
                metadata_dict = json.loads(metadata) if metadata else {}
            except json.JSONDecodeError:
                metadata_dict = {"parsing_error": "Invalid JSON in metadata"}
            
            # Collect all text content
            all_text = text or ""
            processed_files = []
            
            # Process uploaded files
            for file in files:
                if file.filename:
                    try:
                        content = await file.read()
                        file_info = {
                            "filename": file.filename,
                            "content_type": file.content_type,
                            "size_bytes": len(content)
                        }
                        
                        # Handle different file types
                        if file.content_type in ["text/csv", "application/csv"]:
                            csv_text = content.decode('utf-8')
                            all_text += f"\n\nCSV file '{file.filename}':\n{csv_text[:2000]}"
                            file_info["processed"] = "csv_content"
                            
                        elif file.content_type and file.content_type.startswith("text/"):
                            text_content = content.decode('utf-8')
                            all_text += f"\n\nFile '{file.filename}':\n{text_content}"
                            file_info["processed"] = "text_content"
                            
                        elif file.content_type and file.content_type.startswith("image/"):
                            file_info["note"] = "Image uploaded - OCR processing would be applied"
                            file_info["processed"] = "metadata_only"
                            
                        else:
                            file_info["note"] = f"File type {file.content_type} - processed as metadata only"
                            file_info["processed"] = "metadata_only"
                        
                        processed_files.append(file_info)
                        
                    except Exception as e:
                        processed_files.append({
                            "filename": file.filename,
                            "error": f"File processing error: {str(e)}"
                        })
            
            # Use ADK standard endpoint to run analysis
            if all_text.strip():
                # Use proper ADK Runner APIs
                try:
                    from ew_agents.agent import root_agent
                    from google.adk.runners import InMemoryRunner
                    from google.genai import types
                    
                    # Create runner properly (Python ADK takes only agent parameter)
                    runner = InMemoryRunner(root_agent)
                    
                    # Create proper ADK Content object
                    user_content = types.Content(
                        role="user",
                        parts=[types.Part(text=f"Analyze this content for election-related misinformation, narratives, and risks:\n\n{all_text}\n\nMetadata: {json.dumps(metadata_dict)}")]
                    )
                    
                    # Generate unique session ID and create session
                    import uuid
                    session_id = f"analysis_{uuid.uuid4().hex[:8]}"
                    user_id = "api_user"
                    
                    # Create session first if needed (ADK should auto-create, but let's be explicit)
                    try:
                        session = await runner.session_service.create_session(
                            app_name=runner.app_name,
                            user_id=user_id,
                            session_id=session_id
                        )
                        logger.info(f"Created session: {session_id}")
                    except Exception as e:
                        logger.info(f"Session creation note: {e}, proceeding with run_async")
                    
                    # Run the agent properly with user_id, session_id, and new_message
                    analysis_events = []
                    async for event in runner.run_async(
                        user_id=user_id,
                        session_id=session_id,
                        new_message=user_content
                    ):
                        analysis_events.append(event)
                        logger.info(f"ADK Event: {event.author if hasattr(event, 'author') else 'unknown'}")
                    
                    # Extract final response from events
                    analysis_text = "ElectionWatch Analysis Completed"
                    for event in reversed(analysis_events):  # Check from last to first
                        if hasattr(event, 'content') and event.content and event.content.parts:
                            analysis_text = event.content.parts[0].text
                            break
                        elif hasattr(event, 'is_final_response') and event.is_final_response():
                            if hasattr(event, 'content') and event.content:
                                analysis_text = event.content.parts[0].text if event.content.parts else str(event.content)
                            break
                    
                    logger.info(f"Final analysis result: {analysis_text[:200]}...")
                    
                    # Generate ElectionWatch response format
                    end_time = datetime.now()
                    processing_time = (end_time - start_time).total_seconds()
                    
                    analysis_result = {
                        "report_metadata": {
                            "report_id": analysis_id,
                            "analysis_timestamp": end_time.isoformat(),
                            "content_source": source,
                            "content_type": metadata_dict.get("content_type", "text_post"),
                            "processing_time_seconds": processing_time
                        },
                        "content_analysis": {
                            "summary": analysis_text,
                            "files_processed": len(processed_files),
                            "total_content_length": len(all_text)
                        },
                        "risk_assessment": {
                            "overall_risk": "Medium",  # This would come from agent analysis
                            "analysis_method": "adk_agent",
                            "confidence_score": 0.85
                        },
                        "recommendations": [
                            "Content analyzed using ElectionWatch AI system",
                            f"Processed {len(processed_files)} uploaded files",
                            "Monitor for similar content patterns"
                        ],
                        "processing_metadata": {
                            "analysis_id": analysis_id,
                            "priority": priority,
                            "agent_used": "ElectionWatchCoordinator",
                            "files_info": processed_files
                        }
                    }
                    
                    # Store in MongoDB for later retrieval with better error handling
                    storage_success = await store_analysis_result(analysis_id, analysis_result)
                    if not storage_success:
                        logger.warning(f"⚠️ Failed to store analysis {analysis_id} in MongoDB, but analysis completed")
                        # Add storage status to response
                        analysis_result["processing_metadata"]["storage_status"] = "failed"
                    else:
                        analysis_result["processing_metadata"]["storage_status"] = "stored"
                    
                    return analysis_result
                    
                except Exception as e:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Agent processing error: {str(e)}"
                    )
            else:
                raise HTTPException(
                    status_code=400,
                    detail="No content provided for analysis"
                )
                
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Analysis processing error: {str(e)}"
            )

    @app.post("/submitReport")
    async def submit_report(report: ReportSubmission):
        """
        Submit structured analysis reports for processing, storage, and priority routing.
        """
        try:
            submission_id = f"report_{int(datetime.now().timestamp())}"
            timestamp = datetime.now().isoformat()
            
            # Process the report submission
            submission_result = {
                "submission_id": submission_id,
                "status": "accepted",
                "report_id": report.report_id,
                "priority_level": "high" if report.threat_level in ["high", "critical"] else "medium",
                "estimated_processing_time": "5-10 minutes",
                "next_steps": [
                    "Report queued for expert review",
                    "Automated trend analysis initiated" if report.report_type == "trend_report" else "Threat assessment processing"
                ],
                "timestamp": timestamp
            }
            
            # Store in MongoDB
            report_data = {
                "submission": report.dict(),
                "result": submission_result,
                "submitted_at": timestamp
            }
            await store_report_submission(submission_id, report_data)
            
            return submission_result
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Report submission error: {str(e)}"
            )

    @app.get("/analysis/{analysis_id}")
    async def get_analysis_results(analysis_id: str):
        """
        Retrieve previously completed analysis results by ID.
        """
        analysis_doc = await get_analysis_result(analysis_id)
        if not analysis_doc:
            raise HTTPException(
                status_code=404,
                detail=f"Analysis {analysis_id} not found"
            )
        
        analysis = analysis_doc["data"]
        return {
            "analysis_id": analysis_id,
            "status": analysis_doc.get("status", "completed"),
            "results": analysis,
            "created_at": analysis_doc.get("created_at"),
            "completed_at": analysis_doc.get("created_at")
        }

    @app.get("/report/{submission_id}")
    async def get_report_submission_endpoint(submission_id: str):
        """
        Retrieve report submission status and details by ID.
        """
        report_doc = await get_report_submission(submission_id)
        if not report_doc:
            raise HTTPException(
                status_code=404,
                detail=f"Report submission {submission_id} not found"
            )
        
        stored_report = report_doc["data"]
        return {
            "submission_id": submission_id,
            "report_id": stored_report["submission"]["report_id"],
            "status": report_doc.get("status", "completed"),
            "processing_details": stored_report["result"],
            "submitted_at": stored_report["submitted_at"]
        }

    # ===== UTILITY ENDPOINTS =====
    
    @app.get("/health")
    async def health_check():
        """Custom health check endpoint with ElectionWatch metrics."""
        storage_stats = await get_storage_stats()
        
        return {
            "status": "healthy",
            "service": "electionwatch",
            "version": "v2_unified",
            "adk_integration": "standard",
            "timestamp": datetime.now().isoformat(),
            "analysis_count": storage_stats.get("analysis_count", 0),
            "report_count": storage_stats.get("reports_count", 0),
            "database_status": storage_stats.get("status", "unknown"),
            "uptime_seconds": 3600  # This would be calculated from startup time
        }
    
    @app.get("/analysis-template")
    async def get_analysis_template():
        """Get the unified analysis template structure."""
        try:
            from ew_agents.report_templates import ElectionWatchReportTemplate
            template = ElectionWatchReportTemplate.get_analysis_template()
            return {
                "template": template,
                "description": "Unified analysis template for Election Watch reports"
            }
        except Exception as e:
            return {"error": f"Failed to get template: {str(e)}"}
    
    @app.get("/dev-ui")
    async def dev_ui_redirect():
        """Redirect to the development UI."""
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/dev-ui/?app=ew_agents")
    
    @app.get("/analyses")
    async def list_recent_analyses(limit: int = 20):
        """List recent analysis results from MongoDB."""
        try:
            # This would use a proper MongoDB query in the final implementation
            return {
                "analyses": [],  # Placeholder - would query MongoDB
                "total_count": 0,
                "limit": limit,
                "database": "election_watch",
                "collection": "analysis_results"
            }
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to list analyses: {str(e)}"
            )
    
    @app.get("/storage/stats")
    async def get_storage_statistics():
        """
        Get MongoDB storage statistics and collection information.
        """
        try:
            stats = await get_stats()
            return stats
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to retrieve storage stats: {str(e)}"
            )

    @app.get("/storage/recent")
    async def get_recent_analyses(limit: int = 10):
        """
        Get recent analysis results from storage.
        """
        try:
            from ew_agents.mongodb_storage import storage
            recent = await storage.list_recent_analyses(limit=limit)
            return {
                "recent_analyses": recent,
                "count": len(recent),
                "limit": limit
            }
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to retrieve recent analyses: {str(e)}"
            )
    
    @app.get("/storage-info")
    async def get_storage_info():
        """Get detailed MongoDB storage information."""
        storage_stats = await get_storage_stats()
        
        # Get collection info using MCP
        try:
            return {
                "database": "election_watch",
                "collections": {
                    "analysis_results": {
                        "count": storage_stats.get("analysis_count", 0),
                        "description": "Stores analysis results from /AnalysePosts"
                    },
                    "report_submissions": {
                        "count": storage_stats.get("reports_count", 0),
                        "description": "Stores reports from /submitReport"
                    }
                },
                "status": storage_stats.get("status", "unknown"),
                "version": "v2_unified"
            }
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to get storage info: {str(e)}"
            )
    
    @app.get("/storage/test-connection")
    async def test_mongodb_connection():
        """
        Test MongoDB Atlas connection and return status.
        """
        try:
            from ew_agents.mongodb_storage import storage
            
            # Test by getting stats
            stats = await storage.get_collection_stats()
            
            if "error" in stats:
                return {
                    "status": "disconnected",
                    "error": stats["error"],
                    "message": "MongoDB Atlas connection failed",
                    "help": "Make sure MONGODB_ATLAS_URI is set correctly in .env: MONGODB_ATLAS_URI='mongodb+srv://username:password@cluster.mongodb.net/'"
                }
            else:
                return {
                    "status": "connected",
                    "database": stats.get("database"),
                    "collections": stats.get("collections", {}),
                    "message": "MongoDB Atlas connection successful"
                }
                
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "message": "Failed to test MongoDB Atlas connection",
                "help": "Check your MONGODB_ATLAS_URI configuration in .env file"
            }

    @app.get("/debug/env-check")
    async def debug_environment_check():
        """
        Debug endpoint to check environment configuration (without exposing sensitive data).
        """
        try:
            mongodb_atlas_uri = os.getenv("MONGODB_ATLAS_URI")
            mongodb_uri = os.getenv("MONGODB_URI")
            
            env_info = {
                "environment_variables": {
                    "MONGODB_ATLAS_URI": "set" if mongodb_atlas_uri else "not_set",
                    "MONGODB_URI": "set" if mongodb_uri else "not_set (legacy fallback)",
                    "GOOGLE_API_KEY": "set" if os.getenv("GOOGLE_API_KEY") else "not_set"
                },
                "mongodb_config": {}
            }
            
            # Check which URI is being used (without exposing credentials)
            if mongodb_atlas_uri:
                if "<username>" in mongodb_atlas_uri:
                    env_info["mongodb_config"]["status"] = "placeholder_values_detected"
                    env_info["mongodb_config"]["issue"] = "MONGODB_ATLAS_URI contains placeholder values like <username>"
                else:
                    env_info["mongodb_config"]["status"] = "uri_configured"
                    if mongodb_atlas_uri.startswith("mongodb+srv://"):
                        cluster_part = mongodb_atlas_uri.split("@")[1] if "@" in mongodb_atlas_uri else "unknown"
                        env_info["mongodb_config"]["cluster"] = cluster_part
                        env_info["mongodb_config"]["type"] = "atlas"
                    else:
                        env_info["mongodb_config"]["type"] = "custom"
            elif mongodb_uri:
                env_info["mongodb_config"]["status"] = "using_legacy_fallback"
                env_info["mongodb_config"]["recommendation"] = "Consider migrating to MONGODB_ATLAS_URI"
            else:
                env_info["mongodb_config"]["status"] = "no_uri_configured"
                env_info["mongodb_config"]["issue"] = "No MongoDB URI found in environment"
            
            # Check pymongo availability
            try:
                import pymongo
                env_info["pymongo"] = {
                    "available": True,
                    "version": pymongo.version
                }
            except ImportError:
                env_info["pymongo"] = {
                    "available": False,
                    "issue": "pymongo not installed"
                }
            
            return env_info
            
        except Exception as e:
            return {
                "error": "Failed to check environment",
                "details": str(e)
            }

    return app

# Create the app instance
app = create_app()

if __name__ == "__main__":
    # Cloud Run compatible port configuration
    port = int(os.environ.get("PORT", 8080))
    
    # MongoDB Atlas connection info at startup
    logger.info("🔗 MongoDB Atlas Configuration:")
    mongodb_atlas_uri = os.getenv("MONGODB_ATLAS_URI")
    mongodb_uri = os.getenv("MONGODB_URI")

    if mongodb_atlas_uri:
        # Don't log the full URI for security, just indicate it's set
        if mongodb_atlas_uri.startswith("mongodb+srv://"):
            logger.info("✅ MongoDB Atlas URI configured (MONGODB_ATLAS_URI)")
        else:
            logger.info("⚠️ Custom MongoDB URI configured (MONGODB_ATLAS_URI)")
    elif mongodb_uri:
        if mongodb_uri.startswith("mongodb+srv://"):
            logger.info("✅ MongoDB Atlas URI configured (MONGODB_URI - legacy)")
        else:
            logger.info("⚠️ Custom MongoDB URI configured (MONGODB_URI - legacy)")
    else:
        logger.warning("❌ MONGODB_ATLAS_URI not set - storage will be disabled")
        logger.info("💡 Set your MongoDB Atlas connection in .env file:")
        logger.info("💡 MONGODB_ATLAS_URI='mongodb+srv://username:password@cluster.mongodb.net/'")

    print("🚀 Starting ElectionWatch with Standard ADK FastAPI Setup")
    print(f"📍 Agents Directory: {AGENT_DIR}")
    print(f"💾 Session DB: {SESSION_SERVICE_URI}")
    print(f"🌐 Web Interface: {SERVE_WEB_INTERFACE}")
    print(f"🔗 Port: {port}")
    print("\n📋 Available Endpoints:")
    print("   Standard ADK: /run, /run_sse, /list-apps")
    print("   ElectionWatch: /AnalysePosts, /submitReport")
    print("   Utilities: /health, /analysis-template, /dev-ui")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        log_level="info"
    ) 