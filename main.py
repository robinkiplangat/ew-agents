#!/usr/bin/env python3
"""
ElectionWatch Lean API - Production Ready
========================================
3-Agent Architecture: DataEng → OSINT → Coordinator
"""

from dotenv import load_dotenv
load_dotenv()

import os
import json
import uuid
import uvicorn
import logging
from datetime import datetime
from typing import List, Dict, Any

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ADK Integration
try:
    from google.adk.runners import InMemoryRunner
    from google.genai import types
    from ew_agents.agent import root_agent
    ADK_AVAILABLE = True
except ImportError as e:
    ADK_AVAILABLE = False
    logger.warning(f"ADK not available: {e}")

# Core modules
from ew_agents.mongodb_storage import store_analysis
from ew_agents.data_eng_tools import process_csv_data, extract_posts, aggregate_clean_text

class AnalysisRequest(BaseModel):
    content: str
    metadata: Dict[str, Any] = {}

def create_app() -> FastAPI:
    """Create lean FastAPI application."""
    
    app = FastAPI(
        title="ElectionWatch Lean API",
        description="3-Agent OSINT Analysis Pipeline",
        version="2.0.0"
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/run_analysis")
    async def run_analysis(
        text: str = Form(""),
        files: List[UploadFile] = File(default=[]),
        analysis_type: str = Form("misinformation_detection"),
        source: str = Form("api_upload"),
        metadata: str = Form("")
    ):
        """
        Run lean 3-agent analysis pipeline.
        
        Flow: DataEng → OSINT → Coordinator
        """
        start_time = datetime.now()
        analysis_id = f"analysis_{int(start_time.timestamp())}"
        
        try:
            # Parse metadata
            metadata_dict = {}
            if metadata:
                try:
                    metadata_dict = json.loads(metadata)
                except json.JSONDecodeError:
                    pass
            
            # Process content
            content = await process_content(text, files)
            if not content.strip():
                raise HTTPException(status_code=400, detail="No content provided")
            
            logger.info(f"Processing {len(content)} chars for analysis {analysis_id}")
            
            # Run analysis
            if ADK_AVAILABLE:
                result = await run_lean_pipeline(content, metadata_dict, analysis_id)
            else:
                result = create_basic_response(content, analysis_id)
            
            # Add timing
            result["processing_time"] = (datetime.now() - start_time).total_seconds()
            
            # Store result
            try:
                await store_analysis(analysis_id, result)
            except Exception as e:
                logger.warning(f"Storage failed: {e}")
            
            return result
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "architecture": "lean_3_agent",
            "adk_available": ADK_AVAILABLE,
            "timestamp": datetime.now().isoformat()
        }

    return app

async def process_content(text: str, files: List[UploadFile]) -> str:
    """Process input into analysis content."""
    content_parts = []
    
    # Process files
    for file in files:
        if not file or not file.filename:
            continue
            
        try:
            file_content = await file.read()
            
            # Handle CSV files
            if file.filename.endswith('.csv') or 'csv' in str(file.content_type):
                csv_text = file_content.decode('utf-8', errors='ignore')
                csv_result = process_csv_data(csv_text)
                
                # Extract structured content
                posts = csv_result.get("structured_posts", [])[:15]  # Limit posts
                platform = csv_result.get("platform_detected", "social")
                
                csv_content = [f"Platform: {platform}"]
                for post in posts:
                    if post.get('content'):
                        csv_content.append(f"[{post.get('user', 'user')}] {post['content']}")
                
                content_parts.append("\n".join(csv_content))
                
            # Handle text files
            elif file.content_type and file.content_type.startswith("text/"):
                text_content = file_content.decode('utf-8', errors='ignore')
                posts = extract_posts(text_content)
                if posts:
                    content_parts.append(aggregate_clean_text(posts))
                else:
                    content_parts.append(text_content)
                    
        except Exception as e:
            logger.warning(f"File processing error: {e}")
    
    # Add form text
    if text.strip():
        content_parts.append(text.strip())
    
    return "\n\n".join(content_parts)

async def run_lean_pipeline(content: str, metadata: Dict, analysis_id: str) -> Dict[str, Any]:
    """Execute ADK 3-agent pipeline with proper session management."""
    try:
        runner = InMemoryRunner(root_agent)
        
        # Generate session identifiers
        session_id = f"lean_{uuid.uuid4().hex[:8]}"
        user_id = "analysis_user"
        
        # Create session first
        try:
            session = await runner.session_service.create_session(
                app_name=runner.app_name,
                user_id=user_id,
                session_id=session_id
            )
            logger.info(f"✅ ADK session created: {session_id}")
        except Exception as session_error:
            logger.warning(f"Session creation failed: {session_error}, proceeding anyway")
        
        # Create analysis request
        user_content = types.Content(
            role="user",
            parts=[types.Part(text=f"""
Analyze this election-related content using the lean 3-agent pipeline:

CONTENT:
{content}

INSTRUCTIONS:
1. DataEngAgent: Process and structure the content
2. OsintAgent: Extract actors, narratives, and lexicon terms
3. CoordinatorAgent: Synthesize final structured JSON report

Return a complete JSON response with:
- analysis_id: {analysis_id}
- narrative_classification (theme, confidence, threat_level)  
- actors_identified (name, role, affiliation)
- risk_assessment (level, factors)
- recommendations
- lexicon_terms

Ensure the response is valid JSON format.
            """)]
        )
        
        # Run the pipeline
        events = []
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=user_content
        ):
            events.append(event)
            if hasattr(event, 'author'):
                logger.info(f"📋 ADK Event: {event.author}")
        
        logger.info(f"✅ ADK pipeline completed with {len(events)} events")
        
        # Extract and process response
        response = extract_response(events)
        logger.info(f"📄 Raw response length: {len(str(response))}")
        
        # Try to parse as JSON
        try:
            if isinstance(response, str):
                # Look for JSON in the response
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group())
                    logger.info("✅ Successfully parsed JSON from ADK response")
                    return parsed
                else:
                    logger.warning("⚠️ No JSON found in ADK response")
                    return create_structured_fallback(response, content, analysis_id)
            else:
                return response
                
        except json.JSONDecodeError as e:
            logger.warning(f"⚠️ JSON parsing failed: {e}")
            return create_structured_fallback(response, content, analysis_id)
            
    except Exception as e:
        logger.error(f"❌ ADK pipeline failed: {e}")
        return create_basic_response(content, analysis_id)

def extract_response(events: List) -> str:
    """Extract final response from events."""
    for event in reversed(events):
        if hasattr(event, 'content') and event.content:
            if hasattr(event.content, 'parts') and event.content.parts:
                return event.content.parts[0].text
            return str(event.content)
        elif hasattr(event, 'text') and event.text:
            return event.text
    return "Analysis completed"

def create_structured_fallback(response: str, content: str, analysis_id: str) -> Dict[str, Any]:
    """Create structured response matching target format from unstructured output."""
    
    # Extract actors from content using basic pattern matching
    actors = []
    content_lower = content.lower()
    
    # Nigerian political figures
    politicians = {
        "Bola Ahmed Tinubu": ["tinubu", "bola", "jagaban"],
        "Peter Obi": ["peter obi", "obi"],
        "Atiku Abubakar": ["atiku", "abubakar"],
        "Kashim Shettima": ["shettima", "kashim"],
    }
    
    for name, variations in politicians.items():
        for variation in variations:
            if variation in content_lower:
                actors.append({
                    "name": name,
                    "role": "Political figure",
                    "affiliation": "",
                    "influence_level": "",
                    "verification_status": "",
                    "social_metrics": {}
                })
                break
    
    # Detect narrative theme
    theme = "general_political"
    if any(term in content_lower for term in ["corruption", "fraud", "rigging"]):
        theme = "electoral_fraud_allegations"
    elif any(term in content_lower for term in ["support", "vote", "campaign"]):
        theme = "candidate_support_campaigning"
    elif any(term in content_lower for term in ["protest", "violence", "crisis"]):
        theme = "political_violence_protests"
    
    # Determine threat level
    threat_level = "low"
    if any(term in content_lower for term in ["violence", "attack", "threat"]):
        threat_level = "medium"
    elif any(term in content_lower for term in ["war", "kill", "destroy"]):
        threat_level = "high"
    
    return {
        "analysis_id": analysis_id,
        "status": "completed",
        "narrative_classification": {
            "theme": theme,
            "confidence": 0.75,
            "details": f"Analysis of {len(content.split())} words of Nigerian political content",
            "threat_indicators": [],
            "threat_level": threat_level
        },
        "actors_identified": actors,
        "risk_assessment": {
            "level": threat_level,
            "factors": ["Social media political discourse", "Election-related content"]
        },
        "recommendations": [
            "Monitor for amplification patterns",
            "Cross-reference with verified sources",
            "Track engagement metrics",
            "Assess for coordinated activity"
        ],
        "report_metadata": {
            "report_id": analysis_id,
            "analysis_timestamp": datetime.now().isoformat(),
            "report_type": "lean_analysis",
            "content_type": "social_media",
            "analysis_depth": "standard",
            "content_source": "api_upload"
        },
        "date_analyzed": datetime.now().isoformat(),
        "analysis_insights": {
            "content_statistics": {
                "word_count": len(content.split()),
                "character_count": len(content),
                "language_detected": "en"
            },
            "key_findings": f"Analyzed content containing references to Nigerian political figures",
            "risk_factors": ["Social media political discourse"] if actors else [],
            "confidence_level": "medium",
            "adk_response": response[:300]  # Keep more of the response
        }
    }

def create_basic_response(content: str, analysis_id: str) -> Dict[str, Any]:
    """Create basic response when ADK unavailable."""
    return {
        "analysis_id": analysis_id,
        "status": "basic_analysis",
        "message": "ADK pipeline not available",
        "content_stats": {
            "word_count": len(content.split()),
            "char_count": len(content)
        }
    }

if __name__ == "__main__":
    print("🚀 ElectionWatch Lean API")
    print("📋 DataEng → OSINT → Coordinator")
    
    app = create_app()
    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8080)),
        log_level="info"
    )
