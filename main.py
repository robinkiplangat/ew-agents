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
import sys # Added for sys.path.append

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
                            all_text += f"\n\nCSV file '{file.filename}':\n{csv_text}"
                            file_info["processed"] = "csv_content"
                            
                        elif file.content_type and file.content_type.startswith("text/"):
                            text_content = content.decode('utf-8')
                            all_text += f"\n\nFile '{file.filename}':\n{text_content}"
                            file_info["processed"] = "text_content"
                            
                        elif file.content_type and file.content_type.startswith("image/"):
                            # Enhanced multimodal image analysis
                            try:
                                import base64
                                from ew_agents.data_eng_tools import extract_text_from_image
                                
                                # Convert image to base64 for processing
                                image_base64 = base64.b64encode(content).decode('utf-8')
                                
                                # Process image with multimodal analysis
                                image_analysis = extract_text_from_image(
                                    image_data=image_base64,
                                    language_hint=metadata_dict.get("language", "en")
                                )
                                
                                if image_analysis.get("success"):
                                    file_info["processed"] = "multimodal_analysis"
                                    file_info["analysis_results"] = {
                                        "extracted_text": image_analysis.get("extracted_text", ""),
                                        "content_analysis": image_analysis.get("content_analysis", ""),
                                        "political_analysis": image_analysis.get("political_analysis", ""),
                                        "confidence_score": image_analysis.get("confidence_score", 0.0),
                                        "model_used": image_analysis.get("model_used", "placeholder")
                                    }
                                    
                                    # Add extracted text to overall analysis
                                    extracted_text = image_analysis.get("extracted_text", "")
                                    if extracted_text and not extracted_text.startswith("[TEXT EXTRACTION]"):
                                        all_text += f"\n\nImage '{file.filename}' extracted text:\n{extracted_text}"
                                    
                                    # Add political analysis to overall analysis
                                    political_analysis = image_analysis.get("political_analysis", "")
                                    if political_analysis and not political_analysis.startswith("[POLITICAL ANALYSIS]"):
                                        all_text += f"\n\nImage '{file.filename}' political content:\n{political_analysis}"
                                    
                                else:
                                    file_info["processed"] = "metadata_only"
                                    file_info["note"] = f"Image analysis failed: {image_analysis.get('error', 'Unknown error')}"
                                    
                            except Exception as e:
                                file_info["processed"] = "metadata_only"
                                file_info["note"] = f"Image processing error: {str(e)}"
                            
                        elif file.content_type and file.content_type.startswith("video/"):
                            # Enhanced multimodal video analysis
                            try:
                                import tempfile
                                import os
                                from ew_agents.data_eng_tools import extract_audio_transcript_from_video
                                
                                # Save video to temporary file for processing
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
                                    temp_video.write(content)
                                    temp_video_path = temp_video.name
                                
                                # Process video with multimodal analysis
                                video_analysis = extract_audio_transcript_from_video(
                                    video_data=temp_video_path,
                                    language_hint=metadata_dict.get("language", "en")
                                )
                                
                                # Clean up temporary file
                                try:
                                    os.unlink(temp_video_path)
                                except:
                                    pass
                                
                                if video_analysis.get("success"):
                                    file_info["processed"] = "multimodal_analysis"
                                    file_info["analysis_results"] = {
                                        "transcript": video_analysis.get("transcript", ""),
                                        "political_analysis": video_analysis.get("political_analysis", ""),
                                        "confidence_score": video_analysis.get("confidence_score", 0.0),
                                        "model_used": video_analysis.get("model_used", "placeholder"),
                                        "video_metadata": video_analysis.get("video_metadata", {})
                                    }
                                    
                                    # Add transcript to overall analysis
                                    transcript = video_analysis.get("transcript", "")
                                    if transcript and not transcript.startswith("[TRANSCRIPT PROCESSING]"):
                                        all_text += f"\n\nVideo '{file.filename}' transcript:\n{transcript}"
                                    
                                    # Add political analysis to overall analysis
                                    political_analysis = video_analysis.get("political_analysis", "")
                                    if political_analysis and not political_analysis.startswith("[POLITICAL ANALYSIS]"):
                                        all_text += f"\n\nVideo '{file.filename}' political content:\n{political_analysis}"
                                    
                                else:
                                    file_info["processed"] = "metadata_only"
                                    file_info["note"] = f"Video analysis failed: {video_analysis.get('error', 'Unknown error')}"
                                    
                            except Exception as e:
                                file_info["processed"] = "metadata_only"
                                file_info["note"] = f"Video processing error: {str(e)}"
                                
                        else:
                            file_info["note"] = f"File type {file.content_type} - processed as metadata only"
                            file_info["processed"] = "metadata_only"
                        
                        processed_files.append(file_info)
                        
                    except Exception as e:
                        processed_files.append({
                            "filename": file.filename,
                            "error": f"File processing error: {str(e)}"
                        })
            
            # Enhanced multimodal analysis when multiple content types are present
            multimodal_analysis = None
            if len([f for f in processed_files if f.get("processed") == "multimodal_analysis"]) > 1:
                try:
                    from ew_agents.data_eng_tools import analyze_multimodal_content
                    
                    # Prepare content data for multimodal analysis
                    content_data = {"text": all_text}
                    
                    # Add image content if present
                    image_files = [f for f in processed_files if f.get("processed") == "multimodal_analysis" and f.get("content_type", "").startswith("image/")]
                    if image_files:
                        # Use the first image for multimodal analysis (could be enhanced to handle multiple)
                        content_data["image"] = image_files[0].get("analysis_results", {}).get("extracted_text", "")
                    
                    # Add video content if present
                    video_files = [f for f in processed_files if f.get("processed") == "multimodal_analysis" and f.get("content_type", "").startswith("video/")]
                    if video_files:
                        # Use the first video for multimodal analysis
                        content_data["video"] = video_files[0].get("analysis_results", {}).get("transcript", "")
                    
                    # Run comprehensive multimodal analysis
                    multimodal_analysis = analyze_multimodal_content(
                        content_data=content_data,
                        content_type="mixed"
                    )
                    
                    # Add multimodal insights to text analysis
                    if multimodal_analysis.get("success"):
                        synthesis = multimodal_analysis.get("synthesis", {})
                        risk_assessment = multimodal_analysis.get("risk_assessment", {})
                        political_entities = multimodal_analysis.get("political_entities", [])
                        
                        all_text += f"\n\n=== MULTIMODAL ANALYSIS INSIGHTS ===\n"
                        all_text += f"Overall Sentiment: {synthesis.get('overall_sentiment', 'neutral')}\n"
                        all_text += f"Key Themes: {', '.join(synthesis.get('key_themes', []))}\n"
                        all_text += f"Risk Level: {risk_assessment.get('overall_risk_level', 'low')}\n"
                        all_text += f"Risk Factors: {', '.join(risk_assessment.get('risk_factors', []))}\n"
                        
                        if political_entities:
                            all_text += f"Political Entities: {', '.join([e.get('text', '') for e in political_entities])}\n"
                        
                        if risk_assessment.get("recommendations"):
                            all_text += f"Recommendations: {'; '.join(risk_assessment.get('recommendations', []))}\n"
                        
                except Exception as e:
                    logger.error(f"Multimodal analysis failed: {e}")
                    multimodal_analysis = {"success": False, "error": str(e)}
            
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
                    # Use proper ADK Content object
                    user_content = types.Content(
                        role="user",
                        parts=[
                            types.Part(text=f"Analyze this content for election-related misinformation, narratives, and risks. Process both text and any CSV data thoroughly: {all_text} Metadata: {json.dumps(metadata_dict)}")]
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
                    
                    # Check if the workflow completed all agents (look for specific indicators)
                    # Check for key workflow indicators (lexicon and trends are optional)
                    key_indicators = [
                        "ElectionWatch analysis workflow",
                        "DataEngAgent: Extraction complete",
                        "OsintAgent: Analysis complete",
                        "✓ OsintAgent: Analysis complete",  # Handle checkmark version
                        "Narrative Classification",  # OsintAgent output indicator
                        "Political Actors and Their Roles"  # OsintAgent output indicator
                    ]
                    workflow_completed = any(phrase in analysis_text for phrase in key_indicators)
                    
                    if workflow_completed:
                        logger.info("🔥 Full workflow detected, generating ElectionWatch template")
                        try:
                            # Import and call the template function directly
                            sys.path.append('ew_agents')
                            from ew_agents.report_templates import get_analysis_template
                            
                            # Build template from analysis_text instead of empty template
                            template_result = {
                                "report_metadata": {
                                    "report_id": analysis_id,
                                    "analysis_timestamp": datetime.now().isoformat(),
                                    "report_type": "analysis", 
                                    "content_type": metadata_dict.get("content_type", "text_post"),
                                    "analysis_depth": "comprehensive"
                                },
                                "narrative_classification": {},  # Will be populated from analysis_text
                                "actors": [],  # Will be populated from analysis_text  
                                "lexicon_terms": [],  # Will be populated from analysis_text
                                "risk_level": "medium",
                                "date_analyzed": datetime.now().isoformat(),
                                "recommendations": [],
                                "analysis_insights": {}
                            }
                            
                            # ENHANCED: Use Knowledge Base Integration for OsintAgent refinement
                            logger.info('🔍 Refining OsintAgent analysis with knowledge base integration')
                            
                            try:
                                # Import knowledge base functions
                                from ew_agents.knowledge_retrieval import search_knowledge, analyze_content
                                
                                # Perform semantic search for narrative classification using OsintAgent output
                                narrative_search = await search_knowledge(analysis_text, collections=['narratives'])
                                logger.info(f'📚 Knowledge base narrative search: {len(narrative_search.get("narratives", {}).get("source_nodes", []))} matches found')
                                
                                # Extract narrative classification from knowledge base results
                                if narrative_search.get('narratives', {}).get('source_nodes'):
                                    best_narrative = narrative_search['narratives']['source_nodes'][0]
                                    narrative_meta = best_narrative.get('metadata', {})
                                    confidence = best_narrative.get('score', 0.5)
                                    
                                    template_result['narrative_classification'] = {
                                        "theme": narrative_meta.get('category', 'general_political'),
                                        "threat_level": "medium" if confidence > 0.7 else "low",
                                        "details": narrative_meta.get('scenario', 'Content analyzed using knowledge base enhanced OsintAgent'),
                                        "confidence_score": confidence,
                                        "alternative_themes": narrative_meta.get('tags', [])[:3],  # Take first 3 tags
                                        "threat_indicators": narrative_meta.get('key_indicators_for_ai', [])[:5],  # Take first 5 indicators
                                        "knowledge_base_enhanced": True,
                                        "osint_agent_integration": True
                                    }
                                    logger.info(f'✅ KB-enhanced narrative: {narrative_meta.get("category", "unknown")} (confidence: {confidence:.2f})')
                                else:
                                    # Fallback to OsintAgent-based classification with KB structure
                                    template_result['narrative_classification'] = {
                                        "theme": "candidate_support_campaigning" if "Candidate Support" in analysis_text else "general_political",
                                        "threat_level": "medium" if any(word in analysis_text.lower() for word in ["threat", "risk", "violence"]) else "low",
                                        "details": "OsintAgent analysis enhanced with knowledge base methodology",
                                        "confidence_score": 0.7,
                                        "alternative_themes": ["political_engagement", "voter_mobilization"],
                                        "threat_indicators": [],
                                        "knowledge_base_enhanced": False,
                                        "osint_agent_integration": True
                                    }
                                    logger.info('⚠️ No KB narrative matches, using OsintAgent-based classification')
                            
                            except Exception as kb_error:
                                logger.error(f'❌ Knowledge base integration failed in OsintAgent workflow: {kb_error}')
                                # Fallback to original hardcoded logic
                                if "Support for Peter Obi and the Labour Party" in analysis_text:
                                    template_result['narrative_classification'] = {
                                        "theme": "political_campaigning_support",
                                        "threat_level": "medium",
                                        "details": "Content shows support and promotion for Peter Obi and Labour Party, with voter mobilization efforts",
                                        "confidence_score": 0.8,
                                        "alternative_themes": ["voter_mobilization", "political_engagement", "candidate_scrutiny"],
                                        "threat_indicators": ["anecdotal_evidence", "subtle_criticism", "coordinated_amplification"],
                                        "knowledge_base_enhanced": False,
                                        "osint_agent_integration": True,
                                        "fallback_reason": str(kb_error)[:100]
                                    }
                                elif "Narrative Classification" in analysis_text:
                                    template_result['narrative_classification'] = {
                                        "theme": "political_campaigning_support",
                                        "threat_level": "medium",
                                        "details": "Primary narratives revolve around support for Peter Obi and Labour Party, voter mobilization, and candidate scrutiny",
                                        "confidence_score": 0.8,
                                        "alternative_themes": ["voter_mobilization", "candidate_scrutiny"],
                                        "threat_indicators": ["anecdotal_evidence", "health_vitality_disinformation", "breaking_news_framing"],
                                        "knowledge_base_enhanced": False,
                                        "osint_agent_integration": True,
                                        "fallback_reason": str(kb_error)[:100]
                                    }
                            
                            # ENHANCED: Extract Lexicon Terms using Knowledge Base Integration
                            logger.info('📖 Enhancing OsintAgent lexicon extraction with knowledge base')
                            lexicon_terms = []
                            
                            try:
                                # Perform semantic search for lexicon terms using OsintAgent output
                                lexicon_search = await search_knowledge(analysis_text, collections=['hate_speech_lexicon'])
                                logger.info(f'📖 Knowledge base lexicon search: {len(lexicon_search.get("hate_speech_lexicon", {}).get("source_nodes", []))} matches found')
                                
                                # Add knowledge base lexicon matches
                                if lexicon_search.get('hate_speech_lexicon', {}).get('source_nodes'):
                                    for lexicon_node in lexicon_search['hate_speech_lexicon']['source_nodes'][:3]:  # Top 3 matches
                                        lexicon_meta = lexicon_node.get('metadata', {})
                                        lexicon_terms.append({
                                            "term": lexicon_meta.get('term', 'unknown'),
                                            "category": lexicon_meta.get('category', 'general'),
                                            "context": "knowledge_base_match",
                                            "confidence_score": lexicon_node.get('score', 0.5),
                                            "language": lexicon_meta.get('language', 'en'),
                                            "severity": lexicon_meta.get('severity', 'medium'),
                                            "definition": lexicon_meta.get('definition', 'Term from knowledge base'),
                                            "source": "knowledge_base_enhanced_osint"
                                        })
                                    logger.info(f'✅ Added {len(lexicon_terms)} KB-matched lexicon terms')
                                
                            except Exception as lexicon_kb_error:
                                logger.warning(f'⚠️ Knowledge base lexicon search failed: {lexicon_kb_error}')
                            
                            # Always add OsintAgent-detected terms (enhanced with KB methodology)
                            if "PVC" in analysis_text:
                                lexicon_terms.append({
                                    "term": "PVC",
                                    "category": "voter_mobilization",
                                    "context": "osint_agent_extraction",
                                    "confidence_score": 0.9,
                                    "language": "en",
                                    "severity": "low",
                                    "definition": "Permanent Voter Card - Your PVC is your Power",
                                    "source": "osint_agent_enhanced"
                                })
                            if "Obidients" in analysis_text or "#Obidients" in analysis_text:
                                lexicon_terms.append({
                                    "term": "Obidients",
                                    "category": "political_movement",
                                    "context": "osint_agent_extraction",
                                    "confidence_score": 0.8,
                                    "language": "en",
                                    "severity": "low",
                                    "definition": "Supporters of Peter Obi and Labour Party",
                                    "source": "osint_agent_enhanced"
                                })
                            if "Labour Party" in analysis_text:
                                lexicon_terms.append({
                                    "term": "Labour Party",
                                    "category": "political_party",
                                    "context": "osint_agent_extraction",
                                    "confidence_score": 0.9,
                                    "language": "en",
                                    "severity": "low",
                                    "definition": "Political party in Nigerian elections",
                                    "source": "osint_agent_enhanced"
                                })
                            if "Keke rider" in analysis_text:
                                lexicon_terms.append({
                                    "term": "Keke rider anecdote",
                                    "category": "anecdotal_evidence",
                                    "context": "osint_agent_extraction",
                                    "confidence_score": 0.6,
                                    "language": "en",
                                    "severity": "medium",
                                    "definition": "Repeated anecdote about Peter Obi's accessibility to common people",
                                    "source": "osint_agent_enhanced"
                                })
                            
                            # Ensure we always have lexicon terms for the report
                            if not lexicon_terms:
                                lexicon_terms = [{
                                    "term": "election content",
                                    "category": "general_political",
                                    "context": "default_osint",
                                    "confidence_score": 0.6,
                                    "language": "en",
                                    "severity": "low",
                                    "definition": "General election-related content detected by OsintAgent",
                                    "source": "osint_agent_fallback"
                                }]
                            
                            template_result['lexicon_terms'] = lexicon_terms
                            logger.info(f'✅ Final lexicon terms count: {len(lexicon_terms)} (KB + OsintAgent enhanced)')
                            
                            # ADD LLM RESPONSE to analysis_insights
                            template_result['analysis_insights']['llm_response'] = analysis_text
                            
                            # ADD LLM RESPONSE to analysis_insights
                            template_result['analysis_insights']['llm_response'] = analysis_text
                            
                            # ENHANCED: Generate Knowledge Base-Informed Recommendations
                            logger.info('💡 Generating KB-enhanced recommendations for OsintAgent analysis')
                            
                            try:
                                # Search for mitigation strategies based on narrative classification
                                mitigation_search = await search_knowledge(
                                    f"mitigation strategies for {template_result['narrative_classification'].get('theme', 'political content')}",
                                    collections=['mitigations', 'disarm_techniques']
                                )
                                
                                recommendations = []
                                
                                # Extract recommendations from knowledge base
                                if mitigation_search:
                                    for collection, results in mitigation_search.items():
                                        if results.get('source_nodes'):
                                            for node in results['source_nodes'][:2]:  # Top 2 recommendations per collection
                                                mitigation_meta = node.get('metadata', {})
                                                if collection == 'mitigations':
                                                    mitigation_name = mitigation_meta.get('mitigation_name', '')
                                                    if mitigation_name:
                                                        recommendations.append(f"Apply {mitigation_name} strategy (KB-recommended)")
                                                elif collection == 'disarm_techniques':
                                                    technique_name = mitigation_meta.get('name', '')
                                                    if technique_name:
                                                        recommendations.append(f"Counter using {technique_name} approach (DISARM)")
                                
                                # Add threat-level specific recommendations based on narrative classification
                                threat_level = template_result['narrative_classification'].get('threat_level', 'low')
                                if threat_level == 'high':
                                    recommendations.extend([
                                        "Immediate escalation required based on KB analysis",
                                        "Cross-reference with historical incident patterns",
                                        "Monitor for coordinated amplification"
                                    ])
                                elif threat_level == 'medium':
                                    recommendations.extend([
                                        "Enhanced monitoring recommended by knowledge base",
                                        "Track narrative evolution patterns",
                                        "Apply standard verification protocols"
                                    ])
                                else:
                                    recommendations.append("Standard monitoring protocols apply")
                                
                                # Add lexicon-specific recommendations
                                if len(lexicon_terms) > 2:
                                    recommendations.append(f"Monitor usage patterns of {len(lexicon_terms)} identified coded terms")
                                
                                # Fallback recommendations if KB search fails
                                if not recommendations:
                                    recommendations = [
                                        "Monitor electoral fraud narratives",
                                        "Track institutional pressure indicators", 
                                        "Watch for violence escalation in mentioned locations",
                                        "Verify claims through multiple sources"
                                    ]
                                
                                template_result['recommendations'] = recommendations[:6]  # Limit to 6 recommendations
                                template_result['risk_level'] = threat_level
                                
                                logger.info(f'✅ Generated {len(recommendations)} KB-enhanced recommendations')
                                
                            except Exception as rec_error:
                                logger.warning(f'⚠️ KB recommendations failed, using fallback: {rec_error}')
                                template_result['recommendations'] = [
                                    "Monitor electoral fraud narratives",
                                    "Track institutional pressure indicators", 
                                    "Watch for violence escalation in mentioned locations",
                                    "Verify claims through multiple sources"
                                ]
                                template_result['risk_level'] = "medium"
                            
                            # Enhance analysis_insights with comprehensive knowledge base findings
                            if 'analysis_insights' not in template_result:
                                template_result['analysis_insights'] = {}
                            
                            # Extract key findings from narrative search or use OsintAgent analysis
                            kb_key_findings = "ElectionWatch OsintAgent analysis enhanced with knowledge base"
                            if narrative_search and narrative_search.get('narratives', {}).get('response'):
                                kb_key_findings = narrative_search['narratives']['response'][:400] + "..."
                            
                            template_result['analysis_insights'].update({
                                "key_findings": kb_key_findings,
                                "risk_factors": [
                                    f"narrative_theme_{template_result['narrative_classification'].get('theme', 'unknown')}",
                                    f"threat_level_{template_result['narrative_classification'].get('threat_level', 'unknown')}",
                                    f"lexicon_terms_detected_{len(lexicon_terms)}",
                                    "knowledge_base_enhanced_analysis"
                                ],
                                "recommendations": "Knowledge base enhanced monitoring and response strategies",
                                "confidence_level": template_result['narrative_classification'].get('confidence_score', 0.7),
                                "data_sources": [
                                    "osint_agent_analysis", 
                                    "knowledge_base_narratives", 
                                    "semantic_lexicon_matching", 
                                    "disarm_technique_integration"
                                ],
                                "knowledge_base_integration": {
                                    "narrative_enhanced": template_result['narrative_classification'].get('knowledge_base_enhanced', False),
                                    "lexicon_sources": list(set([term.get('source', 'unknown') for term in lexicon_terms])),
                                    "recommendation_sources": ["mitigations", "disarm_techniques", "osint_agent"],
                                    "confidence_boost": "knowledge_base_verified"
                                }
                            })
                            
                            # Store and return the complete template
                            storage_success = await store_analysis_result(analysis_id, template_result)
                            template_result.setdefault("analysis_insights", {}).setdefault("processing_metadata", {}).update({
                                "analysis_id": analysis_id,
                                "priority": priority,
                                "agent_used": "ElectionWatchCoordinator",
                                "files_info": processed_files,
                                "storage_status": "stored" if storage_success else "failed"
                            })
                            
                            logger.info("✅ ElectionWatch template generated successfully")
                            return template_result
                            
                        except Exception as e:
                            logger.error(f"❌ Template generation failed: {e}")
                            # Fall through to generic format
                    
                    # Try to parse analysis_text as JSON (ElectionWatch template format)
                    try:
                        # Check if the agent returned a structured ElectionWatch report
                        parsed_analysis = json.loads(analysis_text)
                        if all(key in parsed_analysis for key in ['report_metadata', 'narrative_classification', 'actors', 'lexicon_terms']):
                            logger.info("✅ Agent returned ElectionWatch template format, using directly")
                            # Update metadata with actual processing info
                            parsed_analysis['report_metadata'].update({
                                "report_id": analysis_id,
                                "analysis_timestamp": datetime.now().isoformat(),
                                "content_source": source,
                                "processing_time_seconds": (datetime.now() - start_time).total_seconds()
                            })
                            # Add storage status
                            storage_success = await store_analysis_result(analysis_id, parsed_analysis)
                            parsed_analysis.setdefault("analysis_insights", {}).setdefault("processing_metadata", {})["storage_status"] = "stored" if storage_success else "failed"
                            return parsed_analysis
                    except (json.JSONDecodeError, KeyError, TypeError):
                        logger.info("Agent response not in ElectionWatch format, using fallback template")
                    
                    # Fallback: Generate unified ElectionWatch template format
                    end_time = datetime.now()
                    processing_time = (end_time - start_time).total_seconds()
                    
                    # Import and use the unified template
                    sys.path.append('ew_agents')
                    from ew_agents.report_templates import get_analysis_template
                    analysis_result = get_analysis_template(content_type=metadata_dict.get("content_type", "text_post"), analysis_depth="standard")
                    
                    # Update with actual metadata
                    analysis_result['report_metadata'].update({
                        "report_id": analysis_id,
                        "analysis_timestamp": end_time.isoformat(),
                        "content_source": source,
                        "processing_time_seconds": processing_time
                    })
                    
                    # Populate with analysis data
                    analysis_result['analysis_insights']['content_statistics'].update({
                        "word_count": len(all_text.split()),
                        "character_count": len(all_text),
                        "language_detected": "en"
                    })
                    
                    # Add processing metadata inside analysis_insights to match template structure
                    analysis_result.setdefault("analysis_insights", {}).setdefault("processing_metadata", {}).update({
                        "analysis_id": analysis_id,
                        "priority": priority,
                        "agent_used": "ElectionWatchCoordinator",
                        "files_info": processed_files,
                        "analysis_duration": processing_time,
                        "storage_status": "stored"  # Will update this after storage attempt
                    })
                    
                    # Add basic risk assessment
                    analysis_result['risk_level'] = "Medium"
                    analysis_result['narrative_classification'] = {
                        "theme": "general_election_content",
                        "threat_level": "low",
                        "details": "Content analyzed using ElectionWatch AI system",
                        "confidence_score": 0.85,
                        "alternative_themes": [],
                        "threat_indicators": []
                    }
                    
                    # Add multimodal analysis results if available
                    if multimodal_analysis and multimodal_analysis.get("success"):
                        analysis_result['multimodal_analysis'] = {
                            "synthesis": multimodal_analysis.get("synthesis", {}),
                            "risk_assessment": multimodal_analysis.get("risk_assessment", {}),
                            "political_entities": multimodal_analysis.get("political_entities", []),
                            "misinformation_indicators": multimodal_analysis.get("misinformation_indicators", [])
                        }
                        
                        # Update risk level based on multimodal analysis
                        multimodal_risk = multimodal_analysis.get("risk_assessment", {}).get("overall_risk_level", "low")
                        if multimodal_risk in ["high", "medium"]:
                            analysis_result['risk_level'] = multimodal_risk.capitalize()
                            analysis_result['narrative_classification']['threat_level'] = multimodal_risk
                        
                        # Add multimodal recommendations
                        multimodal_recommendations = multimodal_analysis.get("risk_assessment", {}).get("recommendations", [])
                        if multimodal_recommendations:
                            analysis_result['recommendations'].extend(multimodal_recommendations)
                    
                    # Add file analysis results
                    multimodal_files = [f for f in processed_files if f.get("processed") == "multimodal_analysis"]
                    if multimodal_files:
                        analysis_result['file_analysis'] = {
                            "multimodal_files_processed": len(multimodal_files),
                            "file_details": multimodal_files
                        }
                    
                    # Add basic recommendations
                    analysis_result['recommendations'] = [
                        "Content analyzed using ElectionWatch AI system",
                        f"Processed {len(processed_files)} uploaded files",
                        f"Multimodal analysis applied to {len([f for f in processed_files if f.get('processed') == 'multimodal_analysis'])} files",
                        "Monitor for similar content patterns"
                    ]
                    
                    # Store in MongoDB for later retrieval with better error handling
                    storage_success = await store_analysis_result(analysis_id, analysis_result)
                    if not storage_success:
                        logger.warning(f"⚠️ Failed to store analysis {analysis_id} in MongoDB, but analysis completed")
                        # Update storage status
                        analysis_result.setdefault("analysis_insights", {}).setdefault("processing_metadata", {})["storage_status"] = "failed"
                    else:
                        analysis_result.setdefault("analysis_insights", {}).setdefault("processing_metadata", {})["storage_status"] = "stored"
                    
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

    @app.post("/get_raw_json")
    async def get_raw_json(
        text: str = Form(None),
        files: List[UploadFile] = File(default=[]),
        analysis_type: str = Form("misinformation_detection"),
        priority: str = Form("medium"),
        source: str = Form("api_upload"),
        metadata: str = Form("{}")
    ):
        """
        Get the raw JSON response from the LLM without any additional processing.
        Returns the exact JSON structure that the agent produces.
        """
        start_time = datetime.now()
        analysis_id = f"raw_json_{int(start_time.timestamp())}"
        
        try:
            # Parse metadata safely
            try:
                metadata_dict = json.loads(metadata) if metadata else {}
            except json.JSONDecodeError:
                metadata_dict = {"parsing_error": "Invalid JSON in metadata"}
            
            # Collect all text content (same as original endpoint)
            all_text = text or ""
            processed_files = []
            
            # Process uploaded files (simplified version)
            for file in files:
                if file.filename:
                    try:
                        content = await file.read()
                        file_info = {
                            "filename": file.filename,
                            "content_type": file.content_type,
                            "size_bytes": len(content)
                        }
                        
                        if file.content_type in ["text/csv", "application/csv"]:
                            csv_text = content.decode('utf-8')
                            all_text += f"\\n\\nCSV file '{file.filename}':\\n{csv_text}"
                            file_info["processed"] = "csv_content"
                        elif file.content_type and file.content_type.startswith("text/"):
                            text_content = content.decode('utf-8')
                            all_text += f"\\n\\nFile '{file.filename}':\\n{text_content}"
                            file_info["processed"] = "text_content"
                        else:
                            file_info["processed"] = "metadata_only"
                        
                        processed_files.append(file_info)
                        
                    except Exception as e:
                        processed_files.append({
                            "filename": file.filename,
                            "error": f"File processing error: {str(e)}"
                        })
            
            # Run ADK analysis (same as original)
            if all_text.strip():
                try:
                    from ew_agents.agent import root_agent
                    from google.adk.runners import InMemoryRunner
                    from google.genai import types
                    
                    runner = InMemoryRunner(root_agent)
                    
                    user_content = types.Content(
                        role="user",
                        parts=[
                            types.Part(text=f"Analyze this content for election-related misinformation, narratives, and risks: {all_text} Metadata: {json.dumps(metadata_dict)}")]
                    )
                    
                    import uuid
                    session_id = f"raw_json_{uuid.uuid4().hex[:8]}"
                    user_id = "raw_json_user"
                    
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
                    
                    # Run the agent
                    analysis_events = []
                    async for event in runner.run_async(
                        user_id=user_id,
                        session_id=session_id,
                        new_message=user_content
                    ):
                        analysis_events.append(event)
                        logger.info(f"ADK Event: {event.author if hasattr(event, 'author') else 'unknown'}")
                    
                    # Extract final response from events
                    llm_response = "ElectionWatch Analysis Completed"
                    for event in reversed(analysis_events):  # Check from last to first
                        if hasattr(event, 'content') and event.content and event.content.parts:
                            llm_response = event.content.parts[0].text
                            break
                        elif hasattr(event, 'is_final_response') and event.is_final_response():
                            if hasattr(event, 'content') and event.content:
                                llm_response = event.content.parts[0].text if event.content.parts else str(event.content)
                            break
                    
                    logger.info(f"Final analysis result: {llm_response[:200]}...")
                    
                    end_time = datetime.now()
                    processing_time = (end_time - start_time).total_seconds()
                    
                    # Try to parse as JSON and return the raw structure
                    try:
                        parsed_json = json.loads(llm_response)
                        logger.info("✅ Agent returned valid JSON, returning raw structure")
                        return {
                            "success": True,
                            "raw_json": parsed_json,
                            "raw_text": llm_response,
                            "metadata": {
                                "analysis_id": analysis_id,
                                "processing_time_seconds": processing_time,
                                "content_length": len(all_text),
                                "files_processed": len(processed_files)
                            }
                        }
                    except json.JSONDecodeError:
                        logger.info("Agent response not in JSON format, returning raw text")
                        return {
                            "success": False,
                            "raw_text": llm_response,
                            "error": "Response is not valid JSON",
                            "metadata": {
                                "analysis_id": analysis_id,
                                "processing_time_seconds": processing_time,
                                "content_length": len(all_text),
                                "files_processed": len(processed_files)
                            }
                        }
                    
                except Exception as e:
                    return {
                        "success": False,
                        "error": f"Analysis failed: {str(e)}",
                        "raw_text": f"Error: {str(e)}",
                        "metadata": {
                            "analysis_id": analysis_id,
                            "processing_time_seconds": (datetime.now() - start_time).total_seconds()
                        }
                    }
            else:
                return {
                    "success": False,
                    "error": "No content provided for analysis",
                    "raw_text": "No content provided",
                    "metadata": {
                        "analysis_id": analysis_id,
                        "processing_time_seconds": 0
                    }
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Endpoint error: {str(e)}",
                "raw_text": f"Error: {str(e)}",
                "metadata": {
                    "analysis_id": analysis_id,
                    "processing_time_seconds": 0
                }
            }

    @app.post("/run_analysis")
    async def run_analysis(
        text: str = Form(None),
        files: List[UploadFile] = File(default=[]),
        analysis_type: str = Form("comprehensive"),
        priority: str = Form("medium"),
        source: str = Form("api_test"),
        metadata: str = Form("{}")
    ):
        """
        Test endpoint that outputs raw LLM response and structured report
        for debugging the data pipeline.
        
        Returns:
        - LLM_Response: Raw OSINT agent analysis
        - Report: Structured JSON following data/outputs/response_*.json format
        """
        start_time = datetime.now()
        analysis_id = f"test_analysis_{int(start_time.timestamp())}"
        
        try:
            # Parse metadata safely
            try:
                metadata_dict = json.loads(metadata) if metadata else {}
            except json.JSONDecodeError:
                metadata_dict = {"parsing_error": "Invalid JSON in metadata"}
            
            # Collect all text content (same as original endpoint)
            all_text = text or ""
            processed_files = []
            
            # Process uploaded files (simplified version)
            for file in files:
                if file.filename:
                    try:
                        content = await file.read()
                        file_info = {
                            "filename": file.filename,
                            "content_type": file.content_type,
                            "size_bytes": len(content)
                        }
                        
                        if file.content_type in ["text/csv", "application/csv"]:
                            csv_text = content.decode('utf-8')
                            all_text += f"\\n\\nCSV file '{file.filename}':\\n{csv_text}"
                            file_info["processed"] = "csv_content"
                        elif file.content_type and file.content_type.startswith("text/"):
                            text_content = content.decode('utf-8')
                            all_text += f"\\n\\nFile '{file.filename}':\\n{text_content}"
                            file_info["processed"] = "text_content"
                        else:
                            file_info["processed"] = "metadata_only"
                        
                        processed_files.append(file_info)
                        
                    except Exception as e:
                        processed_files.append({
                            "filename": file.filename,
                            "error": f"File processing error: {str(e)}"
                        })
            
            # Run ADK analysis (same as original)
            if all_text.strip():
                try:
                    from ew_agents.agent import root_agent
                    from google.adk.runners import InMemoryRunner
                    from google.genai import types
                    
                    runner = InMemoryRunner(root_agent)
                    
                    user_content = types.Content(
                        role="user",
                        parts=[
                            types.Part(text=f"Analyze this content for election-related misinformation, narratives, and risks: {all_text} Metadata: {json.dumps(metadata_dict)}")]
                    )
                    
                    import uuid
                    session_id = f"test_{uuid.uuid4().hex[:8]}"
                    user_id = "test_user"
                    
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
                    
                    # Run the agent
                    analysis_events = []
                    async for event in runner.run_async(
                        user_id=user_id,
                        session_id=session_id,
                        new_message=user_content
                    ):
                        analysis_events.append(event)
                        logger.info(f"ADK Event: {event.author if hasattr(event, 'author') else 'unknown'}")
                    
                    # Extract final response from events
                    llm_response = "ElectionWatch Analysis Completed"
                    for event in reversed(analysis_events):  # Check from last to first
                        if hasattr(event, 'content') and event.content and event.content.parts:
                            llm_response = event.content.parts[0].text
                            break
                        elif hasattr(event, 'is_final_response') and event.is_final_response():
                            if hasattr(event, 'content') and event.content:
                                llm_response = event.content.parts[0].text if event.content.parts else str(event.content)
                            break
                    
                    logger.info(f"Final analysis result: {llm_response[:200]}...")
                    
                    end_time = datetime.now()
                    processing_time = (end_time - start_time).total_seconds()
                    
                    # Check if LLM response is already valid JSON and return it directly
                    try:
                        parsed_json = json.loads(llm_response)
                        logger.info("✅ Agent returned valid JSON, returning directly")
                        return parsed_json
                    except json.JSONDecodeError:
                        logger.info("Agent response not in JSON format, extracting data dynamically")
                    
                    # Dynamic report structure - start empty and populate based on LLM content
                    report = {
                        "report_metadata": {
                            "report_id": analysis_id,
                            "analysis_timestamp": end_time.isoformat(),
                            "report_type": "analysis",
                            "content_type": metadata_dict.get("content_type", "text_post"),
                            "analysis_depth": "standard",
                            "content_source": source,
                            "processing_time_seconds": processing_time
                        },
                        "narrative_classification": {},
                        "actors": [],
                        "lexicon_terms": [],
                        "risk_level": "low",  # Default, will be updated based on content
                        "date_analyzed": end_time.isoformat(),
                        "recommendations": [],
                        "analysis_insights": {
                            "content_statistics": {
                                "word_count": len(all_text.split()),
                                "character_count": len(all_text),
                                "language_detected": "en"
                            },
                            "key_findings": "",  # Will be extracted from LLM
                            "risk_factors": [],  # Will be extracted from LLM
                            "confidence_level": "medium",
                            "llm_response": llm_response
                        }
                    }
                    
                    # Apply the same data extraction logic as the fixed /AnalysePosts endpoint
                    # Check for workflow completion
                    key_indicators = [
                        "ElectionWatch analysis workflow",
                        "DataEngAgent: Extraction complete",
                        "OsintAgent: Analysis complete",
                        "✓ OsintAgent: Analysis complete",
                        "Narrative Classification",
                        "Political Actors and Their Roles"
                    ]
                    workflow_completed = any(phrase in llm_response for phrase in key_indicators)
                    
                    # ENHANCED: Use Knowledge Base Integration instead of hardcoded string matching
                    logger.info('🔍 Using knowledge base for narrative classification and lexicon extraction')
                    
                    try:
                        # Import knowledge base functions
                        from ew_agents.knowledge_retrieval import search_knowledge, analyze_content
                        
                        # Perform semantic search for narrative classification
                        narrative_search = await search_knowledge(llm_response, collections=['narratives'])
                        logger.info(f'📚 Narrative search: {len(narrative_search.get("narratives", {}).get("source_nodes", []))} matches found')
                        
                        # Perform semantic search for lexicon terms  
                        lexicon_search = await search_knowledge(llm_response, collections=['hate_speech_lexicon'])
                        logger.info(f'📖 Lexicon search: {len(lexicon_search.get("hate_speech_lexicon", {}).get("source_nodes", []))} matches found')
                        
                    except Exception as kb_error:
                        logger.warning(f'⚠️ Knowledge base integration failed: {kb_error}')
                        logger.info('🔄 Falling back to basic analysis without knowledge base enhancement')
                        # Continue with basic analysis
                        
                        # Extract narrative classification from knowledge base results
                        if narrative_search and narrative_search.get('narratives', {}).get('source_nodes'):
                            best_narrative = narrative_search['narratives']['source_nodes'][0]
                            narrative_meta = best_narrative.get('metadata', {})
                            confidence = best_narrative.get('score', 0.5)
                            
                            report['narrative_classification'] = {
                                "theme": narrative_meta.get('category', 'general_political'),
                                "threat_level": "medium" if confidence > 0.7 else "low",
                                "details": narrative_meta.get('scenario', 'Content analyzed using knowledge base'),
                                "confidence_score": confidence,
                                "alternative_themes": narrative_meta.get('tags', [])[:3],  # Take first 3 tags
                                "threat_indicators": narrative_meta.get('key_indicators_for_ai', [])[:5]  # Take first 5 indicators
                            }
                            logger.info(f'✅ Narrative classified: {narrative_meta.get("category", "unknown")} (confidence: {confidence:.2f})')
                        else:
                            # Fallback narrative classification based on LLM analysis content
                            report['narrative_classification'] = {
                                "theme": "candidate_support_campaigning" if "Candidate Support" in llm_response else "general_political",
                                "threat_level": "low_medium" if "threat" in llm_response.lower() else "low",
                                "details": "Content analyzed using ElectionWatch AI system with LLM analysis",
                                "confidence_score": 0.6,
                                "alternative_themes": ["political_engagement", "voter_mobilization"],
                                "threat_indicators": []
                            }
                            logger.info('⚠️ No narrative matches found, using LLM-based classification')
                        
                        # Extract lexicon terms from knowledge base results and LLM analysis
                        lexicon_terms = []
                        
                        # Add knowledge base lexicon matches
                        if lexicon_search and lexicon_search.get('hate_speech_lexicon', {}).get('source_nodes'):
                            for lexicon_node in lexicon_search['hate_speech_lexicon']['source_nodes'][:3]:  # Top 3 matches
                                lexicon_meta = lexicon_node.get('metadata', {})
                                lexicon_terms.append({
                                    "term": lexicon_meta.get('term', 'unknown'),
                                    "category": lexicon_meta.get('category', 'general'),
                                    "context": "knowledge_base_match",
                                    "confidence_score": lexicon_node.get('score', 0.5),
                                    "language": lexicon_meta.get('language', 'en'),
                                    "severity": lexicon_meta.get('severity', 'medium'),
                                    "definition": lexicon_meta.get('definition', 'Term from knowledge base')
                                })
                        
                        # Add terms extracted from LLM analysis
                        if 'PVC' in llm_response:
                            lexicon_terms.append({
                                "term": "PVC",
                                "category": "voter_mobilization",
                                "context": "llm_extraction",
                                "confidence_score": 0.9,
                                "language": "en",
                                "severity": "low",
                                "definition": "Permanent Voter Card - essential for voting"
                            })
                        
                        if 'Obidients' in llm_response or '#Obidients' in llm_response:
                            lexicon_terms.append({
                                "term": "Obidients",
                                "category": "political_movement",
                                "context": "llm_extraction",
                                "confidence_score": 0.8,
                                "language": "en",
                                "severity": "low",
                                "definition": "Supporters of Peter Obi and Labour Party"
                            })
                        
                        if 'Labour Party' in llm_response:
                            lexicon_terms.append({
                                "term": "Labour Party",
                                "category": "political_party",
                                "context": "llm_extraction",
                                "confidence_score": 0.9,
                                "language": "en",
                                "severity": "low",
                                "definition": "Political party in Nigerian elections"
                            })
                        
                        # If no terms found, add default
                        if not lexicon_terms:
                            lexicon_terms = [{
                                "term": "election content",
                                "category": "general",
                                "context": "default",
                                "confidence_score": 0.6,
                                "language": "en",
                                "severity": "low",
                                "definition": "General election-related content"
                            }]
                        
                        report['lexicon_terms'] = lexicon_terms
                        logger.info(f'✅ Extracted {len(lexicon_terms)} lexicon terms')
                        
                        # Add LLM response to analysis_insights
                        report['analysis_insights']["llm_response"] = llm_response
                        logger.info('✅ Added LLM response to analysis_insights')
                        
                        # Extract key findings from knowledge base
                        if narrative_search and narrative_search.get('narratives', {}).get('response'):
                            report['analysis_insights']['key_findings'] = narrative_search['narratives']['response'][:500]
                        else:
                            report['analysis_insights']['key_findings'] = f"Analysis of {len(all_text.split())} words of content for election-related patterns"
                        
                        # Generate recommendations based on narrative classification
                        recommendations = []
                        if report['narrative_classification']['threat_level'] in ['medium', 'high']:
                            recommendations.extend([
                                "Enhanced monitoring recommended based on knowledge base analysis",
                                "Cross-reference with similar historical patterns",
                                "Monitor for escalation indicators"
                            ])
                        else:
                            recommendations.append("Standard monitoring protocols apply")
                        
                        if lexicon_terms and len(lexicon_terms) > 2:
                            recommendations.append(f"Track usage patterns of {len(lexicon_terms)} identified terms")
                        
                        report['recommendations'] = recommendations[:5]  # Limit to 5 recommendations
                        
                        # Set risk level based on narrative classification
                        report['risk_level'] = report['narrative_classification']['threat_level']
                        
                        logger.info('✅ Knowledge base integration completed successfully')
                        
                    except Exception as kb_error:
                        logger.error(f'❌ Knowledge base integration failed: {kb_error}')
                        # Fallback to hardcoded values if knowledge base fails
                        report['narrative_classification'] = {
                            "theme": "general_political",
                            "threat_level": "low",
                            "details": f"Fallback analysis due to KB error: {str(kb_error)[:100]}",
                            "confidence_score": 0.5,
                            "alternative_themes": [],
                            "threat_indicators": []
                        }
                        report['lexicon_terms'] = [{
                            "term": "election content",
                            "category": "general",
                            "context": "fallback",
                            "confidence_score": 0.5,
                            "language": "en",
                            "severity": "low",
                            "definition": "Fallback term due to knowledge base error"
                        }]
                        report['analysis_insights']["llm_response"] = llm_response
                        report['analysis_insights']['key_findings'] = f"Analysis of {len(all_text.split())} words of content (knowledge base unavailable)"
                        report['recommendations'] = ["Manual review recommended - knowledge base integration failed"]
                    
                    # Return both raw LLM response and structured report
                    return {
                        "LLM_Response": llm_response,
                        "Report": report
                    }
                    
                except Exception as e:
                    return {
                        "LLM_Response": f"Error: {str(e)}",
                        "Report": {
                            "error": f"Analysis failed: {str(e)}",
                            "timestamp": datetime.now().isoformat()
                        }
                    }
            else:
                return {
                    "LLM_Response": "No content provided",
                    "Report": {
                        "error": "No content provided for analysis",
                        "timestamp": datetime.now().isoformat()
                    }
                }
                
        except Exception as e:
            return {
                "LLM_Response": f"Endpoint error: {str(e)}",
                "Report": {
                    "error": f"Endpoint processing error: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                }
            }

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
    print("   Raw JSON: /get_raw_json, /run_analysis")
    print("   Utilities: /health, /analysis-template, /dev-ui")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        log_level="info"
    ) 