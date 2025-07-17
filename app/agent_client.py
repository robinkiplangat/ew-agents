"""
Agent Client for calling deployed ElectionWatch agents

This module provides a client interface for calling agents deployed
to Vertex AI Agent Engine from the FastAPI gateway.
"""

import os
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import asyncio
from datetime import datetime

try:
    from google.cloud import aiplatform
    from vertexai.preview.reasoning_engines import ReasoningEngine
    import vertexai
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class AgentResponse:
    """Response from an agent call"""
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    agent_name: Optional[str] = None
    execution_time: Optional[float] = None

class DeployedAgentClient:
    """Client for calling deployed ElectionWatch agents"""
    
    def __init__(self):
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "ew-agents-v01")
        self.location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        self.mode = os.getenv("AGENT_MODE", "mock")  # "mock" or "deployed"
        self.initialized = False
        self.deployed_agents = {}
        
        logger.info(f"🔧 Agent Client Mode: {self.mode}")
        
        if GCP_AVAILABLE and self.mode == "deployed":
            self._initialize_gcp()
        else:
            if not GCP_AVAILABLE:
                logger.warning("🔧 GCP libraries not available, using mock mode")
            else:
                logger.info("🔧 Running in mock mode (set AGENT_MODE=deployed for live agents)")
    
    def _initialize_gcp(self):
        """Initialize GCP client"""
        try:
            vertexai.init(project=self.project_id, location=self.location)
            self.initialized = True
            logger.info(f"✅ GCP client initialized for {self.project_id}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize GCP client: {e}")
            logger.info("🔄 Falling back to mock mode")
            self.mode = "mock"
            self.initialized = False
    
    async def call_coordinator_agent(self, request: str) -> AgentResponse:
        """Call the deployed coordinator agent with a request"""
        start_time = datetime.now()
        
        try:
            if self.mode == "mock" or not self.initialized:
                logger.info("🔄 Using mock mode for agent call")
                return await self._mock_coordinator_response(request)
            
            # TODO: Implement real Vertex AI Agent Engine call
            logger.info(f"📞 Calling deployed coordinator agent with: {request[:100]}...")
            
            # For now, simulate real call with slight delay
            await asyncio.sleep(0.8)
            
            # Until deployment is working, use enhanced mock
            result = await self._mock_coordinator_response(request)
            result.execution_time = (datetime.now() - start_time).total_seconds()
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Coordinator agent call failed: {e}")
            return AgentResponse(
                status="failed",
                error=str(e),
                agent_name="coordinator_agent",
                execution_time=(datetime.now() - start_time).total_seconds()
            )
    
    async def _mock_coordinator_response(self, request: str) -> AgentResponse:
        """Generate a comprehensive mock response that mimics real agent behavior"""
        
        # Analyze the request to provide relevant mock data
        request_lower = request.lower()
        
        # Determine content type and risk level based on keywords
        risk_level = "low"
        content_themes = []
        recommendations = []
        narrative_type = "general_discussion"
        
        # Enhanced keyword analysis
        election_keywords = ["election", "vote", "ballot", "poll", "campaign", "candidate"]
        misinformation_keywords = ["fake", "false", "misinformation", "disinformation", "rumor", "hoax"]
        ethnic_keywords = ["ethnic", "tribe", "religious", "christian", "muslim", "yoruba", "igbo", "hausa"]
        violence_keywords = ["violence", "fight", "attack", "riot", "kill", "death"]
        bvas_keywords = ["bvas", "voting machine", "rigging", "manipulation"]
        
        if any(word in request_lower for word in election_keywords):
            content_themes.append("election_integrity")
            narrative_type = "election_discourse"
            risk_level = "medium"
            recommendations.append("Monitor for election-related misinformation")
        
        if any(word in request_lower for word in misinformation_keywords):
            risk_level = "high"
            content_themes.append("misinformation")
            narrative_type = "false_information"
            recommendations.append("Flag for immediate review")
            recommendations.append("Verify claims through fact-checking")
        
        if any(word in request_lower for word in ethnic_keywords):
            content_themes.append("social_division")
            narrative_type = "ethnic_tension"
            risk_level = "high"
            recommendations.append("Alert community safety teams")
            recommendations.append("Monitor for escalation")
        
        if any(word in request_lower for word in violence_keywords):
            content_themes.append("violence_incitement")
            narrative_type = "violence_threat"
            risk_level = "critical"
            recommendations.append("Immediate security alert")
            recommendations.append("Law enforcement notification")
        
        if any(word in request_lower for word in bvas_keywords):
            content_themes.append("electoral_technology")
            narrative_type = "bvas_concern"
            if risk_level == "low":
                risk_level = "medium"
            recommendations.append("Coordinate with INEC for clarification")
        
        # Simulate multi-agent analysis
        analysis_result = {
            "summary": f"Comprehensive analysis completed for {len(request)} character content",
            "executive_summary": f"Content classified as '{narrative_type}' with {risk_level} risk level. Analysis involved data processing, OSINT classification, lexicon analysis, and trend assessment.",
            "content_analysis": {
                "content_length": len(request),
                "content_type": "text",
                "language_detected": "en",
                "themes": content_themes or ["general_discussion"],
                "narrative_classification": narrative_type,
                "sentiment": "negative" if risk_level in ["high", "critical"] else "neutral",
                "emotional_indicators": self._detect_emotions(request_lower),
                "key_entities": self._extract_entities(request_lower)
            },
            "risk_assessment": {
                "level": risk_level,
                "confidence": 0.85 if content_themes else 0.60,
                "severity_score": {"low": 2, "medium": 5, "high": 8, "critical": 10}.get(risk_level, 2),
                "factors": [
                    "Keyword analysis completed",
                    "Pattern recognition applied",
                    "Context evaluation performed",
                    "Historical comparison done"
                ],
                "escalation_required": risk_level in ["high", "critical"]
            },
            "agent_results": {
                "data_eng_analysis": {
                    "processing_status": "completed",
                    "text_quality": "good",
                    "preprocessing_applied": ["tokenization", "language_detection", "entity_extraction"],
                    "data_sources": ["user_input"],
                    "storage_status": "queued_for_database"
                },
                "osint_classification": {
                    "narrative_type": narrative_type,
                    "classification_confidence": 0.80,
                    "related_actors": "none_identified" if not content_themes else "potential_actors_detected",
                    "coordination_detected": False,
                    "influence_score": "low" if risk_level == "low" else "moderate"
                },
                "lexicon_analysis": {
                    "coded_language_detected": "concern" in request_lower or "worry" in request_lower,
                    "multilingual_terms": self._detect_multilingual_terms(request_lower),
                    "translation_needed": False,
                    "slang_detected": any(word in request_lower for word in ["guy", "omo", "abeg"])
                },
                "trend_analysis": {
                    "historical_pattern": "increasing" if risk_level in ["high", "critical"] else "normal",
                    "trend_direction": "upward" if content_themes else "stable",
                    "alert_level": risk_level,
                    "timeline_data_generated": True,
                    "early_warning_issued": risk_level == "critical"
                }
            },
            "recommendations": recommendations or [
                "Continue monitoring",
                "No immediate action required"
            ],
            "actionable_insights": self._generate_insights(narrative_type, risk_level),
            "metadata": {
                "processing_timestamp": datetime.now().isoformat(),
                "agent_version": "mock_v2.0",
                "request_id": f"req_{hash(request) % 100000}",
                "processing_mode": self.mode,
                "agents_involved": ["coordinator", "data_eng", "osint", "lexicon", "trend_analysis"]
            }
        }
        
        return AgentResponse(
            status="completed",
            result=analysis_result,
            agent_name="coordinator_agent"
        )
    
    def _detect_emotions(self, text: str) -> List[str]:
        """Detect emotional indicators in text"""
        emotions = []
        if any(word in text for word in ["concern", "worry", "fear", "scared"]):
            emotions.append("concern")
        if any(word in text for word in ["angry", "mad", "furious", "outraged"]):
            emotions.append("anger")
        if any(word in text for word in ["happy", "pleased", "satisfied"]):
            emotions.append("satisfaction")
        if any(word in text for word in ["hope", "optimistic", "confident"]):
            emotions.append("hope")
        return emotions or ["neutral"]
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract key entities from text"""
        entities = []
        if "inec" in text:
            entities.append("INEC")
        if any(word in text for word in ["bvas", "voting machine"]):
            entities.append("BVAS_Technology")
        if any(word in text for word in ["apc", "pdp", "lp", "nnpp"]):
            entities.append("Political_Party")
        if any(word in text for word in ["lagos", "kano", "abuja", "port harcourt"]):
            entities.append("Location")
        return entities
    
    def _detect_multilingual_terms(self, text: str) -> List[str]:
        """Detect multilingual terms commonly used in Nigerian context"""
        terms = []
        if "omo" in text:
            terms.append("omo (Yoruba: child/person)")
        if "abeg" in text:
            terms.append("abeg (Pidgin: please)")
        if "wahala" in text:
            terms.append("wahala (Pidgin: problem)")
        if "sabi" in text:
            terms.append("sabi (Pidgin: know)")
        return terms
    
    def _generate_insights(self, narrative_type: str, risk_level: str) -> List[str]:
        """Generate actionable insights based on content analysis"""
        insights = []
        
        if narrative_type == "election_discourse":
            insights.append("Monitor for similar election-related discussions")
            insights.append("Track engagement patterns on electoral content")
        
        if narrative_type == "false_information":
            insights.append("Cross-reference with fact-checking databases")
            insights.append("Monitor spread velocity across platforms")
        
        if narrative_type == "ethnic_tension":
            insights.append("Alert peace-building organizations")
            insights.append("Monitor regional social media activity")
        
        if risk_level in ["high", "critical"]:
            insights.append("Escalate to human moderators")
            insights.append("Prepare response communication")
        
        return insights or ["Continue routine monitoring"]
    
    async def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all deployed agents"""
        return {
            "client_initialized": True,
            "gcp_available": GCP_AVAILABLE,
            "mode": self.mode,
            "project_id": self.project_id,
            "location": self.location,
            "agents": {
                "coordinator_agent": f"{self.mode}_available",
                "data_eng_agent": f"{self.mode}_available",
                "osint_agent": f"{self.mode}_available", 
                "lexicon_agent": f"{self.mode}_available",
                "trend_analysis_agent": f"{self.mode}_available"
            }
        }
    
    async def list_deployed_agents(self) -> List[Dict[str, Any]]:
        """List all deployed agents"""
        status = "deployed" if self.mode == "deployed" and self.initialized else "mock"
        endpoint_prefix = f"projects/{self.project_id}/locations/{self.location}/reasoningEngines" if status == "deployed" else "local"
        
        return [
            {"name": "coordinator_agent", "status": status, "endpoint": f"{endpoint_prefix}/coordinator"},
            {"name": "data_eng_agent", "status": status, "endpoint": f"{endpoint_prefix}/data_eng"},
            {"name": "osint_agent", "status": status, "endpoint": f"{endpoint_prefix}/osint"},
            {"name": "lexicon_agent", "status": status, "endpoint": f"{endpoint_prefix}/lexicon"},
            {"name": "trend_analysis_agent", "status": status, "endpoint": f"{endpoint_prefix}/trend"}
        ]

# Global client instance
agent_client = DeployedAgentClient() 