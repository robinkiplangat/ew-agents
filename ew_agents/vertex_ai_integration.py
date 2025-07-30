#!/usr/bin/env python3
"""
Vertex AI Integration for ElectionWatch ADK Agents
=================================================

This module provides integration between the ElectionWatch ADK agent system
and Google Vertex AI services for enhanced AI capabilities and cloud deployment.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

try:
    import vertexai
    from vertexai.language_models import TextGenerationModel, ChatModel, CodeChatModel
    from vertexai.generative_models import GenerativeModel
    from google.cloud import aiplatform
    VERTEX_AI_AVAILABLE = True
except ImportError:
    VERTEX_AI_AVAILABLE = False
    logging.warning("Vertex AI SDK not available - using fallback mode")

from .coordinator_integration import coordinator_bridge

logger = logging.getLogger(__name__)

class VertexAIAgentEngine:
    """
    Vertex AI integration for the ElectionWatch agent system.
    Provides cloud-based AI capabilities and deployment support.
    """
    
    def __init__(self, project_id: str = None, location: str = "europe-west1"):
        """
        Initialize the VertexAIAgentEngine with Google Cloud project and location settings, and attempt to initialize Vertex AI SDK.
        
        Parameters:
            project_id (str, optional): Google Cloud project ID. Defaults to environment variable 'GOOGLE_CLOUD_PROJECT' or 'ew-agents-v02'.
            location (str, optional): Google Cloud region. Defaults to 'europe-west1'.
        
        If the Vertex AI SDK is available, attempts to initialize it and sets the engine's initialization status accordingly.
        """
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT", "ew-agents-v02")
        self.location = location
        self.initialized = False
        
        # Vertex AI models configuration
        self.models = {
            "text_generation": "text-bison@002",
            "chat": "chat-bison@002", 
            "code_generation": "code-bison@002",
            "gemini_pro": "gemini-1.0-pro"
        }
        
        # Initialize Vertex AI if available
        if VERTEX_AI_AVAILABLE:
            try:
                vertexai.init(project=self.project_id, location=self.location)
                self.initialized = True
                logger.info(f"Vertex AI initialized for project {self.project_id} in {self.location}")
            except Exception as e:
                logger.error(f"Failed to initialize Vertex AI: {e}")
                self.initialized = False
        
    async def enhanced_agent_processing(self, user_request: str, 
                                      agent_type: str = "comprehensive",
                                      use_vertex_ai: bool = True) -> Dict[str, Any]:
        """
                                      Processes a user analysis request using the coordinator bridge and optionally enhances the results with Vertex AI models.
                                      
                                      Parameters:
                                          user_request (str): The user's analysis request.
                                          agent_type (str, optional): The type of agent processing to perform (e.g., "comprehensive", "quick", "specialized").
                                          use_vertex_ai (bool, optional): Whether to apply Vertex AI enhancements to the analysis.
                                      
                                      Returns:
                                          Dict[str, Any]: The analysis results, including Vertex AI enhancements if enabled and available.
                                      """
        
        logger.info(f"Processing request with agent_type: {agent_type}, vertex_ai: {use_vertex_ai}")
        
        # Start with coordinator bridge processing
        base_results = await coordinator_bridge.process_request(
            user_request=user_request,
            enhanced_mode=True
        )
        
        # Enhance with Vertex AI if available and requested
        if use_vertex_ai and self.initialized:
            try:
                vertex_enhancement = await self._enhance_with_vertex_ai(
                    user_request, base_results, agent_type
                )
                base_results["vertex_ai_enhancement"] = vertex_enhancement
                base_results["processing_mode"] = "enhanced_vertex_ai"
            except Exception as e:
                logger.error(f"Vertex AI enhancement failed: {e}")
                base_results["vertex_ai_enhancement"] = {"error": str(e)}
                base_results["processing_mode"] = "standard_adk"
        else:
            base_results["processing_mode"] = "standard_adk"
            
        return base_results
    
    async def _enhance_with_vertex_ai(self, user_request: str, 
                                    base_results: Dict[str, Any],
                                    agent_type: str) -> Dict[str, Any]:
        """
                                    Enhances base agent results by applying multiple Vertex AI models for narrative analysis, risk assessment, and actionable recommendations.
                                    
                                    Parameters:
                                        user_request (str): The original user request or query.
                                        base_results (Dict[str, Any]): The initial results from the base agent processing.
                                        agent_type (str): The type of agent or analysis being performed.
                                    
                                    Returns:
                                        Dict[str, Any]: A dictionary containing the enhancement timestamp, agent type, models used, and enhanced outputs for narrative analysis, risk assessment, and recommendations. Includes error information if enhancement fails.
                                    """
        
        enhancement = {
            "timestamp": datetime.now().isoformat(),
            "agent_type": agent_type,
            "models_used": [],
            "enhancements": {}
        }
        
        try:
            # Generate enhanced narrative analysis
            if "summary" in base_results:
                narrative_enhancement = await self._enhance_narrative_analysis(
                    user_request, base_results["summary"]
                )
                enhancement["enhancements"]["narrative_analysis"] = narrative_enhancement
                enhancement["models_used"].append("text-bison")
            
            # Generate risk assessment enhancement
            if "risk_assessment" in base_results.get("summary", {}):
                risk_enhancement = await self._enhance_risk_assessment(
                    user_request, base_results["summary"]["risk_assessment"]
                )
                enhancement["enhancements"]["risk_assessment"] = risk_enhancement
                enhancement["models_used"].append("gemini-pro")
            
            # Generate actionable recommendations
            recommendations = await self._generate_enhanced_recommendations(
                user_request, base_results
            )
            enhancement["enhancements"]["recommendations"] = recommendations
            enhancement["models_used"].append("chat-bison")
            
        except Exception as e:
            logger.error(f"Vertex AI enhancement error: {e}")
            enhancement["error"] = str(e)
        
        return enhancement
    
    async def _enhance_narrative_analysis(self, user_request: str, 
                                        summary: Dict[str, Any]) -> Dict[str, Any]:
        """
                                        Enhances the narrative analysis of an election-related user request using a Vertex AI text generation model.
                                        
                                        This method generates expert-level insights tailored to African electoral contexts, focusing on hidden narrative patterns, cultural and linguistic nuances, historical parallels, and potential escalation pathways. Returns the enhanced analysis, a confidence score, and the model used. If Vertex AI is not initialized or an error occurs, returns an error message.
                                        
                                        Parameters:
                                            user_request (str): The original user request or content to analyze.
                                            summary (Dict[str, Any]): The current narrative analysis summary.
                                        
                                        Returns:
                                            Dict[str, Any]: A dictionary containing the enhanced analysis, confidence score, and model used, or an error message if enhancement fails.
                                        """
        
        if not self.initialized:
            return {"error": "Vertex AI not initialized"}
        
        try:
            model = TextGenerationModel.from_pretrained(self.models["text_generation"])
            
            prompt = f"""
            As an expert election security analyst, enhance this narrative analysis for African elections:
            
            Original Content: {user_request[:500]}
            
            Current Analysis: {json.dumps(summary.get('content_analysis', {}), indent=2)}
            
            Provide enhanced insights on:
            1. Hidden narrative patterns specific to African electoral contexts
            2. Cultural and linguistic nuances that may have been missed
            3. Historical parallels with previous election incidents
            4. Potential escalation pathways
            
            Focus on actionable intelligence for election monitoring teams.
            """
            
            response = model.predict(prompt, max_output_tokens=1024, temperature=0.3)
            
            return {
                "enhanced_analysis": response.text,
                "confidence": 0.85,
                "model_used": self.models["text_generation"]
            }
            
        except Exception as e:
            logger.error(f"Narrative enhancement failed: {e}")
            return {"error": str(e)}
    
    async def _enhance_risk_assessment(self, user_request: str,
                                     risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """
                                     Enhances the provided risk assessment using the Vertex AI Gemini Pro model.
                                     
                                     Analyzes the user request and current risk score to generate a detailed risk assessment, including immediate threats, cascade effects, geographic spread, escalation timelines, and mitigation priorities. Extracts and returns identified risk factors from the model's response.
                                     
                                     Parameters:
                                         user_request (str): The original election-related user request content.
                                         risk_assessment (Dict[str, Any]): The current risk assessment data, including overall risk score.
                                     
                                     Returns:
                                         Dict[str, Any]: A dictionary containing the enhanced risk analysis, identified risk factors, and the model used. If Vertex AI is not initialized or an error occurs, returns an error message.
                                     """
        
        if not self.initialized:
            return {"error": "Vertex AI not initialized"}
        
        try:
            model = GenerativeModel(self.models["gemini_pro"])
            
            prompt = f"""
            Analyze this election-related content for enhanced risk assessment:
            
            Content: {user_request[:500]}
            Current Risk Score: {risk_assessment.get('overall_risk_score', 'N/A')}
            
            Provide a detailed risk enhancement focusing on:
            1. Immediate threat indicators
            2. Cascade effect potential (how this could trigger broader issues)
            3. Geographic spread likelihood
            4. Timeline for potential escalation
            5. Mitigation priority ranking
            
            Format as structured analysis with specific risk factors and timelines.
            """
            
            response = model.generate_content(prompt)
            
            return {
                "enhanced_risk_analysis": response.text,
                "risk_factors_identified": self._extract_risk_factors(response.text),
                "model_used": self.models["gemini_pro"]
            }
            
        except Exception as e:
            logger.error(f"Risk enhancement failed: {e}")
            return {"error": str(e)}
    
    async def _generate_enhanced_recommendations(self, user_request: str,
                                               base_results: Dict[str, Any]) -> Dict[str, Any]:
        """
                                               Generate actionable election security recommendations using the Vertex AI chat model.
                                               
                                               Uses the chat model to provide tailored recommendations for immediate response, short-term strategy, medium-term monitoring, stakeholder communication, and documentation, with a focus on African electoral contexts.
                                               
                                               Parameters:
                                                   user_request (str): The original user request or incident description.
                                                   base_results (Dict[str, Any]): The base analysis results to inform recommendations.
                                               
                                               Returns:
                                                   Dict[str, Any]: A dictionary containing enhanced recommendations, assessed priority level, and the model used. If Vertex AI is not initialized or an error occurs, returns an error message.
                                               """
        
        if not self.initialized:
            return {"error": "Vertex AI not initialized"}
        
        try:
            model = ChatModel.from_pretrained(self.models["chat"])
            
            chat = model.start_chat()
            
            context = f"""
            Election Security Analysis Context:
            - Content analyzed: {user_request[:300]}
            - Risk level: {base_results.get('summary', {}).get('risk_assessment', {}).get('overall_risk_score', 'N/A')}
            - Agents involved: {list(base_results.get('agent_results', {}).keys())}
            """
            
            response = chat.send_message(f"""
            {context}
            
            As an election security expert, provide specific, actionable recommendations for:
            
            1. Immediate Response (next 2-4 hours)
            2. Short-term Strategy (next 24-48 hours) 
            3. Medium-term Monitoring (next week)
            4. Stakeholder Communication (who to notify and when)
            5. Documentation Requirements (what evidence to preserve)
            
            Make recommendations specific to African electoral contexts and available resources.
            """)
            
            return {
                "enhanced_recommendations": response.text,
                "priority_level": self._assess_priority_level(base_results),
                "model_used": self.models["chat"]
            }
            
        except Exception as e:
            logger.error(f"Recommendations enhancement failed: {e}")
            return {"error": str(e)}
    
    def _extract_risk_factors(self, risk_text: str) -> List[str]:
        """
        Extracts predefined risk factor keywords from the provided risk analysis text.
        
        Parameters:
            risk_text (str): The text to analyze for risk factor keywords.
        
        Returns:
            List[str]: A list of matched risk factor keywords found in the input text.
        """
        
        # Simple extraction based on common patterns
        risk_indicators = []
        
        keywords = [
            "immediate threat", "cascade effect", "escalation", 
            "violence potential", "spread likelihood", "timeline"
        ]
        
        for keyword in keywords:
            if keyword.lower() in risk_text.lower():
                risk_indicators.append(keyword)
        
        return risk_indicators
    
    def _assess_priority_level(self, results: Dict[str, Any]) -> str:
        """
        Determine the priority level based on the overall risk score in the results.
        
        Returns:
            str: One of "CRITICAL", "HIGH", "MEDIUM", or "LOW" according to the risk score thresholds.
        """
        
        risk_score = results.get("summary", {}).get("risk_assessment", {}).get("overall_risk_score", 0)
        
        if risk_score >= 8.0:
            return "CRITICAL"
        elif risk_score >= 6.0:
            return "HIGH"
        elif risk_score >= 4.0:
            return "MEDIUM"
        else:
            return "LOW"
    
    def get_deployment_config(self) -> Dict[str, Any]:
        """
        Return deployment configuration details for the Vertex AI integration, including project settings, environment variables, required permissions, and monitoring endpoints.
        """
        
        return {
            "vertex_ai_config": {
                "project_id": self.project_id,
                "location": self.location,
                "initialized": self.initialized,
                "available": VERTEX_AI_AVAILABLE,
                "models": self.models
            },
            "deployment_settings": {
                "cloud_run_compatible": True,
                "environment_variables": {
                    "GOOGLE_CLOUD_PROJECT": self.project_id,
                    "GOOGLE_CLOUD_LOCATION": self.location,
                    "VERTEX_AI_ENABLED": str(self.initialized),
                    "ADK_VERTEX_INTEGRATION": "enabled"
                },
                "required_permissions": [
                    "aiplatform.models.predict",
                    "aiplatform.endpoints.predict", 
                    "vertex-ai.user"
                ]
            },
            "monitoring": {
                "health_check_endpoint": "/vertex-ai/health",
                "metrics_to_track": [
                    "vertex_ai_requests_total",
                    "vertex_ai_latency_seconds", 
                    "vertex_ai_errors_total",
                    "agent_processing_time_seconds"
                ]
            }
        }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Return the current health status of the Vertex AI integration, including initialization state, project details, model configuration, and a basic model availability check.
        
        Returns:
            A dictionary containing health indicators such as timestamp, Vertex AI availability, initialization status, project and location info, number of configured models, overall status, and the result of a test loading the text generation model if initialized.
        """
        
        health = {
            "timestamp": datetime.now().isoformat(),
            "vertex_ai_available": VERTEX_AI_AVAILABLE,
            "initialized": self.initialized,
            "project_id": self.project_id,
            "location": self.location,
            "models_configured": len(self.models),
            "status": "healthy" if self.initialized else "degraded"
        }
        
        # Test model availability if initialized
        if self.initialized:
            try:
                # Quick test of text generation model
                model = TextGenerationModel.from_pretrained(self.models["text_generation"])
                health["model_test"] = "passed"
            except Exception as e:
                health["model_test"] = f"failed: {e}"
                health["status"] = "degraded"
        
        return health

# Global Vertex AI agent engine instance
vertex_ai_engine = VertexAIAgentEngine()

# FastAPI integration functions
async def process_with_vertex_ai(user_request: str, 
                               agent_type: str = "comprehensive") -> Dict[str, Any]:
    """
                               Processes a user request using the ElectionWatch agent system with Vertex AI enhancements enabled.
                               
                               Parameters:
                                   user_request (str): The user's election analysis request.
                                   agent_type (str): The type of agent to use for processing (default is "comprehensive").
                               
                               Returns:
                                   Dict[str, Any]: Combined results from the base agent processing and Vertex AI model enhancements.
                               """
    return await vertex_ai_engine.enhanced_agent_processing(
        user_request=user_request,
        agent_type=agent_type,
        use_vertex_ai=True
    )

def get_vertex_deployment_config() -> Dict[str, Any]:
    """
    Retrieve the current deployment configuration for the Vertex AI integration.
    
    Returns:
        Dict[str, Any]: Deployment configuration details including project, location, model information, environment variables, permissions, and monitoring metrics.
    """
    return vertex_ai_engine.get_deployment_config()

def vertex_health_check() -> Dict[str, Any]:
    """
    Return the current health status of the Vertex AI integration, including initialization state, model availability, and configuration details.
    """
    return vertex_ai_engine.health_check()

# Export main components
__all__ = [
    'VertexAIAgentEngine',
    'vertex_ai_engine', 
    'process_with_vertex_ai',
    'get_vertex_deployment_config',
    'vertex_health_check'
] 