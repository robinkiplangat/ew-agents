"""
ElectionWatch Coordinator Integration Layer

This module provides integration between the enhanced coordinator system
and the existing Google ADK framework, maintaining backward compatibility
while providing enhanced features.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime

try:
    from google.adk.agents import LlmAgent
    from google.adk.tools.agent_tool import AgentTool
    ADK_AVAILABLE = True
except ImportError:
    # Fallback when Google ADK is not available
    ADK_AVAILABLE = False
    logging.warning("Google ADK not available, using enhanced coordinator only")

from .election_watch_agents import coordinator_agent
from . import data_eng_tools, osint_tools, lexicon_tools, trend_analysis_tools

logger = logging.getLogger(__name__)

class CoordinatorBridge:
    """
    Bridge between the enhanced coordinator and Google ADK framework
    Provides both enhanced features and backward compatibility
    """
    
    def __init__(self):
        self.enhanced_coordinator = enhanced_coordinator
        self.use_enhanced = True
        self.fallback_agents = self._create_fallback_agents()
    
    def _create_fallback_agents(self):
        """Create fallback agent structure when ADK is not available"""
        return {
            "DataEngAgent": {
                "tools": {
                    "social_media_collector": data_eng_tools.social_media_collector,
                    "run_nlp_pipeline": data_eng_tools.run_nlp_pipeline,
                    "extract_text_from_image": data_eng_tools.extract_text_from_image,
                    "database_updater": data_eng_tools.database_updater,
                    "manage_graph_db": data_eng_tools.manage_graph_db,
                    "query_knowledge_base": data_eng_tools.query_knowledge_base,
                }
            },
            "OsintAgent": {
                "tools": {
                    "classify_narrative": osint_tools.classify_narrative,
                    "classify_image_content_theme": osint_tools.classify_image_content_theme,
                    "track_keywords": osint_tools.track_keywords,
                    "calculate_influence_metrics": osint_tools.calculate_influence_metrics,
                    "detect_coordinated_behavior": osint_tools.detect_coordinated_behavior,
                    "generate_actor_profile": osint_tools.generate_actor_profile,
                }
            },
            "LexiconAgent": {
                "tools": {
                    "update_lexicon_term": lexicon_tools.update_lexicon_term,
                    "get_lexicon_term": lexicon_tools.get_lexicon_term,
                    "detect_coded_language": lexicon_tools.detect_coded_language,
                    "translate_term": lexicon_tools.translate_term,
                }
            },
            "TrendAnalysisAgent": {
                "tools": {
                    "analyze_narrative_trends": trend_analysis_tools.analyze_narrative_trends,
                    "generate_timeline_data": trend_analysis_tools.generate_timeline_data,
                    "generate_early_warning_alert": trend_analysis_tools.generate_early_warning_alert,
                }
            }
        }
    
    async def process_request(self, user_request: str, 
                            enhanced_mode: bool = True,
                            progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Process a request using either enhanced or traditional mode
        
        Args:
            user_request: The user's request
            enhanced_mode: Whether to use enhanced coordinator (default: True)
            progress_callback: Optional progress callback function
            
        Returns:
            Comprehensive analysis results
        """
        
        if enhanced_mode and self.use_enhanced:
            logger.info("Using enhanced coordinator mode")
            return await self.enhanced_coordinator.process_request(
                user_request=user_request,
                progress_callback=progress_callback
            )
        else:
            logger.info("Using traditional/fallback mode")
            return await self._process_traditional_mode(user_request)
    
    async def _process_traditional_mode(self, user_request: str) -> Dict[str, Any]:
        """Process request using traditional sequential approach"""
        
        logger.info(f"Processing in traditional mode: {user_request[:100]}...")
        
        results = {
            "status": "completed",
            "mode": "traditional",
            "user_request": user_request,
            "timestamp": datetime.now().isoformat(),
            "agent_results": {}
        }
        
        try:
            # Sequential processing similar to original workflow
            
            # Step 1: Data Engineering
            logger.info("Running DataEngAgent tools...")
            nlp_result = data_eng_tools.run_nlp_pipeline(user_request)
            results["agent_results"]["data_eng_nlp"] = nlp_result
            
            # Step 2: OSINT Analysis
            logger.info("Running OsintAgent tools...")
            classification_result = osint_tools.classify_narrative(user_request)
            results["agent_results"]["osint_classification"] = classification_result
            
            # Step 3: Lexicon Analysis
            logger.info("Running LexiconAgent tools...")
            coded_language_result = lexicon_tools.detect_coded_language(user_request, "en")
            results["agent_results"]["lexicon_analysis"] = coded_language_result
            
            # Step 4: Generate simple summary
            summary = self._generate_simple_summary(results["agent_results"])
            results["summary"] = summary
            
            return results
            
        except Exception as e:
            logger.error(f"Traditional processing failed: {e}")
            return {
                "status": "failed",
                "mode": "traditional",
                "error": str(e),
                "user_request": user_request,
                "timestamp": datetime.now().isoformat()
            }
    
    def _generate_simple_summary(self, agent_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a simple summary for traditional mode"""
        
        summary = {
            "classification_found": False,
            "risk_level": "LOW",
            "key_themes": [],
            "confidence_scores": []
        }
        
        # Extract classification results
        if "osint_classification" in agent_results:
            classification = agent_results["osint_classification"]
            if classification.get("status") == "success" and "classifications" in classification:
                summary["classification_found"] = True
                for cls in classification["classifications"]:
                    if "theme" in cls:
                        summary["key_themes"].append(cls["theme"])
                    if "confidence" in cls:
                        summary["confidence_scores"].append(cls["confidence"])
        
        # Assess risk based on lexicon analysis
        if "lexicon_analysis" in agent_results:
            lexicon = agent_results["lexicon_analysis"]
            risk_level = lexicon.get("risk_level", "LOW")
            if risk_level in ["HIGH", "MEDIUM"]:
                summary["risk_level"] = risk_level
        
        # Calculate average confidence
        if summary["confidence_scores"]:
            summary["average_confidence"] = sum(summary["confidence_scores"]) / len(summary["confidence_scores"])
        else:
            summary["average_confidence"] = 0.0
        
        return summary
    
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get workflow status (enhanced mode only)"""
        if self.use_enhanced:
            return self.enhanced_coordinator.get_workflow_status(workflow_id)
        else:
            return {"error": "Workflow tracking not available in traditional mode"}
    
    def list_active_workflows(self) -> List[Dict[str, Any]]:
        """List active workflows (enhanced mode only)"""
        if self.use_enhanced:
            return self.enhanced_coordinator.list_active_workflows()
        else:
            return []
    
    def health_check(self) -> Dict[str, Any]:
        """Check system health and capabilities"""
        
        health = {
            "timestamp": datetime.now().isoformat(),
            "adk_available": ADK_AVAILABLE,
            "enhanced_coordinator": self.use_enhanced,
            "agents_available": {},
            "tools_available": {}
        }
        
        # Check agent availability
        for agent_name, agent_info in self.fallback_agents.items():
            health["agents_available"][agent_name] = True
            health["tools_available"][agent_name] = list(agent_info["tools"].keys())
        
        # Check enhanced coordinator
        if self.use_enhanced:
            try:
                templates = self.enhanced_coordinator.workflow_templates
                health["workflow_templates"] = list(templates.keys())
                health["enhanced_status"] = "available"
            except Exception as e:
                health["enhanced_status"] = f"error: {e}"
                health["enhanced_coordinator"] = False
        
        return health

# Create global bridge instance
coordinator_bridge = CoordinatorBridge()

# Backward compatibility functions
async def process_user_request(user_request: str, 
                             progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Main entry point for processing user requests
    Maintains backward compatibility while providing enhanced features
    """
    return await coordinator_bridge.process_request(
        user_request=user_request,
        enhanced_mode=True,
        progress_callback=progress_callback
    )

def sync_process_user_request(user_request: str) -> Dict[str, Any]:
    """
    Synchronous wrapper for backward compatibility
    """
    try:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(process_user_request(user_request))
    except RuntimeError:
        # Create new event loop if none exists
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(process_user_request(user_request))
        finally:
            loop.close()

# Traditional agent interface (for ADK compatibility)
if ADK_AVAILABLE:
    try:
        # Create traditional agents for ADK compatibility
        data_eng_agent = LlmAgent(
            name="DataEngAgent",
            model="gemini-2.0-flash-lite-001",
            instruction="You are a data engineering specialist. Use your tools to perform data collection, cleaning, preprocessing, and database management as requested.",
            description="Agent for data collection, data cleaning, NLP preprocessing (text, image, video), and database/graph infrastructure management.",
            tools=[
                data_eng_tools.social_media_collector_tool,
                data_eng_tools.run_nlp_pipeline_tool,
                data_eng_tools.extract_text_from_image_tool,
                data_eng_tools.database_updater_tool,
                data_eng_tools.manage_graph_db_tool,
                data_eng_tools.query_knowledge_base_tool,
            ],
        )

        osint_agent = LlmAgent(
            name="OsintAgent", 
            model="gemini-2.0-flash-lite-001",
            instruction="You are an OSINT analysis specialist. Use your tools for narrative classification, keyword tracking, actor profiling, and coordinated behavior detection.",
            description="Agent for OSINT analysis, including narrative/content classification (text, image), keyword tracking, actor profiling, influence calculation, and coordinated behavior detection.",
            tools=[
                osint_tools.classify_narrative_tool,
                osint_tools.classify_image_content_theme_tool,
                osint_tools.track_keywords_tool,
                osint_tools.calculate_influence_metrics_tool,
                osint_tools.detect_coordinated_behavior_tool,
                osint_tools.generate_actor_profile_tool,
            ],
        )

        lexicon_agent = LlmAgent(
            name="LexiconAgent",
            model="gemini-2.0-flash-lite-001", 
            instruction="You are a lexicon management specialist. Use your tools to manage multilingual lexicons, detect coded language, and provide translation support.",
            description="Agent responsible for managing and updating multilingual lexicons, detecting coded language, and providing translation support for terms relevant to misinformation.",
            tools=[
                lexicon_tools.update_lexicon_term_tool,
                lexicon_tools.get_lexicon_term_tool,
                lexicon_tools.detect_coded_language_tool,
                lexicon_tools.translate_term_tool,
            ],
        )

        trend_analysis_agent = LlmAgent(
            name="TrendAnalysisAgent",
            model="gemini-2.0-flash-lite-001",
            instruction="You are a trend analysis specialist. Use your tools to analyze narrative trends, generate timeline data, and issue early warnings.",
            description="Agent focused on analyzing narrative trends, generating timeline data for visualizations, and issuing early warnings for significant shifts in misinformation activity.",
            tools=[
                trend_analysis_tools.analyze_narrative_trends_tool,
                trend_analysis_tools.generate_timeline_data_tool,
                trend_analysis_tools.generate_early_warning_alert_tool,
            ],
        )

        # Enhanced coordinator agent with Google ADK integration
        coordinator_agent = LlmAgent(
            name="CoordinatorAgent",
            model="gemini-2.0-flash",
            description="The enhanced central orchestrator for the ElectionWatch system with progress tracking and workflow management.",
            instruction=f"""
            You are the enhanced coordinator of a multi-agent system for election monitoring.
            
            ENHANCED CAPABILITIES:
            - Real-time progress tracking and intermediate feedback
            - Sophisticated workflow orchestration with dependency management
            - Comprehensive analysis reporting with risk assessment
            - Error handling and recovery mechanisms
            - Multi-template workflow support
            
            When processing requests:
            1. Use the enhanced coordinator system for complex multi-step workflows
            2. Provide real-time progress updates to users
            3. Generate comprehensive analysis reports with executive summaries
            4. Include risk assessments and actionable recommendations
            5. Handle errors gracefully with retry mechanisms
            
            Available workflow templates:
            - narrative_classification: Basic content classification
            - comprehensive_analysis: Full analysis with trends and actors
            - social_media_monitoring: Real-time monitoring with alerts
            - multimedia_analysis: Image and video content analysis
            
            Always synthesize results into a conclusive analysis report.
            """,
            sub_agents=[
                data_eng_agent,
                osint_agent,
                lexicon_agent,
                trend_analysis_agent,
            ],
        )
        
        logger.info("Google ADK agents created successfully")
        
    except Exception as e:
        logger.error(f"Failed to create ADK agents: {e}")
        coordinator_agent = None
        data_eng_agent = None
        osint_agent = None
        lexicon_agent = None
        trend_analysis_agent = None

else:
    # Fallback when ADK is not available
    coordinator_agent = None
    data_eng_agent = None
    osint_agent = None
    lexicon_agent = None
    trend_analysis_agent = None
    logger.info("Using enhanced coordinator without Google ADK integration")

# Export the main coordinator function
__all__ = [
    'coordinator_bridge',
    'process_user_request', 
    'sync_process_user_request',
    'coordinator_agent',
    'data_eng_agent',
    'osint_agent', 
    'lexicon_agent',
    'trend_analysis_agent'
] 