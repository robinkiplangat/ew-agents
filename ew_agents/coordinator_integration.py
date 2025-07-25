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
import uuid

# Set up logging for this module
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from google.adk.agents import LlmAgent
    from google.adk.tools.agent_tool import AgentTool
    ADK_AVAILABLE = True
except ImportError:
    # Fallback when Google ADK is not available
    ADK_AVAILABLE = False
    logging.warning("Google ADK not available, using enhanced coordinator only")

from .agent import coordinator_agent
from . import data_eng_tools, osint_tools, lexicon_tools, trend_analysis_tools
from .report_templates import ElectionWatchReportTemplate

logger = logging.getLogger(__name__)

class EnhancedCoordinator:
    """
    Enhanced coordinator that works as a complementary layer to the main coordinator.
    Focuses on workflow orchestration, task assignment, monitoring, and result synthesis.
    """
    
    def __init__(self, base_coordinator_agent=None):
        self.base_coordinator = base_coordinator_agent or coordinator_agent
        self.active_workflows = {}
        
        # Define workflow templates for task assignment
        self.workflow_templates = {
            "narrative_classification": {
                "name": "Basic Narrative Classification",
                "description": "Basic content classification and narrative analysis",
                "required_agents": ["OsintAgent"],
                "optional_agents": ["LexiconAgent"]
            },
            "comprehensive_analysis": {
                "name": "Comprehensive Analysis",
                "description": "Full analysis with trends, actors, and risk assessment",
                "required_agents": ["DataEngAgent", "OsintAgent", "LexiconAgent", "TrendAnalysisAgent"]
            },
            "social_media_monitoring": {
                "name": "Social Media Monitoring",
                "description": "Real-time monitoring with alerts and trend tracking",
                "required_agents": ["DataEngAgent", "OsintAgent", "TrendAnalysisAgent"]
            },
            "multimedia_analysis": {
                "name": "Multimedia Analysis",
                "description": "Image and video content analysis",
                "required_agents": ["DataEngAgent", "OsintAgent"]
            }
        }
    
    async def process_request(self, user_request: str, 
                            workflow_template: str = "comprehensive_analysis",
                            progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Process a request using the enhanced coordinator as a complementary layer
        
        Args:
            user_request: The user's request
            workflow_template: Template to use for the workflow
            progress_callback: Optional progress callback function
            
        Returns:
            Comprehensive analysis results with enhanced orchestration
        """
        
        workflow_id = str(uuid.uuid4())
        logger.info(f"Starting enhanced workflow {workflow_id} with template: {workflow_template}")
        
        # Initialize workflow tracking
        workflow = {
            "id": workflow_id,
            "status": "in_progress",
            "template": workflow_template,
            "user_request": user_request,
            "start_time": datetime.now().isoformat(),
            "progress": 0.0,
            "current_step": "initializing",
            "task_assignments": {},
            "agent_results": {},
            "coordinator_result": None
        }
        self.active_workflows[workflow_id] = workflow
        
        try:
            # Step 1: Analyze request and assign tasks
            if progress_callback:
                progress_callback({"workflow_id": workflow_id, "progress": 0.1, "status": "analyzing request"})
            
            task_assignments = self._analyze_and_assign_tasks(user_request, workflow_template)
            workflow["task_assignments"] = task_assignments
            
            # Step 2: Let the main coordinator handle the core processing
            if progress_callback:
                progress_callback({"workflow_id": workflow_id, "progress": 0.3, "status": "delegating to coordinator"})
            
            coordinator_result = await self._delegate_to_coordinator(user_request, workflow_id, progress_callback)
            workflow["coordinator_result"] = coordinator_result
            
            # Step 3: Enhance and synthesize results
            if progress_callback:
                progress_callback({"workflow_id": workflow_id, "progress": 0.8, "status": "synthesizing results"})
            
            enhanced_result = self._synthesize_and_enhance_results(
                coordinator_result, task_assignments, workflow_id
            )
            
            # Step 4: Finalize workflow
            workflow["status"] = "completed"
            workflow["progress"] = 1.0
            workflow["current_step"] = "completed"
            workflow["end_time"] = datetime.now().isoformat()
            workflow["final_result"] = enhanced_result
            
            if progress_callback:
                progress_callback({"workflow_id": workflow_id, "progress": 1.0, "status": "completed"})
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Enhanced workflow {workflow_id} failed: {e}")
            workflow["status"] = "failed"
            workflow["error"] = str(e)
            workflow["end_time"] = datetime.now().isoformat()
            
            if progress_callback:
                progress_callback({"workflow_id": workflow_id, "progress": 0.0, "status": f"failed: {e}"})
            
            raise e
    
    def _analyze_and_assign_tasks(self, user_request: str, workflow_template: str) -> Dict[str, Any]:
        """Analyze the request and assign specific tasks to agents"""
        
        template = self.workflow_templates.get(workflow_template, self.workflow_templates["comprehensive_analysis"])
        
        # Extract actual content for analysis
        actual_content = self._extract_actual_content(user_request)
        
        # Analyze content characteristics
        content_analysis = {
            "length": len(actual_content),
            "has_multimedia": any(keyword in user_request.lower() for keyword in ["image", "video", "photo", "media"]),
            "has_hashtags": "#" in actual_content,
            "has_mentions": "@" in actual_content,
            "language": "en"  # Default, could be enhanced with language detection
        }
        
        # Assign tasks based on template and content analysis
        task_assignments = {
            "workflow_template": workflow_template,
            "content_analysis": content_analysis,
            "required_agents": template["required_agents"],
            "optional_agents": template.get("optional_agents", []),
            "task_priorities": {
                "DataEngAgent": "high" if content_analysis["has_multimedia"] else "medium",
                "OsintAgent": "high",  # Always required for narrative analysis
                "LexiconAgent": "high" if any(keyword in actual_content.lower() for keyword in ["ghost", "rigged", "fraud", "manipulate"]) else "medium",
                "TrendAnalysisAgent": "medium" if content_analysis["has_hashtags"] else "low"
            },
            "specific_tasks": {
                "DataEngAgent": ["extract_text", "preprocess_content"] + (["extract_multimedia_features"] if content_analysis["has_multimedia"] else []),
                "OsintAgent": ["classify_narrative", "assess_threat_level", "identify_actors"],
                "LexiconAgent": ["detect_coded_language", "analyze_terms", "assess_risk"],
                "TrendAnalysisAgent": ["analyze_trends", "assess_viral_potential"] if content_analysis["has_hashtags"] else ["basic_trend_analysis"]
            }
        }
        
        logger.info(f"Task assignments created: {len(task_assignments['required_agents'])} required agents, {len(task_assignments['optional_agents'])} optional")
        return task_assignments
    
    async def _delegate_to_coordinator(self, user_request: str, workflow_id: str, 
                                     progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Delegate the core processing to the main coordinator agent"""
        
        try:
            # Check if we have a working coordinator agent with sub-agents
            if (self.base_coordinator and 
                len(self.base_coordinator.sub_agents) > 0):
                
                logger.info("ADK coordinator agent available with sub-agents, but using optimized specialist processing")
                logger.info(f"Coordinator has {len(self.base_coordinator.sub_agents)} sub-agents: {[agent.name for agent in self.base_coordinator.sub_agents]}")
                
                # For now, use the optimized specialist agent processing
                # The ADK coordinator integration requires proper InvocationContext setup
                # which is complex for this use case
                return await self._process_with_specialist_agents(user_request, workflow_id, progress_callback)
                    
            else:
                logger.info("No ADK coordinator with sub-agents available, using fallback processing")
                return await self._process_with_fallback(user_request, workflow_id, progress_callback)
                
        except Exception as e:
            logger.error(f"Coordinator delegation failed: {e}")
            # Fallback to basic processing
            return await self._process_with_fallback(user_request, workflow_id, progress_callback)
    
    def _synthesize_and_enhance_results(self, coordinator_result: Dict[str, Any], 
                                       task_assignments: Dict[str, Any], 
                                       workflow_id: str) -> Dict[str, Any]:
        """Synthesize coordinator results with enhanced orchestration insights"""
        
        # Start with the coordinator's result
        enhanced_result = coordinator_result.copy()
        
        # Add enhanced orchestration metadata
        enhanced_result["enhanced_orchestration"] = {
            "workflow_id": workflow_id,
            "task_assignments": task_assignments,
            "orchestration_insights": self._generate_orchestration_insights(coordinator_result, task_assignments),
            "workflow_efficiency": self._assess_workflow_efficiency(coordinator_result, task_assignments),
            "recommended_optimizations": self._generate_optimization_recommendations(coordinator_result, task_assignments)
        }
        
        # Enhance the summary with orchestration context
        if "summary" in enhanced_result:
            enhanced_result["summary"]["orchestration_context"] = {
                "agents_utilized": len(task_assignments.get("required_agents", [])),
                "task_completion_rate": self._calculate_task_completion_rate(coordinator_result),
                "workflow_complexity": self._assess_workflow_complexity(task_assignments),
                "processing_efficiency": "high" if enhanced_result.get("status") == "completed" else "medium"
            }
        
        return enhanced_result
    
    def _generate_orchestration_insights(self, coordinator_result: Dict[str, Any], 
                                       task_assignments: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights about the orchestration process"""
        
        insights = {
            "workflow_type": task_assignments.get("workflow_template", "unknown"),
            "content_characteristics": task_assignments.get("content_analysis", {}),
            "agent_utilization": {
                "required_agents": len(task_assignments.get("required_agents", [])),
                "optional_agents": len(task_assignments.get("optional_agents", [])),
                "total_agents": len(task_assignments.get("required_agents", [])) + len(task_assignments.get("optional_agents", []))
            },
            "task_distribution": task_assignments.get("specific_tasks", {}),
            "processing_mode": coordinator_result.get("mode", "unknown")
        }
        
        return insights
    
    def _assess_workflow_efficiency(self, coordinator_result: Dict[str, Any], 
                                  task_assignments: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the efficiency of the workflow execution"""
        
        # Calculate efficiency metrics
        required_agents = len(task_assignments.get("required_agents", []))
        status = coordinator_result.get("status", "unknown")
        
        efficiency = {
            "agent_utilization_score": min(required_agents / 4.0, 1.0),  # Normalize to 4 agents max
            "completion_success": 1.0 if status == "completed" else 0.0,
            "workflow_complexity": "high" if required_agents >= 3 else "medium" if required_agents >= 2 else "low",
            "resource_efficiency": "optimal" if required_agents <= 2 else "standard"
        }
        
        return efficiency
    
    def _generate_optimization_recommendations(self, coordinator_result: Dict[str, Any], 
                                             task_assignments: Dict[str, Any]) -> List[str]:
        """Generate recommendations for workflow optimization"""
        
        recommendations = []
        
        # Check agent utilization
        required_agents = len(task_assignments.get("required_agents", []))
        if required_agents > 3:
            recommendations.append("Consider using simplified workflow for faster processing")
        
        # Check content characteristics
        content_analysis = task_assignments.get("content_analysis", {})
        if content_analysis.get("length", 0) < 100:
            recommendations.append("Content is short - consider quick analysis template")
        
        # Check for multimedia
        if content_analysis.get("has_multimedia", False):
            recommendations.append("Multimedia content detected - ensure proper media processing")
        
        # Check completion status
        if coordinator_result.get("status") != "completed":
            recommendations.append("Workflow did not complete successfully - review error handling")
        
        return recommendations
    
    def _calculate_task_completion_rate(self, coordinator_result: Dict[str, Any]) -> float:
        """Calculate the task completion rate based on coordinator results"""
        
        agent_results = coordinator_result.get("agent_results", {})
        if not agent_results:
            return 0.0
        
        completed_tasks = 0
        total_tasks = 0
        
        for agent_name, result in agent_results.items():
            if result.get("status") == "success":
                completed_tasks += 1
            total_tasks += 1
        
        return completed_tasks / total_tasks if total_tasks > 0 else 0.0
    
    def _assess_workflow_complexity(self, task_assignments: Dict[str, Any]) -> str:
        """Assess the complexity of the workflow"""
        
        required_agents = len(task_assignments.get("required_agents", []))
        specific_tasks = task_assignments.get("specific_tasks", {})
        
        total_tasks = sum(len(tasks) for tasks in specific_tasks.values())
        
        if required_agents >= 4 or total_tasks >= 8:
            return "high"
        elif required_agents >= 2 or total_tasks >= 4:
            return "medium"
        else:
            return "low"
    
    async def _process_with_adk_agent(self, user_request: str, workflow_id: str, 
                                     progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Process request using Google ADK coordinator agent or fallback to specialist agents"""
        
        try:
            logger.info(f"Processing with ADK agent: {user_request[:50]}...")
            
            if progress_callback:
                progress_callback({"workflow_id": workflow_id, "progress": 0.5, "status": "coordinating agents"})
            
            # Check if we have a working ADK coordinator agent
            if self.base_coordinator and hasattr(self.base_coordinator, 'process'):
                # Try to use the actual ADK coordinator agent
                try:
                    logger.info("Attempting to use ADK coordinator agent...")
                    # Note: This would require proper ADK setup and authentication
                    # For now, we'll fall back to specialist agent processing
                    raise Exception("ADK coordinator not fully configured")
                except Exception as e:
                    logger.info(f"ADK coordinator not available, falling back to specialist agents: {e}")
                    # Fall back to processing with specialist agents
                    return await self._process_with_specialist_agents(user_request, workflow_id, progress_callback)
            else:
                # No ADK coordinator available, use specialist agents directly
                logger.info("No ADK coordinator available, using specialist agents")
                return await self._process_with_specialist_agents(user_request, workflow_id, progress_callback)
            
        except Exception as e:
            logger.error(f"ADK processing failed: {e}")
            # Fallback to basic processing
            return await self._process_with_fallback(user_request, workflow_id, progress_callback)
    
    async def _process_with_specialist_agents(self, user_request: str, workflow_id: str,
                                             progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Process request using specialist agents directly"""
        
        logger.info(f"Processing with specialist agents: {user_request[:50]}...")
        
        if progress_callback:
            progress_callback({"workflow_id": workflow_id, "progress": 0.3, "status": "using specialist agents"})
        
        # Extract actual content for processing
        actual_content = self._extract_actual_content(user_request)
        logger.info(f"Extracted content: {actual_content[:100]}...")
        
        results = {
            "status": "completed",
            "mode": "enhanced_specialist_agents",
            "workflow_id": workflow_id,
            "user_request": user_request,
            "timestamp": datetime.now().isoformat(),
            "agent_results": {}
        }
        
        try:
            # Step 1: Data Engineering
            if progress_callback:
                progress_callback({"workflow_id": workflow_id, "progress": 0.4, "status": "running data engineering"})
            
            logger.info("Running DataEngAgent tools...")
            nlp_result = data_eng_tools.run_nlp_pipeline(actual_content)  # Send actual content
            results["agent_results"]["data_eng_nlp"] = nlp_result
            
            # Step 2: OSINT Analysis
            if progress_callback:
                progress_callback({"workflow_id": workflow_id, "progress": 0.6, "status": "running OSINT analysis"})
            
            logger.info("Running OsintAgent tools...")
            classification_result = osint_tools.classify_narrative(actual_content)  # Send actual content
            results["agent_results"]["osint_classification"] = classification_result
            
            # Step 3: Lexicon Analysis
            if progress_callback:
                progress_callback({"workflow_id": workflow_id, "progress": 0.8, "status": "running lexicon analysis"})
            
            logger.info("Running LexiconAgent tools...")
            coded_language_result = lexicon_tools.detect_coded_language(actual_content, "en")  # Send actual content
            results["agent_results"]["lexicon_analysis"] = coded_language_result
            
            # Step 4: Let coordinator agent generate the summary
            if progress_callback:
                progress_callback({"workflow_id": workflow_id, "progress": 0.9, "status": "coordinator generating summary"})
            
            # Pass all results to coordinator agent for final summary generation
            coordinator_summary = await self._delegate_summary_to_coordinator(
                user_request, results["agent_results"], workflow_id
            )
            results["summary"] = coordinator_summary
            
            return results
            
        except Exception as e:
            logger.error(f"Specialist agent processing failed: {e}")
            raise e
    
    async def _process_with_fallback(self, user_request: str, workflow_id: str,
                                    progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Fallback processing when ADK is not available - optimized for efficiency"""
        
        logger.info(f"Processing with optimized fallback mode: {user_request[:50]}...")
        
        if progress_callback:
            progress_callback({"workflow_id": workflow_id, "progress": 0.3, "status": "using optimized fallback processing"})
        
        # Extract actual content for processing
        actual_content = self._extract_actual_content(user_request)
        
        results = {
            "status": "completed",
            "mode": "enhanced_fallback_optimized",
            "workflow_id": workflow_id,
            "user_request": user_request,
            "timestamp": datetime.now().isoformat(),
            "agent_results": {}
        }
        
        try:
            # Step 1: Data Engineering - only if content needs preprocessing
            if len(actual_content) > 50:  # Only for substantial content
                if progress_callback:
                    progress_callback({"workflow_id": workflow_id, "progress": 0.4, "status": "running data engineering"})
                
                logger.info("Running DataEngAgent tools...")
                nlp_result = data_eng_tools.run_nlp_pipeline(actual_content)  # Send actual content, not full request
                results["agent_results"]["data_eng_nlp"] = nlp_result
            
            # Step 2: OSINT Analysis - always required
            if progress_callback:
                progress_callback({"workflow_id": workflow_id, "progress": 0.6, "status": "running OSINT analysis"})
            
            logger.info("Running OsintAgent tools...")
            classification_result = osint_tools.classify_narrative(actual_content)  # Send actual content
            results["agent_results"]["osint_classification"] = classification_result
            
            # Step 3: Lexicon Analysis - only if suspicious terms detected
            suspicious_keywords = ["ghost", "rigged", "fraud", "manipulate", "fake", "bogus", "phony"]
            if any(keyword in actual_content.lower() for keyword in suspicious_keywords):
                if progress_callback:
                    progress_callback({"workflow_id": workflow_id, "progress": 0.8, "status": "running lexicon analysis"})
                
                logger.info("Running LexiconAgent tools...")
                coded_language_result = lexicon_tools.detect_coded_language(actual_content, "en")
                results["agent_results"]["lexicon_analysis"] = coded_language_result
            else:
                logger.info("Skipping lexicon analysis - no suspicious terms detected")
                results["agent_results"]["lexicon_analysis"] = {
                    "status": "skipped",
                    "reason": "No suspicious terms detected in content"
                }
            
            # Step 4: Let coordinator agent generate the summary
            if progress_callback:
                progress_callback({"workflow_id": workflow_id, "progress": 0.9, "status": "coordinator generating summary"})
            
            # Pass all results to coordinator agent for final summary generation
            coordinator_summary = await self._delegate_summary_to_coordinator(
                user_request, results["agent_results"], workflow_id
            )
            results["summary"] = coordinator_summary
            
            return results
            
        except Exception as e:
            logger.error(f"Fallback processing failed: {e}")
            raise e
    
    async def _delegate_summary_to_coordinator(self, user_request: str, 
                                              agent_results: Dict[str, Any], 
                                              workflow_id: str) -> Dict[str, Any]:
        """Delegate summary generation to the coordinator agent using unified ElectionWatchReportTemplate"""
        
        try:
            logger.info(f"Delegating summary generation to coordinator for workflow {workflow_id}")
            
            # Use the unified analysis template for all content types
            report_template = ElectionWatchReportTemplate.get_analysis_template()
            
            # Populate the template with agent results in standardized format
            populated_report = self._populate_analysis_template(
                report_template, user_request, agent_results, workflow_id
            )
            
            logger.info(f"Coordinator summary generated for workflow {workflow_id}")
            return populated_report
            
        except Exception as e:
            logger.error(f"Failed to delegate summary to coordinator: {e}")
            # Return fallback in standardized format
            return self._get_fallback_analysis_report(workflow_id, str(e))
    
    def _populate_analysis_template(self, template: Dict[str, Any], 
                                   user_request: str, 
                                   agent_results: Dict[str, Any], 
                                   workflow_id: str) -> Dict[str, Any]:
        """Populate the unified analysis template with agent results"""
        
        # Create a copy of the template to populate
        report = template.copy()
        
        # Generate unique report ID
        report_id = f"AutoGeneratedReport_{workflow_id[:8]}"
        
        # Extract the actual content from the user request
        actual_content = self._extract_actual_content(user_request)
        
        # Determine content type based on analysis
        content_type = self._determine_content_type(user_request, agent_results)
        analysis_depth = self._determine_analysis_depth(agent_results)
        
        # Populate report metadata
        report["report_metadata"]["report_id"] = report_id
        report["report_metadata"]["analysis_timestamp"] = datetime.now().isoformat()
        report["report_metadata"]["content_type"] = content_type
        report["report_metadata"]["analysis_depth"] = analysis_depth
        
        # Populate narrative classification with enhanced details
        narrative_classification = self._extract_enhanced_narrative_classification(agent_results, actual_content)
        report["narrative_classification"] = narrative_classification
        
        # Populate actors with comprehensive information
        actors = self._extract_enhanced_actors(agent_results, actual_content)
        report["actors"] = actors
        
        # Populate lexicon terms with detailed analysis
        lexicon_terms = self._extract_enhanced_lexicon_terms(agent_results, actual_content)
        report["lexicon_terms"] = lexicon_terms
        
        # Set overall risk level
        report["risk_level"] = narrative_classification.get("threat_level", "Medium")
        
        # Set analysis date
        report["date_analyzed"] = datetime.now().isoformat()
        
        # Generate comprehensive recommendations
        recommendations = self._generate_comprehensive_recommendations(agent_results, narrative_classification)
        report["recommendations"] = recommendations
        
        # Populate analysis insights with technical details
        analysis_insights = self._generate_analysis_insights(agent_results, actual_content, workflow_id)
        report["analysis_insights"] = analysis_insights
        
        return report
    
    def _determine_content_type(self, user_request: str, agent_results: Dict[str, Any]) -> str:
        """Determine the content type being analyzed"""
        
        # Check for file uploads or multimedia indicators
        if "image" in user_request.lower() or "photo" in user_request.lower():
            return "image"
        elif "video" in user_request.lower():
            return "video"
        elif "csv" in user_request.lower() or "dataset" in user_request.lower():
            return "csv"
        elif "Files uploaded:" in user_request and "1" in user_request:
            return "multimedia"
        else:
            return "text"
    
    def _determine_analysis_depth(self, agent_results: Dict[str, Any]) -> str:
        """Determine the depth of analysis performed"""
        
        if len(agent_results) >= 4:
            return "comprehensive"
        elif len(agent_results) >= 2:
            return "standard"
        else:
            return "quick"
    
    def _generate_analysis_insights(self, agent_results: Dict[str, Any], 
                                   actual_content: str, 
                                   workflow_id: str) -> Dict[str, Any]:
        """Generate detailed analysis insights"""
        
        insights = {
            "content_statistics": {
                "word_count": len(actual_content.split()) if actual_content else 0,
                "character_count": len(actual_content) if actual_content else 0,
                "language_detected": "en"  # Could be enhanced with language detection
            },
            "threat_assessment": {
                "overall_risk_score": self._calculate_comprehensive_risk_score(agent_results),
                "violence_potential": self._assess_violence_potential_detailed(agent_results),
                "electoral_impact": self._assess_electoral_impact_detailed(agent_results),
                "social_cohesion_threat": self._assess_social_cohesion_detailed(agent_results),
                "urgency_level": self._determine_urgency_level_detailed(agent_results)
            },
            "processing_metadata": {
                "agents_utilized": list(agent_results.keys()),
                "analysis_duration": 0.0,  # Will be calculated by caller
                "confidence_level": self._calculate_overall_confidence_level(agent_results),
                "data_sources": ["agent_analysis", "lexicon_matching", "pattern_detection"]
            }
        }
        
        return insights
    
    def _generate_comprehensive_recommendations(self, agent_results: Dict[str, Any], 
                                              narrative_classification: Dict[str, str]) -> List[str]:
        """Generate comprehensive recommendations based on all agent analysis"""
        
        recommendations = []
        threat_level = narrative_classification.get("threat_level", "Medium")
        
        # Base recommendations based on threat level
        if threat_level == "High":
            recommendations.extend([
                "Immediate escalation and verification required",
                "Monitor for rapid spread across platforms",
                "Deploy counter-narrative strategies",
                "Alert relevant authorities and stakeholders",
                "Increase monitoring frequency for related content"
            ])
        elif threat_level == "Medium":
            recommendations.extend([
                "Further investigation needed to verify claims and sources",
                "Monitor for narrative spread across platforms",
                "Track actors and their influence networks",
                "Document patterns for trend analysis"
            ])
        else:
            recommendations.extend([
                "Continue standard monitoring protocols",
                "Document findings for baseline analysis",
                "Periodic review recommended"
            ])
        
        # Agent-specific recommendations
        if "lexicon_analysis" in agent_results:
            lexicon = agent_results["lexicon_analysis"]
            if lexicon.get("status") == "success":
                potential_terms = lexicon.get("potential_coded_terms", [])
                if len(potential_terms) > 3:
                    recommendations.append("High concentration of coded language detected - prioritize linguistic analysis")
                elif len(potential_terms) > 0:
                    recommendations.append("Monitor for evolution of detected coded language patterns")
        
        if "osint_classification" in agent_results:
            osint = agent_results["osint_classification"]
            if "narrative_classification" in osint:
                classification = osint["narrative_classification"]
                if classification.get("fraud_indicators", 0) > 0:
                    recommendations.append("Electoral fraud narratives detected - coordinate with election monitoring bodies")
                if classification.get("violence_indicators", 0) > 0:
                    recommendations.append("Violence-related content identified - consider security alert protocols")
                if classification.get("division_indicators", 0) > 0:
                    recommendations.append("Social division content detected - deploy unity messaging strategies")
        
        if "data_eng_nlp" in agent_results:
            nlp = agent_results["data_eng_nlp"]
            if "narrative_detection" in nlp:
                narrative_count = nlp["narrative_detection"].get("narrative_count", 0)
                if narrative_count > 5:
                    recommendations.append("Multiple narrative patterns detected - conduct comprehensive narrative mapping")
        
        # Remove duplicates while preserving order
        unique_recommendations = []
        for rec in recommendations:
            if rec not in unique_recommendations:
                unique_recommendations.append(rec)
        
        return unique_recommendations
    
    def _extract_enhanced_narrative_classification(self, agent_results: Dict[str, Any], actual_content: str) -> Dict[str, Any]:
        """Extract and format narrative classification from OSINT agent results"""
        
        # Default structure
        classification = {
            "theme": "General Content",
            "threat_level": "Low",
            "details": "Standard content analysis completed",
            "confidence_score": 0.0,
            "alternative_themes": [],
            "threat_indicators": []
        }
        
        # PRIMARY: Extract from OSINT agent results
        if "osint_classification" in agent_results:
            osint = agent_results["osint_classification"]
            
            if osint.get("status") == "success" and "narrative_classification" in osint:
                osint_narrative = osint["narrative_classification"]
                
                # Map OSINT agent's themes to unified format
                primary_theme = osint_narrative.get("primary_theme", "general_political")
                confidence = osint_narrative.get("confidence", 0.0)
                
                # Theme mapping
                theme_mapping = {
                    "election_fraud": ("Election Fraud", "High", "Content contains allegations of electoral fraud or manipulation"),
                    "violence_incitement": ("Violence Incitement", "High", "Content contains elements promoting violence or intimidation"),
                    "ethnic_religious_division": ("Ethnic Tensions", "High", "Content promotes ethnic or religious divisions"),
                    "general_political": ("Political Content", "Medium", "General political content with moderate concerns")
                }
                
                if primary_theme in theme_mapping:
                    theme, threat_level, details = theme_mapping[primary_theme]
                    classification.update({
                        "theme": theme,
                        "threat_level": threat_level,
                        "details": details,
                        "confidence_score": confidence
                    })
                
                # Extract threat indicators from OSINT scoring
                fraud_indicators = osint_narrative.get("fraud_indicators", 0)
                violence_indicators = osint_narrative.get("violence_indicators", 0)
                division_indicators = osint_narrative.get("division_indicators", 0)
                
                if fraud_indicators > 0:
                    classification["threat_indicators"].append(f"Electoral fraud patterns ({fraud_indicators} indicators)")
                if violence_indicators > 0:
                    classification["threat_indicators"].append(f"Violence-related content ({violence_indicators} indicators)")
                if division_indicators > 0:
                    classification["threat_indicators"].append(f"Social division themes ({division_indicators} indicators)")
                
                # Add alternative themes based on secondary scores
                scores = [
                    (fraud_indicators, "Election Fraud"),
                    (violence_indicators, "Violence Incitement"), 
                    (division_indicators, "Ethnic Tensions")
                ]
                scores.sort(reverse=True, key=lambda x: x[0])
                
                for score, theme_name in scores[1:]:  # Skip the primary theme
                    if score > 0 and theme_name != classification["theme"]:
                        classification["alternative_themes"].append(theme_name)
        
        # SECONDARY: Enhance with lexicon agent results
        if "lexicon_analysis" in agent_results:
            lexicon = agent_results["lexicon_analysis"]
            if lexicon.get("status") == "success":
                potential_terms = lexicon.get("potential_coded_terms", [])
                if len(potential_terms) > 0:
                    classification["threat_indicators"].append(f"Coded language detected ({len(potential_terms)} terms)")
                    # Only boost confidence if OSINT didn't already provide high confidence
                    if classification["confidence_score"] < 0.6:
                        classification["confidence_score"] = 0.6
            elif lexicon.get("status") == "no_coded_language_detected":
                classification["threat_indicators"].append("No significant coded language patterns detected")
        
        # TERTIARY: Basic fallback if no agent results available
        if classification["confidence_score"] == 0.0 and actual_content:
            # Minimal keyword-based fallback only if agents failed
            text_lower = actual_content.lower()
            
            if any(keyword in text_lower for keyword in ["rigged", "fraud", "ghost voter"]):
                classification.update({
                    "theme": "Election Manipulation",
                    "threat_level": "Medium",
                    "details": "Content contains concerning election-related terminology",
                    "confidence_score": 0.3
                })
            elif any(keyword in text_lower for keyword in ["violence", "attack", "fight"]):
                classification.update({
                    "theme": "Violence Concerns",
                    "threat_level": "Medium", 
                    "details": "Content contains concerning violence-related terminology",
                    "confidence_score": 0.3
                })
        
        # Ensure minimum confidence
        if classification["confidence_score"] == 0.0:
            classification["confidence_score"] = 0.2
        
        return classification
    
    def _extract_enhanced_actors(self, agent_results: Dict[str, Any], actual_content: str) -> List[Dict[str, Any]]:
        """Extract and format actors from OSINT agent results"""
        
        actors = []
        
        # PRIMARY: Extract from OSINT agent results
        if "osint_classification" in agent_results:
            osint = agent_results["osint_classification"]
            
            if osint.get("status") == "success" and "actors_identified" in osint:
                for agent_actor in osint["actors_identified"]:
                    # Format OSINT agent results into unified structure
                    actor = {
                        "name": agent_actor.get("name", "Unknown Actor"),
                        "affiliation": self._map_actor_type_to_affiliation(agent_actor.get("type", "")),
                        "role": agent_actor.get("context", "Mentioned in content"),
                        "influence_level": self._map_confidence_to_influence(agent_actor.get("confidence", 0.0)),
                        "verification_status": "Agent-identified",
                        "social_metrics": {
                            "agent_confidence": agent_actor.get("confidence", 0.0),
                            "actor_type": agent_actor.get("type", "")
                        }
                    }
                    actors.append(actor)
        
        # SECONDARY: Basic fallback only if no agents provided actors
        if not actors and actual_content:
            # Minimal content-based extraction
            text_lower = actual_content.lower()
            
            if any(keyword in text_lower for keyword in ["candidate", "governor", "president"]):
                actors.append({
                    "name": "Political Figure",
                    "affiliation": "Government/Political",
                    "role": "Mentioned political actor",
                    "influence_level": "Unknown",
                    "verification_status": "Content-based detection",
                    "social_metrics": {}
                })
            
            if "official" in text_lower:
                actors.append({
                    "name": "Official Source",
                    "affiliation": "Government/Official", 
                    "role": "Cited authority/source",
                    "influence_level": "Unknown",
                    "verification_status": "Content-based detection",
                    "social_metrics": {}
                })
        
        # Default if no actors found
        if not actors:
            actors.append({
                "name": "Content Source",
                "affiliation": "",
                "role": "Original poster/publisher",
                "influence_level": "Unknown",
                "verification_status": "Default",
                "social_metrics": {}
            })
        
        return actors[:5]  # Limit to 5 actors
    
    def _map_actor_type_to_affiliation(self, actor_type: str) -> str:
        """Map OSINT agent actor types to affiliation categories"""
        mapping = {
            "politician": "Government/Political",
            "political_party": "Political Party",
            "institution": "Government Institution",
            "media": "Media Organization",
            "social_media": "Social Media"
        }
        return mapping.get(actor_type.lower(), "Unknown")
    
    def _map_confidence_to_influence(self, confidence: float) -> str:
        """Map OSINT agent confidence scores to influence levels"""
        if confidence >= 0.8:
            return "High"
        elif confidence >= 0.5:
            return "Medium"
        else:
            return "Low"
    
    def _extract_enhanced_lexicon_terms(self, agent_results: Dict[str, Any], actual_content: str) -> List[Dict[str, Any]]:
        """Extract and format lexicon terms from lexicon agent results"""
        
        lexicon_terms = []
        
        # PRIMARY: Extract from lexicon agent results
        if "lexicon_analysis" in agent_results:
            lexicon = agent_results["lexicon_analysis"]
            
            if lexicon.get("status") == "success":
                potential_terms = lexicon.get("potential_coded_terms", [])
                for term_data in potential_terms:
                    lexicon_terms.append({
                        "term": term_data.get("term_candidate", "Unknown term"),
                        "category": term_data.get("category", "Coded Language"),
                        "context": term_data.get("context_phrase", "Detected through lexicon analysis"),
                        "confidence_score": term_data.get("confidence", 0.5),
                        "language": term_data.get("language_code", "en"),
                        "severity": term_data.get("severity", "Medium"),
                        "definition": term_data.get("definition", "Term identified through lexicon matching")
                    })
            
            elif lexicon.get("status") == "no_coded_language_detected":
                lexicon_terms.append({
                    "term": "Standard language",
                    "category": "Normal",
                    "context": "No coded language patterns detected by lexicon agent",
                    "confidence_score": 0.8,
                    "language": "en",
                    "severity": "Low",
                    "definition": "Content uses standard language without coded terminology"
                })
        
        # SECONDARY: Extract from OSINT misinformation indicators
        if "osint_classification" in agent_results:
            osint = agent_results["osint_classification"]
            if osint.get("status") == "success" and "misinformation_indicators" in osint:
                for indicator in osint["misinformation_indicators"][:3]:  # Limit to first 3
                    lexicon_terms.append({
                        "term": indicator.get("term", "OSINT-detected term"),
                        "category": "OSINT Detection",
                        "context": indicator.get("context", "Identified through OSINT analysis"),
                        "confidence_score": indicator.get("confidence", 0.6),
                        "language": "en",
                        "severity": "Medium",
                        "definition": "Term flagged by OSINT analysis as potential concern"
                    })
        
        # TERTIARY: Minimal fallback if no agent results
        if not lexicon_terms:
            lexicon_terms.append({
                "term": "Election-related terminology",
                "category": "Standard",
                "context": "General election-related language detected",
                "confidence_score": 0.3,
                "language": "en",
                "severity": "Low",
                "definition": "Basic election-related content without specific agent analysis"
            })
        
        # Remove duplicates and sort by confidence
        unique_terms = []
        seen_terms = set()
        
        sorted_terms = sorted(lexicon_terms, key=lambda x: -x["confidence_score"])
        
        for term in sorted_terms:
            term_key = term["term"].lower()
            if term_key not in seen_terms:
                seen_terms.add(term_key)
                unique_terms.append(term)
        
        return unique_terms[:6]  # Limit to 6 most relevant terms
    
    def _calculate_comprehensive_risk_score(self, agent_results: Dict[str, Any]) -> float:
        """Calculate detailed risk score from all agent results"""
        risk_score = 0.0
        
        # Risk from lexicon analysis
        if "lexicon_analysis" in agent_results:
            lexicon = agent_results["lexicon_analysis"]
            potential_terms = lexicon.get("potential_coded_terms", [])
            if len(potential_terms) > 0:
                risk_score += min(3.0 + len(potential_terms) * 0.5, 5.0)
                
                # High confidence terms increase risk
                high_conf_terms = [t for t in potential_terms if t.get("confidence", 0) > 0.8]
                if high_conf_terms:
                    risk_score += 1.0
        
        # Risk from OSINT classification
        if "osint_classification" in agent_results:
            osint = agent_results["osint_classification"]
            if "narrative_classification" in osint:
                classification = osint["narrative_classification"]
                fraud_indicators = classification.get("fraud_indicators", 0)
                violence_indicators = classification.get("violence_indicators", 0)
                division_indicators = classification.get("division_indicators", 0)
                
                risk_score += fraud_indicators * 1.5
                risk_score += violence_indicators * 2.0
                risk_score += division_indicators * 1.0
        
        # Risk from NLP analysis
        if "data_eng_nlp" in agent_results:
            nlp = agent_results["data_eng_nlp"]
            if "risk_assessment" in nlp:
                nlp_risk = nlp["risk_assessment"].get("overall_risk_score", 0.0)
                risk_score += nlp_risk * 2.0
        
        return min(risk_score, 10.0)  # Cap at 10.0
    
    def _assess_violence_potential_detailed(self, agent_results: Dict[str, Any]) -> str:
        """Detailed violence potential assessment"""
        risk_score = self._calculate_comprehensive_risk_score(agent_results)
        
        # Check for specific violence indicators
        violence_keywords = ["violence", "attack", "kill", "destroy", "eliminate"]
        has_violence_indicators = False
        
        if "osint_classification" in agent_results:
            osint = agent_results["osint_classification"]
            if "narrative_classification" in osint:
                violence_indicators = osint["narrative_classification"].get("violence_indicators", 0)
                has_violence_indicators = violence_indicators > 0
        
        if has_violence_indicators or risk_score >= 8.0:
            return "High"
        elif risk_score >= 5.0:
            return "Medium"
        else:
            return "Low"
    
    def _assess_electoral_impact_detailed(self, agent_results: Dict[str, Any]) -> str:
        """Detailed electoral impact assessment"""
        risk_score = self._calculate_comprehensive_risk_score(agent_results)
        
        # Check for electoral process indicators
        electoral_keywords = ["vote", "election", "ballot", "candidate", "fraud", "rigged"]
        
        if risk_score >= 7.0:
            return "High"
        elif risk_score >= 4.0:
            return "Medium"
        else:
            return "Low"
    
    def _assess_social_cohesion_detailed(self, agent_results: Dict[str, Any]) -> str:
        """Detailed social cohesion threat assessment"""
        risk_score = self._calculate_comprehensive_risk_score(agent_results)
        
        # Check for ethnic/religious division indicators
        if "osint_classification" in agent_results:
            osint = agent_results["osint_classification"]
            if "narrative_classification" in osint:
                division_indicators = osint["narrative_classification"].get("division_indicators", 0)
                if division_indicators > 2 or risk_score >= 6.0:
                    return "High"
                elif division_indicators > 0 or risk_score >= 3.0:
                    return "Medium"
        
        return "Low"
    
    def _determine_urgency_level_detailed(self, agent_results: Dict[str, Any]) -> str:
        """Detailed urgency level determination"""
        risk_score = self._calculate_comprehensive_risk_score(agent_results)
        
        if risk_score >= 8.0:
            return "Immediate"
        elif risk_score >= 6.0:
            return "Within 24h"
        elif risk_score >= 4.0:
            return "Within Week"
        else:
            return "Monitoring"
    
    def _calculate_overall_confidence_level(self, agent_results: Dict[str, Any]) -> str:
        """Calculate overall confidence in the analysis"""
        confidence_scores = []
        
        # Confidence from OSINT
        if "osint_classification" in agent_results:
            osint = agent_results["osint_classification"]
            confidence_scores.append(osint.get("confidence", 0.0))
        
        # Confidence from lexicon analysis
        if "lexicon_analysis" in agent_results:
            lexicon = agent_results["lexicon_analysis"]
            if lexicon.get("status") == "success":
                confidence_scores.append(0.8)
            else:
                confidence_scores.append(0.3)
        
        # Confidence from NLP
        if "data_eng_nlp" in agent_results:
            nlp = agent_results["data_eng_nlp"]
            if nlp.get("success"):
                confidence_scores.append(0.7)
            else:
                confidence_scores.append(0.2)
        
        if not confidence_scores:
            return "low"
        
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        
        if avg_confidence >= 0.8:
            return "high"
        elif avg_confidence >= 0.6:
            return "medium"
        else:
            return "low"
    
    def _get_fallback_analysis_report(self, workflow_id: str, error_message: str) -> Dict[str, Any]:
        """Get fallback report in unified analysis format when processing fails"""
        
        return {
            "report_metadata": {
                "report_id": f"AutoGeneratedReport_{workflow_id[:8]}",
                "analysis_timestamp": datetime.now().isoformat(),
                "report_type": "analysis",
                "content_type": "text",
                "analysis_depth": "quick"
            },
            "narrative_classification": {
                "theme": "Processing Error",
                "threat_level": "Unknown",
                "details": f"Analysis could not be completed due to processing error: {error_message}",
                "confidence_score": 0.0,
                "alternative_themes": [],
                "threat_indicators": []
            },
            "actors": [
                {
                    "name": "System",
                    "affiliation": "ElectionWatch",
                    "role": "Error reporter",
                    "influence_level": "",
                    "verification_status": "",
                    "social_metrics": {}
                }
            ],
            "lexicon_terms": [
                {
                    "term": "Processing failure",
                    "category": "System",
                    "context": "Unable to complete analysis due to system error",
                    "confidence_score": 1.0,
                    "language": "en",
                    "severity": "System",
                    "definition": "System processing error"
                }
            ],
            "risk_level": "Unknown",
            "date_analyzed": datetime.now().isoformat(),
            "recommendations": [
                "Retry analysis with updated configuration.",
                "Review system logs for error details.",
                "Contact technical support if issue persists."
            ],
            "analysis_insights": {
                "content_statistics": {
                    "word_count": 0,
                    "character_count": 0,
                    "language_detected": "en"
                },
                "threat_assessment": {
                    "overall_risk_score": 0.0,
                    "violence_potential": "Unknown",
                    "electoral_impact": "Unknown",
                    "social_cohesion_threat": "Unknown",
                    "urgency_level": "Unknown"
                },
                "processing_metadata": {
                    "agents_utilized": [],
                    "analysis_duration": 0.0,
                    "confidence_level": "low",
                    "data_sources": ["error_fallback"]
                }
            }
        }
    
    def _generate_quick_recommendations(self, agent_results: Dict[str, Any], narrative_classification: Dict[str, str]) -> List[str]:
        """Generate recommendations in target format"""
        
        recommendations = []
        
        threat_level = narrative_classification.get("threat_level", "Medium")
        
        if threat_level == "High":
            recommendations.extend([
                "Further investigation is needed to verify the claims and origin of the allegations.",
                "Monitor for the spread of the narrative across platforms.",
                "Track the actors and their influence.",
                "Consider immediate intervention to prevent escalation."
            ])
        elif threat_level == "Medium":
            recommendations.extend([
                "Further investigation is needed to verify the claims and origin of the \"Official reports\".",
                "Monitor for the spread of the narrative across platforms.",
                "Track the actors and their influence."
            ])
        else:
            recommendations.extend([
                "Continue standard monitoring protocols.",
                "Document findings for trend analysis.",
                "Periodic review recommended."
            ])
        
        return recommendations
    
    def _get_fallback_quick_analysis(self, workflow_id: str, error_message: str) -> Dict[str, Any]:
        """Get fallback report in exact target format when processing fails"""
        
        return {
            "report_metadata": {
                "report_id": f"AutoGeneratedReport_{workflow_id[:8]}",
                "analysis_timestamp": datetime.now().isoformat(),
                "report_type": "quick_analysis"
            },
            "narrative_classification": {
                "theme": "Processing Error",
                "threat_level": "Unknown",
                "details": f"Analysis could not be completed due to processing error: {error_message}"
            },
            "actors": [
                {
                    "name": "System",
                    "affiliation": "ElectionWatch",
                    "role": "Error reporter"
                }
            ],
            "lexicon_terms": [
                {
                    "term": "Processing failure",
                    "category": "System",
                    "context": "Unable to complete lexicon analysis due to system error"
                }
            ],
            "risk_level": "Unknown",
            "date_analyzed": datetime.now().isoformat(),
            "recommendations": [
                "Retry analysis with updated configuration.",
                "Review system logs for error details.",
                "Contact technical support if issue persists."
            ]
        }
    
    def _populate_report_template(self, template: Dict[str, Any], 
                                 user_request: str, 
                                 agent_results: Dict[str, Any], 
                                 workflow_id: str) -> Dict[str, Any]:
        """Populate the ElectionWatchReportTemplate with agent results"""
        
        # Create a copy of the template to populate
        report = template.copy()
        
        # Generate unique report ID
        report_id = f"EW-{datetime.now().strftime('%Y%m%d')}-{workflow_id[:8]}"
        
        # Extract the actual content from the user request
        # The user_request might contain metadata, so we need to extract the actual content
        actual_content = self._extract_actual_content(user_request)
        
        # Populate report metadata
        report["report_metadata"].update({
            "report_id": report_id,
            "analysis_timestamp": datetime.now().isoformat(),
            "analyzing_agent": "CoordinatorAgent",
            "report_version": "1.0",
            "confidence_level": self._calculate_overall_confidence(agent_results)
        })
        
        # Populate content analysis - always populate with actual content
        report["content_analysis"]["source_content"]["original_text"] = actual_content[:500] + "..." if len(actual_content) > 500 else actual_content
        report["content_analysis"]["source_content"]["content_type"] = "text"
        report["content_analysis"]["source_content"]["language_detected"] = "en"
        
        # Populate narrative classification if OSINT was successful
        if "osint_classification" in agent_results:
            osint = agent_results["osint_classification"]
            if osint.get("status") == "success" and "classifications" in osint and osint["classifications"]:
                primary_class = osint["classifications"][0]
                report["content_analysis"]["narrative_classification"].update({
                    "primary_theme": primary_class.get("theme", "unknown"),
                    "threat_level": primary_class.get("threat_level", "LOW"),
                    "classification_confidence": primary_class.get("confidence", 0.0),
                    "narrative_details": primary_class.get("details", ""),
                    "misinformation_type": primary_class.get("misinformation_type", "unknown")
                })
            else:
                # Set default values for failed OSINT
                report["content_analysis"]["narrative_classification"].update({
                    "primary_theme": "election_related",
                    "threat_level": "MEDIUM",
                    "classification_confidence": 0.3,
                    "narrative_details": "Content appears to be election-related but detailed classification failed",
                    "misinformation_type": "potential_misinformation"
                })
        else:
            # Set default values if no OSINT results
            report["content_analysis"]["narrative_classification"].update({
                "primary_theme": "election_related",
                "threat_level": "MEDIUM",
                "classification_confidence": 0.3,
                "narrative_details": "Content appears to be election-related",
                "misinformation_type": "potential_misinformation"
            })
        
        # Populate lexicon analysis
        if "lexicon_analysis" in agent_results:
            lexicon = agent_results["lexicon_analysis"]
            if lexicon.get("status") == "success":
                # Set coded language detection based on whether terms were found
                potential_terms = lexicon.get("potential_coded_terms", [])
                report["lexicon_analysis"]["coded_language_detected"] = len(potential_terms) > 0
                
                # Convert potential_coded_terms to terms_detected format
                if potential_terms:
                    report["lexicon_analysis"]["terms_detected"] = [
                        {
                            "term": term.get("term_candidate", ""),
                            "language": term.get("language_code", "en"),
                            "category": "coded_language",
                            "severity": "HIGH" if term.get("confidence", 0) > 0.8 else "MEDIUM" if term.get("confidence", 0) > 0.6 else "LOW",
                            "context_usage": term.get("context_phrase", "")[:100] + "..." if len(term.get("context_phrase", "")) > 100 else term.get("context_phrase", ""),
                            "frequency_in_content": 1,
                            "translation": "",
                            "definition": term.get("definition", ""),
                            "confidence": term.get("confidence", 0.0)
                        }
                        for term in potential_terms
                    ]
                else:
                    report["lexicon_analysis"]["terms_detected"] = []
            else:
                report["lexicon_analysis"]["coded_language_detected"] = False
                report["lexicon_analysis"]["terms_detected"] = []
        else:
            report["lexicon_analysis"]["coded_language_detected"] = False
            report["lexicon_analysis"]["terms_detected"] = []
        
        # Populate risk assessment
        risk_score = self._calculate_risk_score(agent_results)
        report["risk_assessment"].update({
            "overall_risk_score": risk_score,
            "violence_potential": self._assess_violence_potential(risk_score),
            "electoral_impact": self._assess_electoral_impact(risk_score),
            "social_cohesion_threat": self._assess_social_cohesion_threat(risk_score),
            "urgency_level": self._determine_urgency_level(risk_score),
            "recommended_actions": self._generate_recommended_actions(agent_results, risk_score)
        })
        
        # Populate recommendations
        report["recommendations"].update({
            "immediate_actions": self._get_immediate_actions(risk_score),
            "monitoring_suggestions": self._get_monitoring_suggestions(agent_results),
            "stakeholder_notifications": self._get_stakeholder_notifications(risk_score),
            "follow_up_required": risk_score > 5.0,
            "escalation_needed": risk_score > 7.0
        })
        
        # Populate technical metadata
        report["technical_metadata"].update({
            "processing_time_seconds": 0.0,  # Will be calculated by caller
            "models_used": ["coordinator_agent", "enhanced_coordinator"],
            "tool_chain": list(agent_results.keys()),
            "data_sources_accessed": ["agent_results"],
            "api_calls_made": len(agent_results)
        })
        
        return report
    
    def _extract_actual_content(self, user_request: str) -> str:
        """Extract the actual content from the user request, removing metadata"""
        
        # First, try to find file content markers
        if "File '" in user_request:
            # Extract content from file markers
            file_start = user_request.find("File '")
            if file_start != -1:
                # Find the end of the filename
                filename_end = user_request.find("':", file_start)
                if filename_end != -1:
                    # Get content after the filename
                    content_start = filename_end + 2  # Skip ':'
                    content = user_request[content_start:].strip()
                    
                    # Stop at "Additional context" if present
                    if "Additional context:" in content:
                        content = content.split("Additional context:")[0].strip()
                    
                    # Stop at "Please provide" if present
                    if "Please provide" in content:
                        content = content.split("Please provide")[0].strip()
                    
                    return content
        
        # Look for BREAKING: or other content markers
        content_markers = [
            "BREAKING:",
            "Content:",
            "Original text:",
            "Text:"
        ]
        
        for marker in content_markers:
            if marker in user_request:
                marker_pos = user_request.find(marker)
                if marker_pos != -1:
                    content_start = marker_pos + len(marker)
                    content = user_request[content_start:].strip()
                    
                    # Stop at metadata sections
                    for stop_marker in ["Additional context:", "Please provide", "Metadata:", "Context:"]:
                        if stop_marker in content:
                            content = content.split(stop_marker)[0].strip()
                    
                    return content
        
        # If no clear markers, try to extract content between common patterns
        # Remove wrapper text and get the core content
        cleaned_request = user_request
        
        # Remove common wrapper text
        wrapper_patterns = [
            "Analyze the following content for misinformation, narratives, and election-related risks:",
            "Additional context:",
            "Please provide a comprehensive analysis following your standard workflow.",
            "Content: string",
            "Content:",
            "Metadata:",
            "Context:"
        ]
        
        for pattern in wrapper_patterns:
            cleaned_request = cleaned_request.replace(pattern, "").strip()
        
        # Clean up extra whitespace and newlines
        lines = [line.strip() for line in cleaned_request.split('\n') if line.strip()]
        return '\n'.join(lines)
    
    def _calculate_overall_confidence(self, agent_results: Dict[str, Any]) -> str:
        """Calculate overall confidence level from agent results"""
        confidence_scores = []
        
        if "osint_classification" in agent_results:
            osint = agent_results["osint_classification"]
            if "classifications" in osint:
                for cls in osint["classifications"]:
                    if "confidence" in cls:
                        confidence_scores.append(cls["confidence"])
        
        if not confidence_scores:
            return "LOW"
        
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        
        if avg_confidence >= 0.8:
            return "HIGH"
        elif avg_confidence >= 0.6:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _calculate_risk_score(self, agent_results: Dict[str, Any]) -> float:
        """Calculate overall risk score (0.0 to 10.0)"""
        risk_score = 0.0
        
        # Base risk from lexicon analysis
        if "lexicon_analysis" in agent_results:
            lexicon = agent_results["lexicon_analysis"]
            # Check for coded language detection based on potential_coded_terms
            potential_terms = lexicon.get("potential_coded_terms", [])
            if len(potential_terms) > 0:
                risk_score += 3.0
                # Additional risk based on number of terms found
                if len(potential_terms) >= 5:
                    risk_score += 2.0
                elif len(potential_terms) >= 3:
                    risk_score += 1.0
                
                # Check average confidence of terms
                if potential_terms:
                    avg_confidence = sum(term.get("confidence", 0) for term in potential_terms) / len(potential_terms)
                    if avg_confidence > 0.8:
                        risk_score += 1.0
                    elif avg_confidence > 0.6:
                        risk_score += 0.5
        
        # Risk from OSINT classification
        if "osint_classification" in agent_results:
            osint = agent_results["osint_classification"]
            if "classifications" in osint:
                for cls in osint["classifications"]:
                    threat_level = cls.get("threat_level", "LOW")
                    if threat_level == "CRITICAL":
                        risk_score += 5.0
                    elif threat_level == "HIGH":
                        risk_score += 3.0
                    elif threat_level == "MEDIUM":
                        risk_score += 1.5
        
        # Additional risk from content analysis
        # Check for election-related keywords that might indicate higher risk
        election_risk_keywords = ["ghost voters", "rigged", "fraud", "manipulate", "fake", "bogus"]
        if any(keyword in str(agent_results).lower() for keyword in election_risk_keywords):
            risk_score += 1.0
        
        return min(risk_score, 10.0)  # Cap at 10.0
    
    def _assess_violence_potential(self, risk_score: float) -> str:
        """Assess violence potential based on risk score"""
        if risk_score >= 8.0:
            return "HIGH"
        elif risk_score >= 5.0:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _assess_electoral_impact(self, risk_score: float) -> str:
        """Assess electoral impact based on risk score"""
        if risk_score >= 7.0:
            return "HIGH"
        elif risk_score >= 4.0:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _assess_social_cohesion_threat(self, risk_score: float) -> str:
        """Assess social cohesion threat based on risk score"""
        if risk_score >= 6.0:
            return "HIGH"
        elif risk_score >= 3.0:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _determine_urgency_level(self, risk_score: float) -> str:
        """Determine urgency level based on risk score"""
        if risk_score >= 8.0:
            return "IMMEDIATE"
        elif risk_score >= 6.0:
            return "WITHIN_24H"
        elif risk_score >= 4.0:
            return "WITHIN_WEEK"
        else:
            return "MONITORING"
    
    def _generate_recommended_actions(self, agent_results: Dict[str, Any], risk_score: float) -> List[str]:
        """Generate recommended actions based on analysis"""
        actions = []
        
        if risk_score >= 8.0:
            actions.extend([
                "Immediate escalation to security authorities",
                "Alert monitoring team for 24/7 surveillance",
                "Prepare emergency response protocols"
            ])
        elif risk_score >= 6.0:
            actions.extend([
                "Enhanced monitoring and verification",
                "Coordinate with relevant stakeholders",
                "Prepare detailed analysis report"
            ])
        elif risk_score >= 4.0:
            actions.extend([
                "Regular monitoring updates",
                "Document patterns for trend analysis",
                "Schedule follow-up assessment"
            ])
        else:
            actions.append("Continue standard monitoring protocols")
        
        return actions
    
    def _get_immediate_actions(self, risk_score: float) -> List[str]:
        """Get immediate actions based on risk score"""
        if risk_score >= 8.0:
            return ["Alert security team", "Initiate emergency protocols", "Contact authorities"]
        elif risk_score >= 6.0:
            return ["Escalate to senior team", "Increase monitoring frequency", "Prepare briefing"]
        else:
            return ["Continue monitoring", "Document findings"]
    
    def _get_monitoring_suggestions(self, agent_results: Dict[str, Any]) -> List[str]:
        """Get monitoring suggestions based on agent results"""
        suggestions = ["Continue standard monitoring protocols"]
        
        if "lexicon_analysis" in agent_results:
            lexicon = agent_results["lexicon_analysis"]
            if lexicon.get("coded_language_detected"):
                suggestions.append("Monitor for similar coded language patterns")
        
        if "osint_classification" in agent_results:
            osint = agent_results["osint_classification"]
            if "classifications" in osint and len(osint["classifications"]) > 1:
                suggestions.append("Track multiple narrative themes")
        
        return suggestions
    
    def _get_stakeholder_notifications(self, risk_score: float) -> List[str]:
        """Get stakeholder notification recommendations"""
        if risk_score >= 8.0:
            return ["Security authorities", "Election commission", "Law enforcement"]
        elif risk_score >= 6.0:
            return ["Election commission", "Monitoring team", "Senior management"]
        elif risk_score >= 4.0:
            return ["Monitoring team", "Analysis team"]
        else:
            return ["Internal monitoring team"]
    
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get workflow status by ID"""
        if workflow_id in self.active_workflows:
            return self.active_workflows[workflow_id]
        else:
            return {"error": f"Workflow {workflow_id} not found"}
    
    def list_active_workflows(self) -> List[Dict[str, Any]]:
        """List all active workflows"""
        return list(self.active_workflows.values())

# Create global enhanced coordinator instance
enhanced_coordinator = EnhancedCoordinator()

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
            
            # Step 4: Let coordinator agent generate the summary
            logger.info("Delegating summary to coordinator agent...")
            if self.enhanced_coordinator and self.enhanced_coordinator.base_coordinator:
                summary = await self.enhanced_coordinator._delegate_summary_to_coordinator(
                    user_request, results["agent_results"], "traditional_mode"
                )
                results["summary"] = summary
            else:
                # Minimal fallback summary if coordinator not available
                results["summary"] = {
                    "source": "fallback_mode",
                    "status": "coordinator_unavailable",
                    "message": "Analysis completed but coordinator summary not available"
                }
            
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
        # Reference existing agents from election_watch_agents.py instead of recreating them
        from .agent import (
            data_eng_agent,
            osint_agent, 
            lexicon_agent,
            trend_analysis_agent,
            coordinator_agent
        )
        
        logger.info("Google ADK agents imported successfully from election_watch_agents")
        
    except ImportError as e:
        logger.error(f"Failed to import ADK agents from election_watch_agents: {e}")
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