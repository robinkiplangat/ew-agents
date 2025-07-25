#!/usr/bin/env python3
"""
ElectionWatch Consolidated Agent System
======================================

This module contains all ElectionWatch agents 
"""

import sys
import os
from google.adk.agents import LlmAgent
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools import FunctionTool

# Handle imports for both local development and deployed environments
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Add paths for different deployment scenarios
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

# Import specialist tools
try:
    from . import (
        data_eng_tools,
        osint_tools,
        lexicon_tools,
        trend_analysis_tools,
        report_templates
    )
except ImportError:
    import data_eng_tools
    import osint_tools
    import lexicon_tools
    import trend_analysis_tools
    import report_templates

# Wrap data_eng_tools functions with FunctionTool locally
collect_social_media_data_tool = FunctionTool(func=data_eng_tools.process_social_media_data)
process_csv_data_tool = FunctionTool(func=data_eng_tools.process_csv_data)
run_nlp_pipeline_tool = FunctionTool(func=data_eng_tools.run_nlp_pipeline)
extract_text_from_image_tool = FunctionTool(func=data_eng_tools.extract_text_from_image)
extract_audio_transcript_from_video_tool = FunctionTool(func=data_eng_tools.extract_audio_transcript_from_video)
store_analysis_results_tool = FunctionTool(func=data_eng_tools.store_analysis_results)
query_stored_results_tool = FunctionTool(func=data_eng_tools.query_stored_results)

# ===== SPECIALIST AGENTS =====

data_eng_agent = LlmAgent(
    name="DataEngAgent",
    model="gemini-2.0-flash-lite-001",
    description="Specialist for data collection, cleaning, NLP preprocessing, and database management.",
    instruction="""You are a data engineering specialist for election monitoring.
    
    Your responsibilities:
    - Extract and clean content from various file formats (CSV, text, images, video)
    - Perform NLP preprocessing and text normalization
    - Manage database and knowledge base operations
    - Prepare data for analysis by other specialists
    
    Use your tools systematically and provide clear status updates.""",
    tools=[
        collect_social_media_data_tool,
        process_csv_data_tool,
        run_nlp_pipeline_tool,
        extract_text_from_image_tool,
        extract_audio_transcript_from_video_tool, 
        store_analysis_results_tool,
        query_stored_results_tool,
    ],
    output_key="data_eng_results"
)

osint_agent = LlmAgent(
    name="OsintAgent", 
    model="gemini-2.0-flash-lite-001",
    description="Specialist for OSINT analysis, narrative classification, and actor profiling.",
    instruction="""You are an OSINT analysis specialist for election monitoring.
    
    Your responsibilities:
    - Classify narratives and content themes
    - Identify and profile political actors
    - Detect coordinated behavior patterns
    - Calculate influence metrics
    - Assess misinformation and disinformation
    
    Provide detailed analysis with confidence scores and evidence.""",
    tools=[
        osint_tools.classify_narrative_tool,
        osint_tools.classify_image_content_theme_tool,
        osint_tools.track_keywords_tool,
        osint_tools.calculate_influence_metrics_tool,
        osint_tools.detect_coordinated_behavior_tool,
        osint_tools.generate_actor_profile_tool,
    ],
    output_key="osint_results"
)

lexicon_agent = LlmAgent(
    name="LexiconAgent",
    model="gemini-2.0-flash-lite-001", 
    description="Specialist for lexicon management and coded language detection.",
    instruction="""You are a lexicon management specialist for election monitoring.
    
    Your responsibilities:
    - Manage multilingual lexicons of election-related terms
    - Detect coded language and dog whistles
    - Provide term translations and context
    - Update lexicon with new terms and patterns
    
    Focus on accuracy and cultural context in language analysis.""",
    tools=[
        lexicon_tools.update_lexicon_term_tool,
        lexicon_tools.get_lexicon_term_tool,
        lexicon_tools.detect_coded_language_tool,
        lexicon_tools.translate_term_tool,
    ],
    output_key="lexicon_results"
)

trend_analysis_agent = LlmAgent(
    name="TrendAnalysisAgent",
    model="gemini-2.0-flash-lite-001",
    description="Specialist for narrative trend analysis and early warning detection.",
    instruction="""You are a trend analysis specialist for election monitoring.
    
    Your responsibilities:
    - Analyze narrative trends and patterns over time
    - Detect early warning signals of escalation
    - Track evolving themes and messaging strategies
    - Provide predictive insights and recommendations
    
    Focus on temporal patterns and emerging threats.""",
    tools=[
        trend_analysis_tools.analyze_narrative_trends_tool,
        trend_analysis_tools.generate_timeline_data_tool,
        trend_analysis_tools.generate_early_warning_alert_tool, 
    ],
    output_key="trend_results"
)

# ===== COORDINATOR AGENT =====

coordinator_agent = LlmAgent(
    name="ElectionWatchCoordinator",
    model="gemini-2.0-flash-lite-001",
    description="Central coordinator for ElectionWatch election analysis system.",
    instruction="""You are the ElectionWatch Coordinator, managing comprehensive election content analysis.

    **WORKFLOW PROTOCOL:**
    
    1. **Content Assessment**: First analyze the input to determine content type and analysis needs
    
    2. **Specialist Delegation**: Route tasks to appropriate specialists:
       - DataEngAgent: For data extraction, cleaning, and preprocessing
       - OsintAgent: For narrative classification and actor analysis
       - LexiconAgent: For coded language detection and term analysis  
       - TrendAnalysisAgent: For pattern analysis and early warnings
    
    3. **Result Synthesis**: Combine specialist outputs into the unified analysis template
    
    **DELEGATION RULES:**
    - Always start with DataEngAgent for content preprocessing
    - Use OsintAgent for narrative and actor analysis
    - Include LexiconAgent for language-specific analysis
    - Add TrendAnalysisAgent for temporal patterns
    - Generate final report using the analysis template
    
    **OUTPUT FORMAT:**
    Always provide results in the standardized analysis template format.
    Include all specialist findings with proper attribution.
    
    **BALANCED EXECUTION:**
    - By default, execute the full workflow and deliver a comprehensive JSON report
    - If the user requests specific outputs (e.g., "only actors"), prioritize that data
    - Communicate progress clearly with status updates
    - Adopt a professional yet approachable tone""",
    
    # Use ADK's native sub-agent pattern
    sub_agents=[
        data_eng_agent,
        osint_agent, 
        lexicon_agent,
        trend_analysis_agent
    ],
    
    # Include reporting tools and agent tools
    tools=[
        AgentTool(data_eng_agent),
        AgentTool(osint_agent),
        AgentTool(lexicon_agent),
        AgentTool(trend_analysis_agent),
        report_templates.generate_analysis_template_tool,   
        report_templates.generate_report_template_tool,    ],
    
    # Save final synthesis to state
    output_key="final_analysis_report"
)

# ===== EXPORTS FOR ADK =====

# The ADK will look for 'root_agent' in this file
root_agent = coordinator_agent

# For backwards compatibility
__all__ = [
    'root_agent', 
    'coordinator_agent',
    'data_eng_agent',
    'osint_agent', 
    'lexicon_agent',
    'trend_analysis_agent'
] 