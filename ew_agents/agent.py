#!/usr/bin/env python3
"""
ElectionWatch Consolidated Agent System
======================================

This module contains all ElectionWatch agents 
"""

import sys
import os
import logging
from datetime import datetime
from typing import Dict, Any
from google.adk.agents import LlmAgent
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools import FunctionTool
import inspect

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        report_templates,
        knowledge_retrieval
    )
except ImportError:
    import data_eng_tools
    import osint_tools
    import lexicon_tools
    import trend_analysis_tools
    import report_templates
    import knowledge_retrieval

# === DRY TOOL WRAPPING UTILITY ===
def wrap_module_functions_with_functiontool(module, include=None, exclude=None):
    """
    Wraps all public functions in a module with FunctionTool, optionally filtering by include/exclude lists.
    Returns a dict: {function_name: FunctionTool}
    """
    tools = {}
    for name, func in inspect.getmembers(module, inspect.isfunction):
        if name.startswith("_"):
            continue
        if include and name not in include:
            continue
        if exclude and name in exclude:
            continue
        tools[name] = FunctionTool(func=func)
    return tools

# === WRAP TOOLS FOR EACH MODULE ===
data_eng_tools_wrapped = wrap_module_functions_with_functiontool(data_eng_tools)
osint_tools_wrapped = wrap_module_functions_with_functiontool(osint_tools)
lexicon_tools_wrapped = wrap_module_functions_with_functiontool(lexicon_tools)
trend_analysis_tools_wrapped = wrap_module_functions_with_functiontool(trend_analysis_tools)
report_templates_wrapped = wrap_module_functions_with_functiontool(report_templates)

# Wrap knowledge retrieval tools
knowledge_tools_wrapped = {
    "search_knowledge": FunctionTool(func=knowledge_retrieval.search_knowledge),
    "analyze_content": FunctionTool(func=knowledge_retrieval.analyze_content)
}

# === AGENT CONFIGURATION ===
AGENT_CONFIGS = [
    {
        "name": "DataEngAgent",
        "model": "gemini-2.5-flash",
        "description": "Specialist for data mining, cleaning, NLP preprocessing, and database management with optimized token usage.",
        "instruction": 
        """You are a data engineering specialist for election monitoring with optimized token usage.

            Your role is to process user-provided content including text, CSV files, images, and other data for analysis.

            PROCESSING STEPS:

            1. For text content: Use process_text_content_optimized() for efficient processing
            2. For CSV files: Use process_csv_data() for multi-platform support (Twitter, TikTok, Facebook, etc.)
            3. For images: Extract text using OCR (when available)
            4. For videos: Extract audio transcript (when available)
            5. Create pipeline handoff using create_pipeline_handoff() for downstream agents

            PLATFORM SUPPORT:
            - Twitter: Tweets, retweets, political affiliations
            - TikTok: Video transcripts, hashtags, engagement metrics
            - Facebook: Posts, comments, shares, reactions
            - Instagram: Captions, hashtags, engagement
            - YouTube: Video titles, descriptions, comments
            - Generic: Any CSV with social media data

            OPTIMIZATION GUIDELINES:
            - Use token-efficient processing to minimize costs
            - Extract key insights and structured data
            - Provide clean handoffs to downstream agents
            - Focus on election-related content analysis
            - Automatically detect platform and apply appropriate processing

            Use your tools systematically:

            - process_text_content_optimized() for text content
            - process_csv_data() for multi-platform CSV files
            - create_pipeline_handoff() for agent handoffs
            - extract_text_from_image() for image OCR
            - extract_audio_transcript_from_video() for video transcription

            Always provide clear status updates:

            → DataEngAgent: Processing content with platform detection...
            ✓ DataEngAgent: Processing complete
            ✗ DataEngAgent: Processing failed

            Focus on preparing clean, structured data for downstream agents with minimal token usage.
        """,
        "tools": [
            tool for tool in [
                data_eng_tools_wrapped.get("process_text_content_optimized"),
                data_eng_tools_wrapped.get("process_csv_data"),
                data_eng_tools_wrapped.get("create_pipeline_handoff"),
                data_eng_tools_wrapped.get("extract_text_from_image"),
                data_eng_tools_wrapped.get("extract_audio_transcript_from_video"),
                data_eng_tools_wrapped.get("store_analysis_results"),
                data_eng_tools_wrapped.get("query_stored_results"),
            ] if tool is not None
        ],
        "output_key": "data_eng_results"
    },
    {
        "name": "OsintAgent",
        "model": "gemini-2.5-flash",
        "description": "Specialist for OSINT analysis, narrative classification, and actor profiling.",
        "instruction": 
        """You are an OSINT analysis specialist for election monitoring.
            Your role is to analyze processed content for narratives, actors, and potential misinformation.

            ANALYSIS STEPS:
            1. Classify narratives using classify_narrative() tool
            2. Identify political actors and their roles
            3. Detect misinformation patterns and indicators
            4. Assess potential risks and threats

            Use your tools systematically:
            - classify_narrative() for narrative classification
            - search_knowledge() for background information
            - analyze_content() for detailed content analysis

            Always provide clear status updates:
            → OsintAgent: Analyzing for actors and narratives...
            ✓ OsintAgent: Analysis complete
            ✗ OsintAgent: Analysis failed

            Provide detailed analysis with confidence scores and evidence.
        """,
        "tools": [
            tool for tool in [
                osint_tools_wrapped.get("classify_narrative"),
                knowledge_tools_wrapped["search_knowledge"],
                knowledge_tools_wrapped["analyze_content"]
            ] if tool is not None
        ],
        "output_key": "osint_results"
    },
    {
        "name": "LexiconAgent",
        "model": "gemini-2.5-flash",
        "description": "Multilingual lexicon specialist for coded language detection.",
        "instruction": 
        """You are the LexiconAgent specializing in multilingual coded language detection.

            Your role is to identify coded language, dog whistles, and potentially harmful terminology in election-related content.

            ANALYSIS STEPS:
            1. Detect coded language and dog whistles
            2. Identify potentially harmful terminology
            3. Provide context and definitions for identified terms
            4. Assess severity and potential impact

            Always provide clear status updates:
            → LexiconAgent: Detecting coded language...
            ✓ LexiconAgent: Analysis complete
            ✗ LexiconAgent: Analysis failed

            Focus on accuracy and cultural context in language analysis.
        """,
        "tools": [
            tool for tool in [
                lexicon_tools_wrapped.get("update_lexicon_term"),
                lexicon_tools_wrapped.get("get_lexicon_term"),
                lexicon_tools_wrapped.get("detect_coded_language"),
            ] if tool is not None
        ]
    },
    {
        "name": "TrendAnalysisAgent",
        "model": "gemini-2.5-flash",
        "description": "Temporal pattern analysis and early warning specialist.",
        "instruction": 
        """You are the TrendAnalysisAgent specializing in temporal pattern analysis.
            Your role is to analyze processed content for temporal patterns, trends, and early warning indicators.

            ANALYSIS STEPS:
            1. Analyze narrative trends over time using analyze_narrative_trends()
            2. Identify emerging patterns and potential threats
            3. Generate timeline data for visualization
            4. Create early warning alerts when necessary

            Use your tools systematically:
            - analyze_narrative_trends() for trend analysis
            - get_analysis_template() for template structure

            Always provide clear status updates:
            → TrendAnalysisAgent: Analyzing temporal patterns...
            ✓ TrendAnalysisAgent: Analysis complete
            ✗ TrendAnalysisAgent: Analysis failed

            Focus on temporal patterns and emerging threats.
        """,
        "tools": [
            tool for tool in [
                trend_analysis_tools_wrapped.get("analyze_narrative_trends"),
                report_templates_wrapped.get("get_analysis_template")
            ] if tool is not None
        ]
    }
]

# === AGENT FACTORY ===
def create_agent(config):
    # Filter out None tools and log any missing tools
    available_tools = [t for t in config["tools"] if t]
    missing_tools = [i for i, t in enumerate(config["tools"]) if t is None]
    
    if missing_tools:
        logger.warning(f"Agent {config['name']}: Missing tools at indices {missing_tools}")
    
    if not available_tools:
        logger.error(f"Agent {config['name']}: No tools available!")
        # Create a minimal agent with basic functionality
        return LlmAgent(
            name=config["name"],
            model=config["model"],
            description=config["description"],
            instruction=config["instruction"] + "\n\n⚠️ WARNING: No specialized tools available. Using basic analysis only.",
            tools=[],
            output_key=config.get("output_key")
        )
    
    return LlmAgent(
        name=config["name"],
        model=config["model"],
        description=config["description"],
        instruction=config["instruction"],
        tools=available_tools,
        output_key=config.get("output_key")
    )

# === INSTANTIATE AGENTS ===
logger.info("Creating ElectionWatch agents...")

data_eng_agent = create_agent(AGENT_CONFIGS[0])
logger.info(f"✓ DataEngAgent created with {len(data_eng_agent.tools)} tools")

osint_agent = create_agent(AGENT_CONFIGS[1])
logger.info(f"✓ OsintAgent created with {len(osint_agent.tools)} tools")

lexicon_agent = create_agent(AGENT_CONFIGS[2])
logger.info(f"✓ LexiconAgent created with {len(lexicon_agent.tools)} tools")

trend_analysis_agent = create_agent(AGENT_CONFIGS[3])
logger.info(f"✓ TrendAnalysisAgent created with {len(trend_analysis_agent.tools)} tools")

# === COORDINATOR AGENT (unchanged) ===

# ===== COORDINATOR AGENT =====

coordinator_agent = LlmAgent(
    name="CoordinatorAgent",
    model="gemini-2.5-flash",
    description="The central orchestrator for the ElectionWatch system, managing a methodical workflow while adapting to user needs with clear, actionable outputs.",
    instruction="""
        You are the CoordinatorAgent for the ElectionWatch system, designed to analyze election-related data with precision and clarity. Your primary role is to execute a methodical workflow and deliver a comprehensive JSON report by default. However, you are adaptable—capable of responding to specific requests (e.g., only actors or interim results) while maintaining a transparent, user-friendly approach.

        **CORE OBJECTIVE: BALANCED EXECUTION**
        - By default, execute the full workflow and deliver a JSON report synthesizing all agent outputs.
        - If the user requests specific outputs (e.g., "only actors"), prioritize delivering that data immediately, using available results or running only the necessary steps.
        - Communicate progress clearly with status updates (→ in-progress, ✓ completed, ✗ failed) to keep the user informed.
        - Adopt a professional yet approachable tone, ensuring the user feels supported rather than dictated to.

        **METHODICAL WORKFLOW PROTOCOL**
        Execute the following steps unless the user specifies a partial request. Track and report progress methodically.

        **→ Step 1: Retrieve Analysis Template**
        - Announce: "Starting with the analysis template..."
        - Call the `get_analysis_template` function.
        - Update to `✓ Template retrieved` upon success, then proceed.

        **→ Step 2: Coordinate Specialized Agents**
        - Announce: "Coordinating analysis with specialized agents..."
        - Call agents (`DataEngAgent`, `OsintAgent`, `LexiconAgent`, `TrendAnalysisAgent`) sequentially, reporting real-time status for each (e.g., "→ DataEngAgent: Extracting content...").
        - Ensure that the actual content from the user request is passed to each agent for processing.
        - If the user requests specific data (e.g., "only actors"), only call relevant agents (e.g., `OsintAgent`) and deliver the requested output immediately.

        **→ Step 3: Generate Final Report**
        - Do not announce this step separately.
        - Synthesize all agent outputs into a JSON report (unless the user requested partial output).
        - **CRITICAL: Always return your final response in valid JSON format.**
        - Deliver the JSON as the default output for a full workflow.
        - Example JSON structure:
        ```json
        {
          "analysis_id": "analysis_123",
          "status": "completed",
          "narrative_classification": {
            "theme": "election_integrity",
            "confidence": 0.85
          },
          "actors_identified": [
            {"name": "Candidate X", "role": "political_figure"}
          ],
          "risk_assessment": {
            "level": "medium",
            "factors": ["misinformation", "polarization"]
          },
          "recommendations": [
            "Monitor social media for similar narratives",
            "Verify claims with official sources"
          ]
        }
        ```

        **PARTIAL OUTPUT HANDLING**
        - If the user requests specific data (e.g., "only actors"), check available results from prior agent calls or run only the necessary agent (e.g., `OsintAgent`).
        - Deliver the requested data in a concise format, e.g.:
          ```json
          {
            "actors_identified": [
              {"actor": "Candidate X", "role": "Mentioned in relation to voter fraud"},
              {"actor": "Unknown - 'Official Reports'", "role": "Source of the claim (needs verification)"}
            ]
          }
          """,
    
    # Use ADK's native sub-agent pattern
    sub_agents=[data_eng_agent, osint_agent, lexicon_agent, trend_analysis_agent],
    tools=[
        # Report generation tools - CRITICAL FOR FINAL STEP
        report_templates.generate_analysis_template_tool,
        report_templates.export_analysis_report_tool,
        knowledge_tools_wrapped["search_knowledge"],
        knowledge_tools_wrapped["analyze_content"]
    ]
)

# ===== EXPORTS FOR ADK =====

# The ADK will look for 'root_agent' in this file
root_agent = coordinator_agent

# === HEALTH CHECK FUNCTION ===
def check_agent_pipeline_health() -> Dict[str, Any]:
    """
    Check the health of the agent pipeline and return status information.
    """
    health_status = {
        "timestamp": datetime.now().isoformat(),
        "agents": {},
        "overall_status": "healthy",
        "issues": []
    }
    
    # Check each agent
    agents_to_check = [
        ("DataEngAgent", data_eng_agent),
        ("OsintAgent", osint_agent),
        ("LexiconAgent", lexicon_agent),
        ("TrendAnalysisAgent", trend_analysis_agent),
        ("CoordinatorAgent", coordinator_agent)
    ]
    
    for agent_name, agent in agents_to_check:
        agent_status = {
            "name": agent_name,
            "model": agent.model,
            "tools_count": len(agent.tools),
            "status": "healthy" if len(agent.tools) > 0 else "warning"
        }
        
        if len(agent.tools) == 0:
            health_status["overall_status"] = "warning"
            health_status["issues"].append(f"{agent_name} has no tools available")
        
        health_status["agents"][agent_name] = agent_status
    
    return health_status

# For backwards compatibility
__all__ = [
    'root_agent', 
    'coordinator_agent',
    'data_eng_agent',
    'osint_agent', 
    'lexicon_agent',
    'trend_analysis_agent',
    'check_agent_pipeline_health'
] 