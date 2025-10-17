#!/usr/bin/env python3
"""
ElectionWatch Agent System
==========================

This module contains all ElectionWatch agents and their tool configurations
for election monitoring and misinformation analysis.
"""

import sys
import os
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
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

# === TOOL VALIDATION AND WRAPPING UTILITIES ===
def validate_and_wrap_tool(func, tool_name: str) -> Optional[FunctionTool]:
    """
    Safely wrap a function as a FunctionTool with validation.
    
    Args:
        func: Function to wrap
        tool_name: Name of the tool for logging
        
    Returns:
        FunctionTool if successful, None if failed
    """
    try:
        if func and callable(func):
            return FunctionTool(func=func)
        else:
            logger.warning(f"Tool {tool_name} is not callable or missing")
            return None
    except Exception as e:
        logger.error(f"Failed to wrap tool {tool_name}: {e}")
        return None

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
        
        wrapped_tool = validate_and_wrap_tool(func, f"{module.__name__}.{name}")
        if wrapped_tool:
            tools[name] = wrapped_tool
    
    logger.info(f"Wrapped {len(tools)} tools from {module.__name__}")
    return tools



# === WRAP TOOLS FOR EACH MODULE ===
logger.info("Wrapping tools from all modules...")

# === CLEAN TOOL WRAPPING ===
data_eng_tools_all = wrap_module_functions_with_functiontool(data_eng_tools)
osint_tools_all = wrap_module_functions_with_functiontool(osint_tools)
lexicon_tools_all = wrap_module_functions_with_functiontool(lexicon_tools)
trend_tools_all = wrap_module_functions_with_functiontool(trend_analysis_tools)
report_tools_all = wrap_module_functions_with_functiontool(report_templates)

# Wrap knowledge retrieval tools
knowledge_tools_wrapped = {}
if hasattr(knowledge_retrieval, 'search_knowledge'):
    knowledge_tools_wrapped["search_knowledge"] = validate_and_wrap_tool(
        knowledge_retrieval.search_knowledge, "knowledge_retrieval.search_knowledge"
    )
if hasattr(knowledge_retrieval, 'analyze_content'):
    knowledge_tools_wrapped["analyze_content"] = validate_and_wrap_tool(
        knowledge_retrieval.analyze_content, "knowledge_retrieval.analyze_content"
    )

# Filter out None values
knowledge_tools_wrapped = {k: v for k, v in knowledge_tools_wrapped.items() if v is not None}

# === DEBUGGING AND VALIDATION ===
def log_tool_summary():
    """Log a summary of all available tools for debugging."""
    logger.info("=== AGENT TOOL SUMMARY ===")
    logger.info(f"DataEng tools: {len(data_eng_tools_all)}")
    logger.info(f"OSINT tools: {len(osint_tools_all)}")
    logger.info(f"Lexicon tools: {len(lexicon_tools_all)}")
    logger.info(f"Trend tools: {len(trend_tools_all)}")
    logger.info(f"Report tools: {len(report_tools_all)}")
    logger.info(f"Knowledge tools: {len(knowledge_tools_wrapped)}")
    
    # Log critical tools
    critical_tools = [
        ("process_csv_data", data_eng_tools_all.get("process_csv_data")),
        ("classify_narrative", osint_tools_all.get("classify_narrative")),
        ("detect_coded_language", lexicon_tools_all.get("detect_coded_language")),
        ("get_analysis_template", report_tools_all.get("get_analysis_template")),
        ("search_knowledge", knowledge_tools_wrapped.get("search_knowledge"))
    ]
    
    for tool_name, tool in critical_tools:
        status = "✓" if tool else "✗"
        logger.info(f"{status} {tool_name}")

# Log the summary during import
log_tool_summary()

# === ENHANCED AGENT CONFIGURATION ===
AGENT_CONFIGS = [
    {
        "name": "DataEngAgent",
        "model": "gemini-2.5-flash",
        "description": "Comprehensive data engineering specialist for election monitoring with multimodal analysis capabilities.",
        "instruction": 
        """You are a data engineering specialist for election monitoring with optimized token usage.

            Your role is to process user-provided content including text, CSV files, images, and other data for analysis.

            PROCESSING STEPS:

            1. For text content: Use process_csv_data() or run_nlp_pipeline() for efficient processing
            2. For CSV files: Use process_csv_data() for multi-platform support (Twitter, TikTok, Facebook, etc.)
            3. For images: Extract text using extract_text_from_image() OCR (when available)
            4. For videos: Extract audio transcript using extract_audio_transcript_from_video() (when available)
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

            - process_csv_data() for CSV files and structured data
            - run_nlp_pipeline() for text processing
            - create_pipeline_handoff() for agent handoffs
            - extract_text_from_image() for image OCR
            - extract_audio_transcript_from_video() for video transcription
            - store_analysis_results() for data persistence
            - query_stored_results() for data retrieval

            Always provide clear status updates:

            → DataEngAgent: Processing content with platform detection...
            ✓ DataEngAgent: Processing complete
            ✗ DataEngAgent: Processing failed

            Focus on preparing clean, structured data for downstream agents with minimal token usage.
        """,
        "tools": [
            tool for tool in [
                # Core processing tools
                data_eng_tools_all.get("process_csv_data"),
                data_eng_tools_all.get("run_nlp_pipeline"),
                data_eng_tools_all.get("create_pipeline_handoff"),
                
                # Multimodal tools
                data_eng_tools_all.get("extract_text_from_image"),
                data_eng_tools_all.get("extract_audio_transcript_from_video"),
                
                # Storage and retrieval
                data_eng_tools_all.get("store_analysis_results"),
                data_eng_tools_all.get("query_stored_results"),
                
                # Additional processing tools
                data_eng_tools_all.get("process_social_media_data"),
            ] if tool is not None
        ],
        "output_key": "data_eng_results"
    },
    {
        "name": "OsintAgent",
        "model": "gemini-2.5-flash",  # Using Gemini for ADK compatibility, DISARM analysis through tools
        "description": "Specialist for OSINT analysis using DISARM framework for narrative classification and actor profiling with enhanced election security capabilities.",
        "instruction": 
        """You are an OSINT analysis specialist for election monitoring using a fine-tuned DISARM model.
            Your role is to analyze processed content for narratives, actors, and potential misinformation with enhanced accuracy.

            ANALYSIS STEPS:
            1. Classify narratives using classify_narrative() tool with DISARM framework
            2. Identify political actors and their roles with improved precision
            3. Detect misinformation patterns and indicators using specialized training
            4. Assess potential risks and threats with enhanced detection capabilities
            5. Track keywords and generate actor profiles
            6. Detect coordinated behavior patterns
            7. Analyze image content themes

            Use your tools systematically:
            - classify_narrative() for narrative classification with DISARM framework
            - generate_actor_profile() for detailed actor analysis
            - track_keywords() for keyword monitoring
            - detect_coordinated_behavior() for behavior analysis
            - classify_image_content_theme() for image analysis
            - calculate_influence_metrics() for influence assessment
            - search_knowledge() for background information
            - analyze_content() for detailed content analysis

            Always provide clear status updates:
            → OsintAgent: Analyzing for actors and narratives with DISARM model...
            ✓ OsintAgent: Analysis complete
            ✗ OsintAgent: Analysis failed

            Provide detailed analysis with confidence scores and evidence using your specialized training.
        """,
        "tools": [
            tool for tool in [
                # Core OSINT tools
                osint_tools_all.get("classify_narrative"),
                osint_tools_all.get("generate_actor_profile"),
                osint_tools_all.get("track_keywords"),
                osint_tools_all.get("detect_coordinated_behavior"),
                osint_tools_all.get("classify_image_content_theme"),
                osint_tools_all.get("calculate_influence_metrics"),
                
                # Knowledge integration
                knowledge_tools_wrapped.get("search_knowledge"),
                knowledge_tools_wrapped.get("analyze_content")
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
            1. Detect coded language and dog whistles using detect_coded_language()
            2. Identify potentially harmful terminology
            3. Provide context and definitions for identified terms
            4. Assess severity and potential impact
            5. Update lexicon database with new terms
            6. Retrieve existing term definitions
            7. Provide translation support for multilingual content

            Use your tools systematically:
            - detect_coded_language() for primary coded language detection
            - update_lexicon_term() for adding new terms to database
            - get_lexicon_term() for retrieving term definitions
            - translate_term() for multilingual support

            Always provide clear status updates:
            → LexiconAgent: Detecting coded language...
            ✓ LexiconAgent: Analysis complete
            ✗ LexiconAgent: Analysis failed

            Focus on accuracy and cultural context in language analysis.
        """,
        "tools": [
            tool for tool in [
                # Core lexicon tools
                lexicon_tools_all.get("detect_coded_language"),
                lexicon_tools_all.get("update_lexicon_term"),
                lexicon_tools_all.get("get_lexicon_term"),
                lexicon_tools_all.get("translate_term"),
            ] if tool is not None
        ],
        "output_key": "lexicon_results"
    },
    {
        "name": "TrendAnalysisAgent",
        "model": "gemini-2.5-flash",
        "description": "Comprehensive temporal pattern analysis and early warning specialist.",
        "instruction": 
        """You are the TrendAnalysisAgent specializing in temporal pattern analysis.
            Your role is to analyze processed content for temporal patterns, trends, and early warning indicators.

            ANALYSIS STEPS:
            1. Analyze narrative trends over time using analyze_narrative_trends()
            2. Identify emerging patterns and potential threats
            3. Generate timeline data for visualization using generate_timeline_data()
            4. Create early warning alerts when necessary using generate_early_warning_alert()
            5. Support report template analysis

            Use your tools systematically:
            - analyze_narrative_trends() for comprehensive trend analysis
            - generate_timeline_data() for timeline visualization
            - generate_early_warning_alert() for alert generation
            - get_analysis_template() for template structure

            Always provide clear status updates:
            → TrendAnalysisAgent: Analyzing temporal patterns...
            ✓ TrendAnalysisAgent: Analysis complete
            ✗ TrendAnalysisAgent: Analysis failed

            Focus on temporal patterns and emerging threats.
        """,
        "tools": [
            tool for tool in [
                # Core trend analysis tools
                trend_tools_all.get("analyze_narrative_trends"),
                trend_tools_all.get("generate_timeline_data"),
                trend_tools_all.get("generate_early_warning_alert"),
                
                # Report template integration
                report_tools_all.get("get_analysis_template")
            ] if tool is not None
        ],
        "output_key": "trend_results"
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

# Removed: LexiconAgent and TrendAnalysisAgent for lean 3-agent architecture
# Lexicon functionality integrated into OsintAgent
logger.info("✓ Using lean 3-agent architecture: DataEng → OSINT → Coordinator")

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

        **→ Step 2: Execute Lean Agent Pipeline**
        - Announce: "Executing lean analysis pipeline..."
        - Call agents in sequence: `DataEngAgent` → `OsintAgent`
        - Report real-time status for each (e.g., "→ DataEngAgent: Processing content...", "→ OsintAgent: Extracting actors and narratives...")
        - Ensure that the actual content from the user request is passed to each agent for processing.
        - **AGENT STATUS INTERPRETATION**: 
          * ✓ Completed: Agent returned results with status "success" or "limited_analysis" with meaningful data
          * ✗ Failed: Agent returned status "error" or no usable results
          * Use actual tool outputs, not assumptions - inspect the "status" field and "has_content" field when available
        - If the user requests specific data (e.g., "only actors"), only call `OsintAgent` and deliver the requested output immediately.

        **→ Step 3: Generate Final Report**
        - Do not announce this step separately.
        - Synthesize all agent outputs into a JSON report (unless the user requested partial output).
        - **CRITICAL: Always return your final response in valid JSON format.**
        - **IMPORTANT: Use actual tool results** - extract data from tool responses, don't generate fake data
        - **Agent Status in Report**: Include accurate agent statuses based on actual tool responses:
          * Check the "status" field from each agent's response
          * Look for "has_content", "confidence", "analysis_method" fields
          * Report what actually happened, not what you assume happened
        - Deliver the JSON as the default output for a full workflow.
        - Example JSON structure (use dynamic IDs and content-specific analysis):
        ```json
        {
          "analysis_id": "dynamic_analysis_id",
          "status": "completed",
          "narrative_classification": {
            "theme": "content_specific_theme",
            "confidence": "dynamic_score_based_on_content"
          },
          "actors_identified": [
            {"name": "content_specific_actor", "role": "content_specific_role"}
          ],
          "risk_assessment": {
            "level": "content_based_assessment",
            "factors": ["content_specific_factors"]
          },
          "recommendations": [
            "content_specific_recommendations"
          ]
        }
        ```

           **IMPORTANT: DYNAMIC ANALYSIS REQUIREMENTS**
        - Always analyze the ACTUAL content provided, not pre-defined templates
        - Generate unique analysis IDs based on timestamp and content hash
        - Vary confidence scores based on actual content analysis (0.1-0.95 range)
        - Identify actors and themes specific to the provided content
        - Generate recommendations tailored to the specific analysis findings         
          """,
    
    # Use ADK's native sub-agent pattern - lean 3-agent architecture
    sub_agents=[data_eng_agent, osint_agent],
    tools=[
        # Core agent tools for lean coordination
        AgentTool(data_eng_agent),
        AgentTool(osint_agent),
        
        # Report generation tools
        *[tool for tool in [
            report_tools_all.get("get_analysis_template"),
            report_tools_all.get("export_analysis_report"),
        ] if tool is not None],
        
        # Knowledge integration tools
        *[tool for tool in [
            knowledge_tools_wrapped.get("search_knowledge"),
            knowledge_tools_wrapped.get("analyze_content")
        ] if tool is not None]
    ]
)

# ===== EXPORTS FOR ADK =====

# The ADK will look for 'root_agent' in this file
root_agent = coordinator_agent

# === ENHANCED HEALTH CHECK FUNCTION ===
def check_agent_pipeline_health() -> Dict[str, Any]:
    """
    Check the health of the agent pipeline and return comprehensive status information.
    """
    health_status = {
        "timestamp": datetime.now().isoformat(),
        "agents": {},
        "tool_modules": {},
        "overall_status": "healthy",
        "issues": [],
        "total_tools": 0,
        "consolidation_info": {
            "consolidated_agents": True,
            "consolidation_date": datetime.now().isoformat()
        }
    }
    
    # Check each agent in lean architecture
    agents_to_check = [
        ("DataEngAgent", data_eng_agent),
        ("OsintAgent", osint_agent),
        ("CoordinatorAgent", coordinator_agent)
    ]
    
    for agent_name, agent in agents_to_check:
        tools_count = len(agent.tools) if hasattr(agent, 'tools') else 0
        sub_agents_count = len(agent.sub_agents) if hasattr(agent, 'sub_agents') else 0
        
        agent_status = {
            "name": agent_name,
            "model": getattr(agent, 'model', 'unknown'),
            "tools_count": tools_count,
            "sub_agents_count": sub_agents_count,
            "status": "healthy" if tools_count > 0 else "warning",
            "output_key": getattr(agent, 'output_key', None)
        }
        
        if tools_count == 0:
            health_status["overall_status"] = "warning"
            health_status["issues"].append(f"{agent_name} has no tools available")
        
        health_status["agents"][agent_name] = agent_status
        health_status["total_tools"] += tools_count
    
    # Check tool modules
    tool_modules = {
        "data_eng_tools": len(data_eng_tools_all),
        "osint_tools": len(osint_tools_all),
        "lexicon_tools": len(lexicon_tools_all),
        "trend_analysis_tools": len(trend_tools_all),
        "report_templates": len(report_tools_all),
        "knowledge_retrieval": len(knowledge_tools_wrapped)
    }
    
    health_status["tool_modules"] = tool_modules
    
    # Check for missing critical tools
    critical_tools = [
        ("data_eng_tools", "process_csv_data", data_eng_tools_all),
        ("osint_tools", "classify_narrative", osint_tools_all),
        ("lexicon_tools", "detect_coded_language", lexicon_tools_all),
        ("knowledge_retrieval", "search_knowledge", knowledge_tools_wrapped)
    ]
    
    for module_name, tool_name, tool_dict in critical_tools:
        if tool_name not in tool_dict:
            health_status["issues"].append(f"Critical tool missing: {module_name}.{tool_name}")
    
    # Final status determination
    if len(health_status["issues"]) > 3:
        health_status["overall_status"] = "critical"
    elif len(health_status["issues"]) > 0:
        health_status["overall_status"] = "warning"
    
    return health_status

def get_tool_inventory() -> Dict[str, Any]:
    """
    Get a detailed inventory of all available tools across all modules.
    """
    return {
        "data_eng_tools": list(data_eng_tools_all.keys()),
        "osint_tools": list(osint_tools_all.keys()),
        "lexicon_tools": list(lexicon_tools_all.keys()),
        "trend_analysis_tools": list(trend_tools_all.keys()),
        "report_templates": list(report_tools_all.keys()),
        "knowledge_retrieval": list(knowledge_tools_wrapped.keys()),
        "total_unique_tools": len(set(
            list(data_eng_tools_all.keys()) +
            list(osint_tools_all.keys()) +
            list(lexicon_tools_all.keys()) +
            list(trend_tools_all.keys()) +
            list(report_tools_all.keys()) +
            list(knowledge_tools_wrapped.keys())
        ))
    }

# Clean exports
__all__ = [
    # Core agent exports
    'root_agent', 
    'coordinator_agent',
    'data_eng_agent',
    'osint_agent', 
    'lexicon_agent',
    'trend_analysis_agent',
    
    # Health and diagnostics
    'check_agent_pipeline_health',
    'get_tool_inventory',
    
    # Agent factory
    'create_agent',
    'AGENT_CONFIGS'
] 