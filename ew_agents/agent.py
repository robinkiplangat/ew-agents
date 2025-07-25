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
    
    **WORKFLOW PROTOCOL:**
    1. Process the input data using your tools systematically
    2. Store analysis results for other agents to use
    3. **CRITICAL**: After completing data preprocessing, IMMEDIATELY transfer to OsintAgent
    
    **NEXT STEP:**
    When your analysis is complete, you MUST transfer to OsintAgent for narrative classification and actor analysis.
    
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
    
    **WORKFLOW PROTOCOL:**
    1. Analyze content for narrative classification and actor identification
    2. Use your tools to provide detailed analysis with confidence scores
    3. **CRITICAL**: After completing OSINT analysis, IMMEDIATELY transfer to LexiconAgent
    
    **NEXT STEP MANDATORY:**
    When your analysis is complete, you MUST transfer to LexiconAgent for coded language detection.
    DO NOT transfer back to the coordinator.
    
    Provide detailed analysis with confidence scores and evidence.""",
    tools=[
        osint_tools.classify_narrative_tool,
        # Other tools temporarily disabled for workflow testing
    ],
    output_key="osint_results"
)

lexicon_agent = LlmAgent(
      name="LexiconAgent",
      model="gemini-2.0-flash-lite-001",
      description="Multilingual lexicon specialist for coded language detection.",
      instruction="""You are the LexiconAgent specializing in multilingual coded language detection.
      
      **PRIMARY MISSION:**
      Detect coded language, dog whistles, and significant terms in the content.
      
      **ANALYSIS REQUIREMENTS:**
      1. Identify coded language and dog whistles
      2. Extract significant political/electoral terms
      3. Provide cultural context for language usage
      4. Assess linguistic risk factors
      
      **MANDATORY OUTPUT STRUCTURE:**
      Provide your analysis in this exact format:
      ```
      **LEXICON ANALYSIS COMPLETE:**
      - Coded Language: [list of detected coded terms]
      - Significant Terms: [key political/electoral terms]
      - Risk Assessment: [high/medium/low with explanation]
      - Cultural Context: [relevant background]
      ```
      
      **CRITICAL FINAL STEP:**
      After completing your analysis, you MUST say:
      "LEXICON ANALYSIS COMPLETE - TRANSFERRING TO TREND ANALYSIS"
      Then immediately call: transfer_to_agent(agent_name="TrendAnalysisAgent")
      
      Focus on accuracy and cultural context in language analysis.""",
      tools=[
          # lexicon_tools.update_lexicon_term_tool,
          # lexicon_tools.get_lexicon_term_tool,
          # lexicon_tools.detect_coded_language_tool,
          # lexicon_tools.translate_term_tool,
      ]
  )

trend_analysis_agent = LlmAgent(
      name="TrendAnalysisAgent", 
      model="gemini-2.0-flash-lite-001",
      description="Temporal pattern analysis and early warning specialist.",
      instruction="""You are the TrendAnalysisAgent specializing in temporal pattern analysis.
      
      **PRIMARY MISSION:**
      Analyze narrative trends, temporal patterns, and generate early warnings.
      
      **ANALYSIS REQUIREMENTS:**
      1. Identify narrative trends and patterns
      2. Assess temporal risk factors
      3. Generate early warning indicators
      4. Provide strategic recommendations
      
      **🔥 CRITICAL WORKFLOW COMPLETION:**
      After completing your analysis, you MUST:
      
      1. Call generate_analysis_template_tool(content_type="text", analysis_depth="comprehensive")
      2. Return ONLY the JSON output from that tool call
      3. Do NOT add any additional text or commentary
      4. Do NOT say "TREND ANALYSIS COMPLETE" or any other text
      
      **MANDATORY RESPONSE FORMAT:**
      Your final response must be PURE JSON from the generate_analysis_template_tool.
      Nothing else. Just the JSON.
      
      Example final response:
      {"report_metadata": {...}, "narrative_classification": {...}, "actors": [...], ...}
      
      Focus on temporal patterns and emerging threats.""",
      tools=[
          trend_analysis_tools.analyze_narrative_trends_tool,
          # Add template tool to ensure workflow completion
          report_templates.generate_analysis_template_tool,
          # trend_analysis_tools.generate_timeline_data_tool,
          # trend_analysis_tools.generate_early_warning_alert_tool, 
      ]
  )

# ===== COORDINATOR AGENT =====

coordinator_agent = LlmAgent(
    name="ElectionWatchCoordinator",
    model="gemini-2.0-flash-lite-001",
    description="Central coordinator for ElectionWatch election analysis system.",
    instruction="""You are the ElectionWatchCoordinator with a MANDATORY 5-STEP WORKFLOW.

    **🔥 ABSOLUTE REQUIREMENTS - NO EXCEPTIONS:**
    
    1️⃣ **DataEngAgent** (Data processing) → REQUIRED
    2️⃣ **OsintAgent** (Narrative & actors) → REQUIRED  
    3️⃣ **LexiconAgent** (Coded language) → REQUIRED
    4️⃣ **TrendAnalysisAgent** (Patterns & warnings) → REQUIRED
    5️⃣ **generate_analysis_template_tool()** → MANDATORY FINAL STEP
    
    **⚡ WORKFLOW ENFORCEMENT:**
    
    After receiving output from ANY sub-agent, you MUST:
    1. Say "CHECKPOINT X COMPLETE" (where X = step number)
    2. IMMEDIATELY proceed to the next step
    3. NEVER stop until ALL 5 steps are done
    
    **🎯 CRITICAL FINAL RULE:**
    When TrendAnalysisAgent completes (Step 4), you MUST IMMEDIATELY call:
    ```
    generate_analysis_template_tool(
        content_type="text", 
        analysis_depth="comprehensive"
    )
    ```
    
    **🚫 FORBIDDEN BEHAVIORS:**
    - Stopping after any single agent
    - Skipping the template tool call
    - Providing generic summaries
    - Ending without structured ElectionWatch format
    
    **✅ SUCCESS CRITERIA:**
    Your final output MUST contain:
    - report_metadata 
    - narrative_classification
    - actors
    - lexicon_terms
    - risk_level
    - recommendations
    - analysis_insights
    
    **🚀 START COMMAND:**
    Begin with "🔥 ELECTIONWATCH ANALYSIS - 5 STEPS MANDATORY" then transfer to DataEngAgent.""",
    
    # Use ADK's native sub-agent pattern
    sub_agents=[data_eng_agent, osint_agent, lexicon_agent, trend_analysis_agent],
    tools=[
        # Report generation tools - CRITICAL FOR FINAL STEP
        report_templates.generate_analysis_template_tool,
        report_templates.export_analysis_report_tool,
    ]
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