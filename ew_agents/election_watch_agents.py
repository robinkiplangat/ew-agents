from google.adk.agents import LlmAgent
from google.adk.tools.agent_tool import AgentTool

# Import modules with fallback for different environments
try:
    from . import data_eng_tools, osint_tools, lexicon_tools, trend_analysis_tools, report_templates
except ImportError:
    import data_eng_tools, osint_tools, lexicon_tools, trend_analysis_tools, report_templates

# Define the Specialist Agents

data_eng_agent = LlmAgent(
    name="DataEngAgent",
    model="gemini-2.0-flash-lite-001",
    instruction="You are a data engineering specialist. Use your tools to perform data collection, cleaning, preprocessing, and database management as requested.",
    description="Agent for data collection, data cleaning, NLP preprocessing (text, image, video), and database/graph infrastructure management.",
    tools=[
        data_eng_tools.social_media_collector_tool,
        data_eng_tools.run_nlp_pipeline_tool,
        data_eng_tools.extract_text_from_image_tool,
        data_eng_tools.extract_audio_transcript_from_video_tool,
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

# Define the Coordinator Agent

# Note: To run this agent, you need to set up Google Cloud authentication.
# For local development, you can run `gcloud auth application-default login`.
# The PROJECT_ID and LOCATION should be configured for your GCP environment.
# You might need to set these as environment variables.

coordinator_agent = LlmAgent(
    name="CoordinatorAgent",
    model="gemini-2.0-flash-lite-001",
    description="The central orchestrator for the ElectionWatch system with automated workflow execution.",
    instruction="""
        You are the coordinator of a multi-agent system for election monitoring.
        
        **STREAMLINED WORKFLOW - FULLY AUTOMATED:**
        
        When a user requests analysis of content (csv, text, image, video), you must:
        
        1. **AUTOMATICALLY EXECUTE ALL STEPS** without waiting for user confirmation
        2. **Process in parallel where possible** to minimize response time  
        3. **Generate complete report** in a single response
        
        **FULLY AUTOMATED EXECUTION SEQUENCE:**
        
        When user requests analysis, you MUST automatically execute ALL steps in ONE response:
        
        1. **Get Template**: Call `get_quick_analysis_template` 
        2. **Call All Agents**: Invoke DataEngAgent, OsintAgent, LexiconAgent, TrendAnalysisAgent in parallel
        3. **IMMEDIATELY Generate Report**: After receiving agent responses, automatically populate the template with findings and return complete JSON report
        
        **CRITICAL RULE: NEVER STOP AFTER CALLING AGENTS**
        - After agents respond, you MUST immediately generate and return the final report
        - Do NOT wait for user to ask for the report
        - Do NOT ask user to "proceed" at any step
        - ALWAYS complete the full workflow in a single response
        
        **MANDATORY WORKFLOW:**
        ```
        User: "Analyze this content: [content]"
        
        You automatically execute:
        Step 1: Call get_quick_analysis_template
        Step 2: Call all 4 agents with the content  
        Step 3: IMMEDIATELY return complete JSON report populated with agent findings
        ```
        
        **EXAMPLE RESPONSE STRUCTURE:**
        After calling agents, immediately return:
        ```json
        {
          "report_metadata": { "report_id": "...", "analysis_timestamp": "...", ... },
          "content_analysis": { ... populated from agent responses ... },
          "actors_identified": [ ... from OsintAgent ... ],
          "lexicon_analysis": { ... from LexiconAgent ... },
          "risk_assessment": { ... synthesized from all agents ... },
          "recommendations": [ ... actionable items ... ]
        }
        ```
        
        **NEVER STOP AFTER AGENT CALLS - ALWAYS COMPLETE THE REPORT AUTOMATICALLY!**
    """,
    tools=[
        AgentTool(data_eng_agent),
        AgentTool(osint_agent),
        AgentTool(lexicon_agent),
        AgentTool(trend_analysis_agent),
        report_templates.get_comprehensive_analysis_template_tool,
        report_templates.get_quick_analysis_template_tool,
        report_templates.get_multimedia_analysis_template_tool,
        report_templates.get_trend_monitoring_template_tool,
        report_templates.export_report_json_tool,
    ]
)