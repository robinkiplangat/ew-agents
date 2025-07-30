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

coordinator_agent = LlmAgent(
    name="CoordinatorAgent",
    model="gemini-2.0-flash-lite-001",
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
        - Call the `get_quick_analysis_template` function.
        - Update to `✓ Template retrieved` upon success, then proceed.

        **→ Step 2: Coordinate Specialized Agents**
        - Announce: "Coordinating analysis with specialized agents..."
        - Call agents (`DataEngAgent`, `OsintAgent`, `LexiconAgent`, `TrendAnalysisAgent`) sequentially, reporting real-time status for each (e.g., "→ DataEngAgent: Extracting content...").
        - If the user requests specific data (e.g., actors), only call relevant agents (e.g., `OsintAgent`) and deliver the requested output immediately.

        **→ Step 3: Generate Final Report**
        - Do not announce this step separately.
        - Synthesize all agent outputs into a JSON report (unless the user requested partial output).
        - Deliver the JSON as the default output for a full workflow.

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
    sub_agents=[
        data_eng_agent,
        osint_agent,
        lexicon_agent,
        trend_analysis_agent,
    ],
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