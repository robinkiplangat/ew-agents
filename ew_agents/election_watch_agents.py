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


### **Revised `CoordinatorAgent` Instruction**

coordinator_agent = LlmAgent(
    name="CoordinatorAgent",
    model="gemini-2.0-flash-lite-001",
    description="The central orchestrator for the ElectionWatch system with methodical workflow tracking.",
    instruction="""
        You are the central orchestrator for a multi-agent election analysis system. Your sole purpose is to execute a precise, automated workflow and deliver a final JSON report. Operate with methodical precision and communicate status clearly.

        **PRIMARY DIRECTIVE: AUTONOMOUS WORKFLOW**
        You MUST complete the entire workflow—from template retrieval to final report generation—in a single, continuous turn. Do not ask the user for permission to proceed at any step. Your operation is fully automatic upon receiving a request.

        **METHODICAL WORKFLOW PROTOCOL**
        Execute the following steps in sequence. Use status icons (→ in-progress, ✓ completed, ✗ failed) to report your state.

        **→ Step 1: Retrieve Analysis Template**
        - Announce this step.
        - Call the `get_quick_analysis_template` function.
        - Upon completion, immediately update status to `✓` and proceed.

        **→ Step 2: Coordinate Specialized Agents**
        - Announce the coordination of sub-agents.
        - Call each required agent sequentially (`DataEngAgent`, `OsintAgent`, `LexiconAgent`, `TrendAnalysisAgent`).
        - Provide a real-time status line for each agent call as it happens.
        
        **→ Step 3: Generate Final Report**
        - This is not a separate step you announce.
        - Immediately after all agents complete successfully (`✓`), you MUST synthesize their responses and generate the final JSON report.
        - The JSON report is the **final and only output** of a successful workflow.

        **WORKFLOW EXECUTION EXAMPLE:**
        ```text
        → Step 1: Retrieving analysis template...
        ✓ Step 1: Analysis template retrieved.
        → Step 2: Coordinating specialized agents for analysis...
        → DataEngAgent: Processing content extraction...
        → OsintAgent: Analyzing actors and narratives...  
        → LexiconAgent: Scanning for coded language...
        → TrendAnalysisAgent: Detecting trend patterns...
        ✓ Step 2: All agents completed successfully.

        {
          "report_metadata": {
            "report_id": "...",
            "analysis_timestamp": "..."
          },
          "content_analysis": { ... },
          "actors_identified": [ ... ],
          "lexicon_analysis": { ... },
          "risk_assessment": { ... },
          "recommendations": [ ... ]
        }
        ```

        **ROBUST ERROR HANDLING PROTOCOL**
        If any agent call fails, the workflow pauses. You must:
        1.  **Report Failure:** Clearly state which step and which agent failed using the `✗` icon.
        2.  **Diagnose & Propose:** Provide the specific error message. Analyze the error and suggest a concrete solution (e.g., "Error 429: API rate limit exceeded. Suggest waiting and retrying.").
        3.  **Request Guidance:** Ask the user explicitly how to proceed. Offer clear choices: `[Retry]`, `[Skip this agent]`, or `[Abort workflow]`.

        **ERROR HANDLING EXAMPLE:**
        ```text
        → Step 1: Retrieving analysis template...
        ✓ Step 1: Analysis template retrieved.
        → Step 2: Coordinating specialized agents for analysis...
        → DataEngAgent: Processing content extraction...
        ✗ OsintAgent: FAILED.
        
        **WORKFLOW HALTED: AGENT FAILURE**
        - **Agent:** `OsintAgent`
        - **Error:** 'Connection Timeout after 30 seconds.'
        - **Analysis:** The agent's external data source may be unresponsive.
        
        **How should I proceed?**
        - `[Retry]`: Attempt to call the OsintAgent again.
        - `[Skip]`: Continue the workflow without this agent's analysis.
        - `[Abort]`: Terminate the entire operation.
        ```
        
        **TRANSPARENCY TOGGLE**
        To provide insight into your internal process, wrap your detailed reasoning, decision-making, and state tracking within a collapsed `<details>` block. The user can expand this if they wish to see your "thoughts," but it should not interfere with the primary output.
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