# ElectionWatch ML Agent System

This directory (`ml/`) contains the multi-agent system designed for the ElectionWatch platform. It uses the Google Agent Development Kit (ADK) to orchestrate various specialized agents for tasks related to misinformation tracking, actor identification, trend analysis, and lexicon management.

## System Overview

The system is built around a `CoordinatorAgent` that intelligently delegates tasks to a suite of specialist agents:

*   **`DataEngAgent`**: Handles data ingestion, preprocessing (text, and conceptually image/video), and database interactions.
*   **`OsintAgent`**: Performs open-source intelligence tasks like narrative classification (text, and conceptually image), actor profiling, and network analysis.
*   **`LexiconAgent`**: Manages multilingual lexicons, detects coded language, and offers (mock) translation.
*   **`TrendAnalysisAgent`**: Analyzes narrative trends over time, generates data for visualizations, and issues early warnings.

All tools within these agents are currently implemented as mock functions, simulating their behavior without live API calls or complex computations. This allows for testing the agent interaction logic.

## Prerequisites

*   Python 3.8+
*   Google Cloud SDK installed and configured (for running `LlmAgent` instances live).
    *   Authenticate using: `gcloud auth application-default login`
*   Environment variables for GCP:
    *   `GOOGLE_CLOUD_PROJECT`: Your GCP project ID.
    *   `GOOGLE_CLOUD_LOCATION`: Your GCP region (e.g., `europe-west1`).
*   Required Python packages (see `ml/requirements.txt`). Install using:
    ```bash
    pip install -r ml/requirements.txt
    ```

## Running the Agent System

The primary way to interact with the system is through the `CoordinatorAgent`. You can run the system for interactive testing and development using the `adk run` command:

```bash
adk run ml.main:coordinator_agent
```

For batch processing, you can use the `agents.py` script:
```bash
python -m ml.agents
```
This script demonstrates how to send queries to the `CoordinatorAgent` for batch processing of data.

**To run live queries that invoke the LLM-backed `CoordinatorAgent`:**
1.  Ensure you have met all GCP prerequisites mentioned above.
2.  Modify `ml/agents/election_watch_agents.py`:
    *   In the `coordinator_agent` definition, uncomment and set your `project_id` and `location`.
    *   You may also specify a `model` (e.g., "gemini-1.5-pro-001").

## Agent and Tool Details

For detailed information on each agent, their specific tools, and how to extend them, please refer to:

*   **`ml/agents/AGENTS.md`**: Provides an overview of each agent's role, its tools, and development guidelines.
*   Tool implementations:
    *   `ml/agents/data_eng_tools.py`
    *   `ml/agents/osint_tools.py`
    *   `ml/agents/lexicon_tools.py`
    *   `ml/agents/trend_analysis_tools.py`
*   Agent definitions:
    *   `ml/agents/election_watch_agents.py`


## Project Structure (Current - `ml` directory focus)

*   `agents/`: Contains all agent definitions and tool implementations.
    *   `election_watch_agents.py`: Defines all agents (`CoordinatorAgent`, `DataEngAgent`, etc.).
    *   `data_eng_tools.py`, `osint_tools.py`, `lexicon_tools.py`, `trend_analysis_tools.py`: Implement mock tools for each specialist agent.
    *   `AGENTS.md`: Detailed documentation about the agents.
*   `main.py`: Entry point for `adk run`.
*   `requirements.txt`: Python dependencies for the agent system.
*   `README.md`: This file.
*   (Potentially old service files: `models.py`, `services.py`, `Dockerfile`)
