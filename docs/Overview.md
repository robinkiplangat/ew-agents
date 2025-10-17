## Project index (high level)

- API app
  - `main.py`: FastAPI app factory, endpoints, report formatting, PDF export, Mongo storage adapters.
    - Endpoints: `/run_analysis`, `/submitReport`, `/view_reports`, `/api/reports/available`, `/api/reports/generate/{id}`, `/api/reports/download/{id}`, `/health`
    - Runs on port 8080 by default (Docker/compose/env)

- Agents and orchestration
  - `ew_agents/agent.py`: Defines ADK agents
    - `DataEngAgent` (CSV/text preprocessing), `OsintAgent` (narratives, actors), `LexiconAgent` (coded language), `TrendAnalysisAgent` (trends)
  - `ew_agents/coordinator_integration.py`: Enhanced coordinator orchestration and fallbacks; synthesizes unified report structure

- Tools (agent capabilities)
  - `ew_agents/data_eng_tools.py`: CSV platform detection (TikTok/Twitter/Facebook), column mapping, content structuring, NLP helpers
  - `ew_agents/osint_tools.py`: Narrative classification via fine-tuned model if available else pattern-based fallback; actor extraction; misinfo indicators
  - `ew_agents/lexicon_tools.py`: Mongo-backed lexicon CRUD and coded language detection
  - `ew_agents/trend_analysis_tools.py`: Trend/timeline/alert placeholders

- Knowledge and models
  - `ew_agents/knowledge_retrieval.py`: LlamaIndex + MongoDB semantic search across collections (narratives, techniques, meta-narratives, actors); used to enrich `/run_analysis`
  - `ew_agents/model_loader.py`: Optional HF `ArapCheruiyot/disarm_ew-llama3_merged_16bit` loader (torch/transformers) for DISARM; used by `osint_tools` if available

- Storage and secrets
  - `ew_agents/mongodb_storage.py`: Pymongo-based storage with global `storage`; async helpers for insert/find/list
  - `ew_agents/secret_manager.py`: Google Secret Manager helper for `MONGODB_ATLAS_URI` (fallback to env)

- Reporting
  - `ew_agents/report_templates.py`: Unified analysis JSON template + helpers
  - HTML template under `ew_agents/templates/view_reports.html`; PDF generation in `main.py`

- Deployment/config
  - `Dockerfile`: Slim Python 3.12; installs root `requirements.txt`; exposes 8080
  - `docker-compose.yml`: Maps 8080:8080; references non-present `Dockerfile.optimized` (note)
  - `env.example.txt`: Shows required env (e.g., `MONGODB_ATLAS_URI`, `OPEN_ROUTER_API_KEY`, Google Project, etc.)

- Documentation
  - `docs/API_GUIDE.md`, `docs/API_REFERENCE.md`, quick start and endpoint specs

- Requirements
  - Root `requirements.txt`: FastAPI, ADK, Vertex AI, pymongo, pandas/numpy, transformers/torch, reportlab
  - `ew_agents/requirements.txt`: pinned libs for agent tools (optional duplicate)

### Key endpoint details

```700:771:/Users/robin/Downloads/Robin/projets/ew-agents/main.py
@app.post("/run_analysis")
async def run_analysis(
    text: str = Form(None),
    files: List[UploadFile] = File(default=[]),
    analysis_type: str = Form("misinformation_detection"),
    priority: str = Form("medium"),
    source: str = Form("api_upload"),
    metadata: str = Form("{}")
):
    # Processes CSV/text files, builds structured content,
    # runs ADK InMemoryRunner over `root_agent`, enriches via knowledge base,
    # stores to MongoDB, returns unified report JSON
```

- File processing supports CSV (`text/csv`) with structured TikTok detection via `data_eng_tools.process_csv_data`.
- Knowledge enrichment via `search_knowledge` on Mongo collections; sets narrative/lexicon/risk and recommendations.
- Storage: `store_analysis_result` persists to `analysis_results` if MongoDB configured.

### Env and runtime

- Server: `PORT=8080` (env/Docker/compose) and `HOST=0.0.0.0`
- Critical env: `MONGODB_ATLAS_URI` (or `MONGODB_URI`), `OPEN_ROUTER_API_KEY` (optional for pretty report HTML), `GOOGLE_CLOUD_PROJECT`
- Optional: Vertex AI creds if using `vertex_ai_integration.py`

### Data and samples

- Input CSV for TikTok: `data/raw/Sample_tiktok_20l.csv`
- Expected sample output: `data/outputs/tiktok_sample_response.json`
- Actual responses are stored by our test under `data/outputs/tiktok_actual_response.json`

- Known gap: compose references `Dockerfile.optimized` which is not present.

- The API is already running on 8080 (confirmed). No changes made.

- If you want, next I can:
  - Add a lightweight “semantic comparator” script for regression checks against `tiktok_sample_response.json`
  - Tighten `/run_analysis` TikTok path to map actors/themes closer to the sample (using `data_eng_tools` outputs + `osint_tools` actor extraction)
  - Wire a local Mongo (or mock) for knowledge enrichment if `MONGODB_ATLAS_URI` isn’t set

- Summary:
  - Mapped endpoints, agents, tools, knowledge, storage, and deployment files.
  - Highlighted `/run_analysis` flow and CSV handling paths.
  - Flagged required env for storage/reporting and a compose Dockerfile mismatch.
  - Ready to proceed with feature additions or testing harness integration.