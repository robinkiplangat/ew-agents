# ElectionWatch ML Agent System
   pip install accelerate bitsandbytes sentencepiece


This directory (`ml/`) contains the multi-agent system designed for the ElectionWatch platform. It uses the Google Agent Development Kit (ADK) to orchestrate various specialized agents for tasks related to misinformation tracking, actor identification, trend analysis, and lexicon management.

## 🎯 System Overview

ElectionWatch is a comprehensive platform designed to monitor, analyze, and combat misinformation during elections. The system uses advanced AI to detect, classify, and track disinformation narratives in African electoral contexts.

### Core Components

The system is built around a `CoordinatorAgent` that intelligently delegates tasks to a suite of specialist agents:

*   **`DataEngAgent`**: Handles data ingestion, preprocessing (text, and conceptually image/video), and database interactions.
*   **`OsintAgent`**: Performs open-source intelligence tasks like narrative classification (text, and conceptually image), actor profiling, and network analysis.
*   **`LexiconAgent`**: Manages multilingual lexicons, detects coded language, and offers translation.
*   **`TrendAnalysisAgent`**: Analyzes narrative trends over time, generates data for visualizations, and issues early warnings.


#### Enhanced Analysis
- **Real-time Processing**: Fast analysis of text, images, and documents
- **Risk Assessment**: Comprehensive risk evaluation with actionable recommendations
- **Actor Identification**: Track and profile content sources and amplifiers
- **Multilingual Support**: Analysis in multiple African languages

## 🚀 Quick Start

### Prerequisites

*   Python 3.8+
*   Google Cloud SDK installed and configured (for running `LlmAgent` instances live).
    *   Authenticate using: `gcloud auth login`
*   Environment variables for GCP:
    *   `GOOGLE_CLOUD_PROJECT`: Your GCP project ID.
    *   `GOOGLE_CLOUD_LOCATION`: Your GCP region (e.g., `us-central1`).
    *   `MONGODB_ATLAS_URI`: MongoDB connection string
*   Required Python packages (see `requirements.txt`). Install using:
    ```bash
    # For development (flexible versions)
    pip install -r requirements.txt
    
    # For production (exact versions)
    pip install -r requirements.lock
    ```

### Running the System

#### 1. Start the FastAPI Server
```bash
python3 -m venv .venv
source .venv/bin/activate
python3 main.py
```

The server will start on `http://localhost:8080`

#### 2. Access Key Interfaces

- **Development UI**: `http://localhost:8080/dev-ui/?app=ew_agents`
- **Health Check**: `http://localhost:8080/health`



## 📊 API Endpoints

The system provides comprehensive REST API endpoints:

### Core Analysis
- `POST /run_analysis` - **Main misinformation analysis endpoint**
<!-- - `POST /AnalysePosts` - Legacy endpoint (deprecated)
- `POST /get_raw_json` - Raw analysis results
- `POST /submitReport` - Submit manual reports -->

### Data Management
- `GET /analysis/{analysis_id}` - Get specific analysis
- `GET /analyses` - List recent analyses
- `GET /storage/stats` - Storage statistics
- `GET /storage/recent` - Recent analyses

### Utilities
- `GET /health` - Health check
- `GET /dev-ui` - Development UI
- `GET /debug/env-check` - Environment check

**📖 Full API Documentation**: See [`docs/API_GUIDE.md`](docs/API_GUIDE.md) for complete endpoint documentation.

## 🎨 Reports Features

### Professional Report Generation
- **AI-Powered Formatting**: Uses AI to transform raw data into reports
- **Structured Content**: Executive Summary, Key Findings, Risk Assessment, Recommendations
- **Visual Design**: Professional styling with icons, gradients, and clean typography
- **Risk Level Indicators**: Color-coded risk levels (High/Medium/Low)

### PDF Export
- **Professional Layout**: Clean, official document formatting
- **Enhanced Typography**: Proper fonts, spacing, and hierarchy
- **Section Recognition**: Automatic detection and formatting of report sections
- **Metadata Inclusion**: Report ID, generation date, and platform branding

### Web Interface
- **Dropdown Selection**: Choose from available analysis reports
- **Real-time Preview**: View formatted reports instantly
- **Download Options**: One-click PDF download
- **Responsive Design**: Works on desktop and mobile devices

## 🚀 Deployment

### Google Cloud Run (One-command deploy)

We ship a unified script that builds the Docker image and deploys to Cloud Run.

#### Prerequisites
- Google Cloud SDK installed and authenticated
  - `gcloud auth login`
  - Ensure your project has billing enabled
- A billing-enabled GCP project and region (e.g., `us-central1`)

#### Quick start (copy-paste)
```bash
chmod +x build_and_deploy.sh

# Minimal (uses script defaults)
./build_and_deploy.sh

# Recommended (explicit values)
./build_and_deploy.sh \
  --project <YOUR_PROJECT_ID> \
  --region us-central1 \
  --service electionwatch-api \
  --memory 2Gi --cpu 1 --max-instances 5
```

After deploy, get the URL and verify health:
```bash
SERVICE_URL=$(gcloud run services describe electionwatch-api \
  --region us-central1 --format='value(status.url)')
echo "$SERVICE_URL"
curl -s "$SERVICE_URL/health"
```

#### Configuration options
`build_and_deploy.sh` accepts flags or environment variables:
- `--project` (or `PROJECT_ID`)
- `--region` (or `REGION`)
- `--service` (or `SERVICE_NAME`)
- `--memory`, `--cpu`, `--min-instances`, `--max-instances`
- `--concurrency`, `--timeout`, `--port`
- `--sa` (optional service account email)
- `--no-enable-apis` (skip enabling APIs)
- `--skip-test` (skip the health check)

Examples:
```bash
# Using environment variables
PROJECT_ID=my-proj REGION=us-central1 SERVICE_NAME=electionwatch-api \
  ./build_and_deploy.sh --memory 2Gi --cpu 1 --max-instances 5

# Custom service account
./build_and_deploy.sh --project my-proj --region us-central1 \
  --service electionwatch-api --sa ew-agent-service@my-proj.iam.gserviceaccount.com
```

Cloud Run service name rules: lowercase letters, digits, hyphens; must start with a letter and not end with a hyphen.

#### Optional: Secrets (MongoDB, etc.)
If you use persistent storage, store your MongoDB URI in Secret Manager and grant the service access:
```bash
gcloud services enable secretmanager.googleapis.com
echo 'mongodb+srv://...' | gcloud secrets create mongodb-atlas-uri --data-file=- --project <YOUR_PROJECT_ID>
gcloud secrets add-iam-policy-binding mongodb-atlas-uri \
  --member="serviceAccount:ew-agent-service@<YOUR_PROJECT_ID>.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor" --project <YOUR_PROJECT_ID>
gcloud run services update <SERVICE_NAME> --region <REGION> \
  --update-secrets MONGODB_ATLAS_URI=mongodb-atlas-uri:latest
```

#### Advanced wrappers
- `deploy.sh` delegates to `build_and_deploy.sh` and includes optional Secret Manager setup.

#### Troubleshooting
- Billing error when enabling services: link a billing account to your project.
- Invalid service name: follow the service name rules above.
- Permission denied: run `gcloud auth login` and ensure you have `roles/run.admin` and Cloud Build permissions.

## 📦 Dependency Management

### Version Strategy

The project uses a dual-approach dependency management strategy:

**Development (`requirements.txt`)**
- Uses minimum version requirements (`>=`) for flexibility
- Allows for automatic security updates and bug fixes
- Recommended for development and testing environments

**Production (`requirements.lock`)**
- Uses exact version pins (`==`) for reproducibility
- Ensures consistent deployments across environments
- Recommended for production deployments

### Updating Dependencies

```bash
# Update development requirements
pip install --upgrade -r requirements.txt

# Generate new lock file for production
pip freeze > requirements.lock
```

## 🔧 Configuration

### Environment Variables
```bash
# Required
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=europe-west1
MONGODB_ATLAS_URI=mongodb+srv://...

# Optional (for enhanced features)
OPEN_ROUTER_API_KEY=your-openrouter-key  # For AI report formatting
```

### MongoDB Setup
The system uses MongoDB Atlas for persistent storage:
- Analysis results are stored in the `analysis_results` collection
- Report submissions are stored in the `report_submissions` collection
- Automatic indexing and optimization for fast queries

## 📁 Project Structure

```
ew-agents/
├── main.py                 # FastAPI server and endpoints
├── requirements.txt        # Python dependencies
├── ew_agents/             # Agent definitions and tools
│   ├── agent.py           # Main agent orchestrator
│   ├── data_eng_tools.py  # Data engineering tools
│   ├── osint_tools.py     # OSINT analysis tools
│   ├── lexicon_tools.py   # Language and lexicon tools
│   ├── trend_analysis_tools.py  # Trend analysis tools
│   └── mongodb_storage.py # Database storage layer
├── docs/                  # Documentation
│   ├── API_GUIDE.md       # Complete API documentation
│   ├── API_REFERENCE.md   # Legacy API reference
│   └── ...                # Additional guides
├── data/                  # Data storage
│   └── outputs/           # Generated reports and outputs
└── README.md              # This file
```

## 🛠️ Development

### Adding New Endpoints
1. Add the endpoint in `main.py`
2. Update the API documentation in `docs/API_GUIDE.md`
3. Test with curl or the development UI

### Extending Agents
1. Add new tools in the appropriate `*_tools.py` file
2. Update agent definitions in `ew_agents/agent.py`
3. Test with the ADK runner

### Customizing Reports
1. Modify the AI prompt in `format_report_with_ai()`
2. Update PDF styling in `generate_pdf_report()`
3. Customize HTML template in `create_html_template()`

## 📚 Documentation

- **[API Guide](docs/API_GUIDE.md)**: Complete API documentation
- **[API Reference](docs/API_REFERENCE.md)**: Legacy API reference
- **[Project Overview](docs/project_overview.md)**: System architecture
- **[OSINT Tools](docs/osint_tools.md)**: OSINT analysis guide
- **[Multimodal Enhancements](docs/MULTIMODAL_ENHANCEMENTS.md)**: Image/video analysis

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Update documentation
5. Test thoroughly
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the documentation in the `/docs` directory
2. Review the API guide for endpoint usage
3. Test with the health check endpoint
4. Check server logs for error details

---

**Election Watch** 
