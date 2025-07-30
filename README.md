# ElectionWatch ML Agent System

This directory (`ml/`) contains the multi-agent system designed for the ElectionWatch platform. It uses the Google Agent Development Kit (ADK) to orchestrate various specialized agents for tasks related to misinformation tracking, actor identification, trend analysis, and lexicon management.

## 🎯 System Overview

ElectionWatch is a comprehensive platform designed to monitor, analyze, and combat misinformation during elections. The system uses advanced AI to detect, classify, and track disinformation narratives in African electoral contexts.

### Core Components

The system is built around a `CoordinatorAgent` that intelligently delegates tasks to a suite of specialist agents:

*   **`DataEngAgent`**: Handles data ingestion, preprocessing (text, and conceptually image/video), and database interactions.
*   **`OsintAgent`**: Performs open-source intelligence tasks like narrative classification (text, and conceptually image), actor profiling, and network analysis.
*   **`LexiconAgent`**: Manages multilingual lexicons, detects coded language, and offers (mock) translation.
*   **`TrendAnalysisAgent`**: Analyzes narrative trends over time, generates data for visualizations, and issues early warnings.

### 🆕 New Features

#### Reports System
- **Report Generation**: Transform raw analysis data into clean, professional reports using Qwen LLM
- **PDF Export**: Download reports as professionally formatted PDF documents
- **Web Interface**: User-friendly interface for viewing and managing reports

#### Enhanced Analysis
- **Real-time Processing**: Fast analysis of text, images, and documents
- **Risk Assessment**: Comprehensive risk evaluation with actionable recommendations
- **Actor Identification**: Track and profile content sources and amplifiers
- **Multilingual Support**: Analysis in multiple African languages

## 🚀 Quick Start

### Prerequisites

*   Python 3.8+
*   Google Cloud SDK installed and configured (for running `LlmAgent` instances live).
    *   Authenticate using: `gcloud auth application-default login`
*   Environment variables for GCP:
    *   `GOOGLE_CLOUD_PROJECT`: Your GCP project ID.
    *   `GOOGLE_CLOUD_LOCATION`: Your GCP region (e.g., `europe-west1`).
    *   `OPEN_ROUTER_API_KEY`: For Qwen LLM report formatting
    *   `MONGODB_ATLAS_URI`: MongoDB connection string
*   Required Python packages (see `ml/requirements.txt`). Install using:
    ```bash
    pip install -r ml/requirements.txt
    ```

### Running the System

#### 1. Start the FastAPI Server
```bash
cd ml
source .venv/bin/activate
python3 main.py
```

The server will start on `http://localhost:8080`

#### 2. Access Key Interfaces

- **Development UI**: `http://localhost:8080/dev-ui/?app=ew_agents`
- **Reports Interface**: `http://localhost:8080/view_reports`
- **Health Check**: `http://localhost:8080/health`

#### 3. Run Agent System (Alternative)
For interactive testing and development using the `CoordinatorAgent`:

```bash
adk run ml.main:coordinator_agent
```

For batch processing:
```bash
python -m ml.agents
```

## 📊 API Endpoints

The system provides comprehensive REST API endpoints:

### Core Analysis
- `POST /run_analysis` - **Main misinformation analysis endpoint**
- `POST /AnalysePosts` - Legacy endpoint (deprecated)
- `POST /get_raw_json` - Raw analysis results
- `POST /submitReport` - Submit manual reports

### Reports System
- `GET /view_reports` - Web interface for reports
- `GET /api/reports/available` - List available reports
- `GET /api/reports/generate/{analysis_id}` - Generate formatted report
- `GET /api/reports/download/{analysis_id}` - Download PDF report

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
- **LLM-Powered Formatting**: Uses Qwen LLM to transform raw data into reports
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

## 🔧 Configuration

### Environment Variables
```bash
# Required
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=europe-west1
MONGODB_ATLAS_URI=mongodb+srv://...

# Optional (for enhanced features)
OPEN_ROUTER_API_KEY=your-openrouter-key  # For Qwen LLM report formatting
```

### MongoDB Setup
The system uses MongoDB Atlas for persistent storage:
- Analysis results are stored in the `analysis_results` collection
- Report submissions are stored in the `report_submissions` collection
- Automatic indexing and optimization for fast queries

## 📁 Project Structure

```
ml/
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
1. Modify the LLM prompt in `format_report_with_qwen()`
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
