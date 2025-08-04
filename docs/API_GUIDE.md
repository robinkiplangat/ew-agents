# ElectionWatch API Guide

## Overview

ElectionWatch is a comprehensive platform designed to monitor, analyze, and combat misinformation during elections. The system uses advanced AI to detect, classify, and track disinformation narratives in African electoral contexts.

**Base URL**: `http://localhost:8080` (local development)  
**Production URL**: `https://electionwatch-agents-api-97682702333.europe-west1.run.app`

## 🎯 Main Analysis Endpoint

**`POST /run_analysis`** is the **primary and recommended endpoint** for all analysis requests. The legacy `/AnalysePosts` endpoint is deprecated but maintained for backward compatibility.

## 🚀 Quick Start

### Health Check
```bash
curl http://localhost:8080/health
```

### Quick Analysis Example
```bash
curl -X POST "http://localhost:8080/run_analysis" \
  -H "Content-Type: application/json" \
  -d '{
    "analysis_type": "misinformation_detection",
    "text_content": "Sample content to analyze for misinformation...",
    "priority": "medium",
    "metadata": {
      "content_type": "text_post",
      "region": "nigeria",
      "language": "en"
    }
  }'
```

### Development UI
Access the interactive misinformation analysis interface at:
```
http://localhost:8080/dev-ui/?app=ew_agents
```

---

## 📋 API Endpoints Overview

### Core Analysis Endpoints
- `POST /run_analysis` - **Main misinformation analysis endpoint**
- `POST /AnalysePosts` - Legacy analysis endpoint (deprecated)
- `POST /get_raw_json` - Get raw analysis results
- `POST /submitReport` - Submit manual reports

### Data Retrieval Endpoints
- `GET /analysis/{analysis_id}` - Get specific analysis results
- `GET /report/{submission_id}` - Get specific report submission
- `GET /analyses` - List recent analyses
- `GET /analysis-template` - Get analysis template

### Storage & Management Endpoints
- `GET /storage/stats` - Get storage statistics
- `GET /storage/recent` - Get recent analyses from storage
- `GET /storage-info` - Get detailed storage information
- `GET /storage/test-connection` - Test MongoDB connection

### Reports System Endpoints
- `GET /view_reports` - Web interface for viewing reports
- `GET /api/reports/available` - Get list of available reports
- `GET /api/reports/generate/{analysis_id}` - Generate formatted report
- `GET /api/reports/download/{analysis_id}` - Download PDF report

### Utility Endpoints
- `GET /health` - Health check
- `GET /dev-ui` - Development UI redirect
- `GET /debug/env-check` - Environment variable check

---

## 🛡️ Core Analysis Endpoints

### 1. Main Analysis Endpoint
**Endpoint**: `POST /run_analysis`  
**Description**: **Primary endpoint** for analyzing text, images, documents, or CSV data for misinformation patterns, narrative extraction, actor identification, and lexicon analysis. This is the recommended endpoint for all analysis requests.

#### Request Format
```json
{
  "analysis_type": "misinformation_detection",
  "text_content": "Social media post or content to analyze for misinformation...",
  "priority": "high",
  "files": ["optional file uploads - images, PDFs, CSV"],
  "metadata": {
    "content_type": "text_post|image_content|video_content|document|csv_data",
    "region": "nigeria|kenya|ghana|south_africa|senegal|uganda|other",
    "language": "en|ha|ig|yo|sw|fr|ar|auto"
  }
}
```

#### Response Format
```json
{
  "report_metadata": {
    "report_id": "analysis_1234567890",
    "analysis_timestamp": "2024-07-18T04:57:59Z",
    "content_source": "Social Media Post",
    "content_type": "text_post"
  },
  "content_analysis": {
    "data_preprocessing": "Text extracted and cleaned using advanced NLP techniques",
    "key_themes": "Ethnic division, economic issues, political commentary, potential hate speech",
    "sentiment_analysis": "Mixed. Negative sentiment towards specific ethnic groups evident",
    "topic_modeling": "Ethnic relations, politics, economics"
  },
  "actors_identified": [
    {
      "actor": "Content Source",
      "role": "Original Poster/Publisher", 
      "activity": "Sharing content with potentially divisive messaging"
    }
  ],
  "lexicon_analysis": {
    "coded_language_detection": "Identified potential dog whistles and coded language",
    "harmful_terminology": "Terms promoting negative stereotypes about specific ethnic groups",
    "translation_support": "Content analyzed in detected language"
  },
  "risk_assessment": {
    "overall_risk": "High|Medium|Low",
    "risk_factors": [
      "Promotion of ethnic division and potential hate speech",
      "Dissemination of misinformation about electoral processes"
    ],
    "vulnerability_assessment": "Content targets specific groups, potentially making them vulnerable"
  },
  "recommendations": [
    "Monitor content for further amplification and spread",
    "Flag content for potential violations of community standards",
    "Alert relevant authorities regarding potential hate speech",
    "Conduct further investigation into narratives and actors involved"
  ]
}
```

### 2. Legacy Analysis Endpoint
**Endpoint**: `POST /AnalysePosts`  
**Description**: **Legacy endpoint** - Use `/run_analysis` instead. This endpoint is maintained for backward compatibility but is deprecated.

#### Request Format
Same as `/run_analysis`

#### Response Format
Same as `/run_analysis`

### 3. Raw JSON Analysis
**Endpoint**: `POST /get_raw_json`  
**Description**: Get raw analysis results in JSON format without additional processing.

#### Request Format
Same as `/run_analysis`

#### Response Format
Raw analysis data in JSON format with all intermediate processing steps.

### 4. Submit Report
**Endpoint**: `POST /submitReport`  
**Description**: Submit manual reports for analysis and tracking.

#### Request Format
```json
{
  "report_id": "unique_report_identifier",
  "report_type": "misinformation|threat|intelligence",
  "findings": "Detailed findings and analysis",
  "threat_level": "high|medium|low",
  "metadata": {
    "analyst": "analyst_name",
    "region": "geographic_region",
    "time_period": "analysis_time_period",
    "sources_analyzed": 5
  },
  "supporting_data": {
    "analysis_ids": ["analysis_1", "analysis_2"],
    "confidence_scores": [0.85, 0.92],
    "trend_indicators": ["increasing", "spreading"]
  }
}
```

---

## 📊 Reports System Endpoints

### 1. View Reports Interface
**Endpoint**: `GET /view_reports`  
**Description**: Web interface for viewing and managing analysis reports.

**Response**: HTML page with dropdown to select reports and view/download options.

### 2. Get Available Reports
**Endpoint**: `GET /api/reports/available`  
**Description**: Get list of all available analysis reports from the database.

#### Response Format
```json
{
  "success": true,
  "reports": [
    {
      "analysis_id": "analysis_1753775756",
      "analysis_type": "misinformation_detection",
      "timestamp": "2024-07-29T14:56:22Z",
      "risk_level": "high",
      "content_preview": "Brief preview of analyzed content..."
    }
  ]
}
```

### 3. Generate Formatted Report
**Endpoint**: `GET /api/reports/generate/{analysis_id}`  
**Description**: Generate a professionally formatted HTML report from analysis data using AI.

#### Path Parameters
- `analysis_id`: The ID of the analysis to format

#### Response Format
```json
{
  "success": true,
  "analysis_id": "analysis_1753775756",
  "formatted_report": "<html>...professionally formatted report...</html>"
}
```

### 4. Download PDF Report
**Endpoint**: `GET /api/reports/download/{analysis_id}`  
**Description**: Download a professionally formatted PDF report.

#### Path Parameters
- `analysis_id`: The ID of the analysis to download

#### Response
- **Content-Type**: `application/pdf`
- **Headers**: 
  - `Content-Disposition`: `attachment; filename=electionwatch_report_{analysis_id}.pdf`
- **Body**: PDF file content

---

## 📈 Data Retrieval Endpoints

### 1. Get Analysis Results
**Endpoint**: `GET /analysis/{analysis_id}`  
**Description**: Retrieve specific analysis results by ID.

#### Path Parameters
- `analysis_id`: The ID of the analysis to retrieve

#### Response Format
```json
{
  "analysis_id": "analysis_1753775756",
  "data": {
    "llm_response": "Detailed analysis response...",
    "structured_report": {
      "report_metadata": {...},
      "risk_level": "high",
      "date_analyzed": "2024-07-29T14:56:22Z"
    },
    "metadata": {...},
    "timestamp": "2024-07-29T14:56:22Z"
  }
}
```

### 2. Get Report Submission
**Endpoint**: `GET /report/{submission_id}`  
**Description**: Retrieve specific report submission by ID.

#### Path Parameters
- `submission_id`: The ID of the report submission to retrieve

### 3. List Recent Analyses
**Endpoint**: `GET /analyses`  
**Description**: Get list of recent analyses with optional limit.

#### Query Parameters
- `limit` (optional): Number of analyses to return (default: 20)

### 4. Get Analysis Template
**Endpoint**: `GET /analysis-template`  
**Description**: Get template for analysis requests.

---

## 💾 Storage & Management Endpoints

### 1. Storage Statistics
**Endpoint**: `GET /storage/stats`  
**Description**: Get comprehensive storage statistics.

#### Response Format
```json
{
  "total_analyses": 150,
  "total_reports": 25,
  "storage_size": "2.5MB",
  "recent_activity": {
    "last_24h": 15,
    "last_week": 45
  }
}
```

### 2. Recent Analyses
**Endpoint**: `GET /storage/recent`  
**Description**: Get recent analyses from storage with optional limit.

#### Query Parameters
- `limit` (optional): Number of analyses to return (default: 10)

### 3. Storage Information
**Endpoint**: `GET /storage-info`  
**Description**: Get detailed storage information including database status.

### 4. Test Connection
**Endpoint**: `GET /storage/test-connection`  
**Description**: Test MongoDB connection and configuration.

---

## 🛠️ Utility Endpoints

### 1. Health Check
**Endpoint**: `GET /health`  
**Description**: Check system health and status.

#### Response Format
```json
{
  "status": "healthy",
  "timestamp": "2024-07-29T14:56:22Z",
  "version": "1.0.0"
}
```

### 2. Development UI
**Endpoint**: `GET /dev-ui`  
**Description**: Redirect to development UI interface.

### 3. Environment Check
**Endpoint**: `GET /debug/env-check`  
**Description**: Check environment variables and configuration.

---

## 🔧 Error Handling

All endpoints return appropriate HTTP status codes:

- `200 OK`: Successful request
- `400 Bad Request`: Invalid request parameters
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error

Error responses include a JSON object with:
```json
{
  "detail": "Error description",
  "error_code": "ERROR_CODE"
}
```

---

## 📝 Usage Examples

### Basic Analysis Request
```bash
curl -X POST http://localhost:8080/AnalysePosts \
  -H "Content-Type: application/json" \
  -d '{
    "analysis_type": "misinformation_detection",
    "text_content": "Sample content to analyze",
    "priority": "high",
    "metadata": {
      "content_type": "text_post",
      "region": "nigeria",
      "language": "en"
    }
  }'
```

### Generate Report
```bash
curl http://localhost:8080/api/reports/generate/analysis_1753775756
```

### Download PDF Report
```bash
curl -o report.pdf http://localhost:8080/api/reports/download/analysis_1753775756
```

### Get Available Reports
```bash
curl http://localhost:8080/api/reports/available
```

---

## 🔐 Authentication & Security

Currently, the API does not require authentication for local development. For production deployment, consider implementing:

- API key authentication
- Rate limiting
- CORS configuration
- Input validation and sanitization

---

## 📚 Additional Resources

- **Development UI**: `http://localhost:8080/dev-ui/?app=ew_agents`
- **Reports Interface**: `http://localhost:8080/view_reports`
- **Health Check**: `http://localhost:8080/health`
- **API Documentation**: This guide
- **Project Documentation**: See `/docs` directory for detailed guides 