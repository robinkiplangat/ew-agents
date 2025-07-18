# ElectionWatch API Reference Guide

## Overview

ElectionWatch is a comprehensive platform designed to monitor, analyze, and combat misinformation during elections. The system uses advanced AI to detect, classify, and track disinformation narratives in African electoral contexts.

**Base URL**: `http://localhost:8080` (local development)  
**Production URL**: `https://ew-agent-service-97682702333.europe-west1.run.app`

## 🚀 Quick Start

### Health Check
```bash
curl http://localhost:8080/health
```

### Development UI
Access the interactive misinformation analysis interface at:
```
http://localhost:8080/dev-ui/?app=ew_agents
```

---

## 🛡️ Core API Endpoints

### 1. Misinformation Detection & Analysis
**Endpoint**: `POST /AnalysePosts`  
**Description**: Analyze text, images, documents, or CSV data for misinformation patterns, narrative extraction, actor identification, and lexicon analysis during African elections.

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

#### Example Request
```bash
curl -X POST http://localhost:8080/AnalysePosts \
  -H "Content-Type: application/json" \
  -d '{
    "analysis_type": "misinformation_detection",
    "text_content": "Yoruba people are fraudsters! They control all the banks and are stealing our resources. Do not let them rig this election like they did in Lagos. #Nigeria2024 #StopTheFraud",
    "priority": "high",
    "metadata": {
      "content_type": "text_post",
      "region": "nigeria", 
      "language": "en"
    }
  }'
```

---

### 2. Submit Report
**Endpoint**: `POST /submitReport`  
**Description**: Submit structured analysis reports for processing, storage, and priority routing.

#### Request Format
```json
{
  "report_id": "unique_report_identifier",
  "report_type": "threat_assessment|narrative_analysis|trend_report|incident_report",
  "findings": "Detailed analysis findings and recommendations...",
  "threat_level": "minimal|low|moderate|high|critical",
  "metadata": {
    "analyst": "system|human_analyst_id",
    "region": "country/region",
    "time_period": "2024-07-01 to 2024-07-18",
    "sources_analyzed": 1500
  },
  "supporting_data": {
    "analysis_ids": ["analysis_123", "analysis_456"],
    "confidence_scores": [0.85, 0.92],
    "trend_indicators": ["increasing", "stable"]
  }
}
```

#### Response Format
```json
{
  "submission_id": "report_1234567890",
  "status": "accepted|processing|rejected",
  "report_id": "ethnic_tension_analysis_001",
  "priority_level": "high",
  "estimated_processing_time": "5-10 minutes",
  "next_steps": [
    "Report queued for expert review",
    "Automated trend analysis initiated"
  ],
  "timestamp": "2024-07-18T04:57:59Z"
}
```

#### Example Request
```bash
curl -X POST http://localhost:8080/submitReport \
  -H "Content-Type: application/json" \
  -d '{
    "report_id": "threat_analysis_001",
    "report_type": "threat_assessment",
    "findings": "Detected elevated ethnic tension narratives with 85% confidence. Identified 3 potential threat actors spreading divisive content.",
    "threat_level": "high",
    "metadata": {
      "analyst": "system",
      "region": "nigeria",
      "sources_analyzed": 250
    }
  }'
```

---

## 🔍 Utility Endpoints

### 3. Health Check
**Endpoint**: `GET /health`  
**Description**: Service health check and system status information.

#### Response Format
```json
{
  "status": "healthy|degraded|unhealthy",
  "timestamp": "2024-07-18T04:57:59Z",
  "adk_available": true,
  "analysis_count": 15,
  "report_count": 3,
  "version": "1.0.0",
  "uptime_seconds": 3600
}
```

### 4. List Applications
**Endpoint**: `GET /list-apps`  
**Description**: List available agent applications and their status.

#### Response Format
```json
["ew_agents"]
```

### 5. Get Analysis Results
**Endpoint**: `GET /analysis/{analysis_id}`  
**Description**: Retrieve previously completed analysis results by ID.

#### Response Format
```json
{
  "analysis_id": "analysis_1234567890",
  "status": "completed",
  "results": { /* Analysis results object */ },
  "created_at": "2024-07-18T04:57:59Z",
  "completed_at": "2024-07-18T04:58:02Z"
}
```

### 6. Get Report Submission
**Endpoint**: `GET /report/{submission_id}`  
**Description**: Retrieve report submission status and details by ID.

#### Response Format
```json
{
  "submission_id": "report_1234567890",
  "report_id": "threat_analysis_001",
  "status": "completed",
  "processing_details": { /* Report processing information */ },
  "submitted_at": "2024-07-18T04:57:59Z"
}
```

---

## 🧪 Development & Testing

### Interactive Misinformation Analysis UI
**Endpoint**: `GET /dev-ui/?app=ew_agents`  
**Description**: Web-based interface for interactive misinformation detection and analysis during African elections.

**Features**:
- File upload support (images, PDFs, CSV files) with OCR processing
- Multi-language support for African languages
- Comprehensive report generation with actor identification
- Report sharing capabilities (copy, download, share)
- Real-time narrative and lexicon analysis

### API Documentation
**Endpoint**: `GET /docs`  
**Description**: Interactive FastAPI documentation (Swagger UI)

**Features**:
- Complete endpoint documentation
- Request/response schema definitions
- Built-in API testing interface
- Model definitions and examples

---

## 📋 Request/Response Schemas

### Analysis Types
- **`misinformation_detection`**: Comprehensive analysis including narrative extraction, actor identification, lexicon analysis, and risk assessment for African electoral contexts

### Content Types
- **`text_post`**: Social media posts and text content
- **`image_content`**: Images and screenshots (with OCR processing)
- **`video_content`**: Video content analysis
- **`document`**: PDF documents and text files
- **`csv_data`**: Structured data in CSV format

### Priority Levels
- **`low`**: Standard processing (5-10 minutes)
- **`medium`**: Expedited processing (2-5 minutes)
- **`high`**: Priority processing (30 seconds - 2 minutes)
- **`critical`**: Immediate processing (< 30 seconds)

### Threat Levels
- **`minimal`**: No significant threats detected
- **`low`**: Minor concerns, routine monitoring
- **`moderate`**: Notable patterns requiring attention
- **`high`**: Significant threats requiring intervention
- **`critical`**: Immediate threats requiring urgent response

---

## 🚨 Error Handling

### HTTP Status Codes
- **200 OK**: Request successful
- **400 Bad Request**: Invalid request format or parameters
- **404 Not Found**: Resource not found (analysis_id, submission_id)
- **422 Unprocessable Entity**: Validation errors
- **500 Internal Server Error**: Server processing error

### Error Response Format
```json
{
  "detail": "Error description",
  "error_code": "VALIDATION_ERROR|PROCESSING_ERROR|NOT_FOUND",
  "timestamp": "2024-07-18T04:57:59Z",
  "request_id": "req_1234567890"
}
```

---

## 🔒 Security & Rate Limiting

### Authentication
- **Development**: No authentication required for localhost
- **Production**: API key authentication (header: `X-API-Key`)

### Rate Limiting
- **Analysis Requests**: 60 requests/minute per IP
- **Report Submissions**: 20 requests/minute per IP
- **Health Checks**: 300 requests/minute per IP

---

## 🌍 Multi-language Support

### Supported Languages (African Election Focus)
- **English** (`en`): Full misinformation detection with African electoral context
- **Hausa** (`ha`): Nigerian election narratives and coded language detection
- **Igbo** (`ig`): Nigerian ethnic and political narrative patterns
- **Yoruba** (`yo`): Nigerian ethnic tension and electoral misinformation
- **Swahili** (`sw`): East African election monitoring (Kenya, Uganda, Tanzania)
- **French** (`fr`): Francophone African elections (Senegal, Ivory Coast, Mali)
- **Arabic** (`ar`): North African election contexts with cultural nuances

### Language Detection
The system automatically detects language when not specified, with fallback to English for unsupported languages.

---

## 📊 Integration Examples

### Python Integration
```python
import requests

# Analyze content for misinformation during African elections
response = requests.post(
    'http://localhost:8080/AnalysePosts',
    json={
        'analysis_type': 'misinformation_detection',
        'text_content': 'Yoruba people are controlling our banks and stealing resources...',
        'priority': 'high',
        'metadata': {
            'content_type': 'text_post',
            'region': 'nigeria',
            'language': 'en'
        }
    }
)
result = response.json()
print(f"Report ID: {result['report_metadata']['report_id']}")
print(f"Risk Level: {result['risk_assessment']['overall_risk']}")
print(f"Actors: {[actor['actor'] for actor in result['actors_identified']]}")
```

### cURL Examples
```bash
# Misinformation detection for African elections
curl -X POST http://localhost:8080/AnalysePosts \
  -H "Content-Type: application/json" \
  -d '{"analysis_type":"misinformation_detection","text_content":"Ethnic content for analysis","priority":"high","metadata":{"content_type":"text_post","region":"nigeria","language":"en"}}'

# Check analysis results
curl http://localhost:8080/analysis/analysis_1234567890

# Upload image for OCR and analysis
curl -X POST http://localhost:8080/AnalysePosts \
  -F "files=@screenshot.png" \
  -F "analysis_type=misinformation_detection" \
  -F "metadata={\"content_type\":\"image_content\",\"region\":\"kenya\"}"
```

---

## 🏗️ System Architecture

### Agent Integration
- **ADK Framework**: Google Agent Development Kit integration
- **Multi-Agent Coordination**: Specialized agents for different analysis types
- **Fallback Processing**: Graceful degradation when ADK unavailable

### Misinformation Detection Pipeline
1. **Content Ingestion**: Text, image (OCR), document, or CSV processing
2. **Language Detection**: Automatic detection of African languages
3. **Narrative Extraction**: Identify harmful narratives and coded language
4. **Actor Identification**: Detect content sources and amplifiers  
5. **Lexicon Analysis**: Analyze terminology for dog whistles and hate speech
6. **Risk Assessment**: Evaluate threat levels and vulnerability factors
7. **Report Generation**: Comprehensive analysis report with recommendations

---

## 📈 Performance & Monitoring

### Response Times
- **Health Check**: < 100ms
- **Simple Analysis**: 1-3 seconds
- **Complex Multi-file**: 5-15 seconds
- **Report Processing**: 2-10 seconds

### Monitoring Endpoints
- **Health**: `/health` - System status and metrics
- **Metrics**: Available via server logs
- **Uptime**: Tracked in health endpoint response

---

## 🤝 Support & Resources

### Documentation
- **API Reference**: This document
- **Interactive Docs**: `/docs` endpoint
- **Development UI**: `/dev-ui/?app=ew_agents`

### Getting Help
- **Local Development**: Check server logs for debugging
- **Production Issues**: Monitor health endpoint for system status
- **API Testing**: Use development UI for interactive testing

---

*Last Updated: July 18, 2024*  
*API Version: 1.0.0*  
*ADK Integration: Enabled* 