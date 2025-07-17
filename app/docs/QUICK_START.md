# 🚀 ElectionWatch Agent API - Quick Start

## ✅ Working Solution

After debugging the import issues, here's the **working setup** for your ElectionWatch multi-agent system as an API service:

## 📁 Files Summary

- **`app/simple_agent_api.py`** - ✅ **Working FastAPI service**
- **`app/test_simple_api.py`** - Test script to verify the API
- **`ew_agents/election_watch_agents.py`** - Your agent definitions (✅ Working)

## 🚀 Quick Start

### 1. Start the Service
```bash
cd app/
python simple_agent_api.py
```

Expected output:
```
🚀 Starting ElectionWatch Simple Agent API...
📡 Service will be available at: http://localhost:8000
📚 API Documentation: http://localhost:8000/docs
🔧 Agent Status: Available
```

### 2. Test the Service
```bash
cd app/
python test_simple_api.py
```

### 3. Check Status
```bash
curl http://localhost:8000/health
```

Should return:
```json
{
  "status": "healthy",
  "agents_available": true,
  "coordinator": true,
  "components": {
    "data_eng_agent": "ok",
    "osint_agent": "ok", 
    "lexicon_agent": "ok",
    "trend_analysis_agent": "ok"
  }
}
```

## 📡 API Endpoints

### Core Endpoints
- `GET /` - Service information
- `GET /health` - Health check
- `GET /agents` - List available agents
- `GET /status` - Detailed service status

### Content Processing
- `POST /analyze` - Analyze content
- `POST /submit_finding` - Submit new findings

### Interactive Documentation
- Visit: http://localhost:8000/docs

## 🧪 Usage Examples

### Analyze Content
```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "URGENT: BVAS machines showing errors!",
    "source_platform": "twitter",
    "request_type": "narrative_classification"
  }'
```

### Submit Finding
```bash
curl -X POST "http://localhost:8000/submit_finding" \
  -H "Content-Type: application/json" \
  -d '{
    "content_text": "Election irregularities detected...",
    "source_platform": "facebook",
    "source_url": "https://facebook.com/post/123"
  }'
```

## 🔧 What Was Fixed

### The Problem
- Import error: `ModuleNotFoundError: No module named 'ew_agents.enhanced_coordinator'`
- The complex `agent_api_service.py` was trying to import non-existent modules

### The Solution
- Created `simple_agent_api.py` that uses **existing** `ew_agents/election_watch_agents.py`
- Fixed Python path issues by adding current directory to `sys.path`
- Used simple, working imports from your existing agent files
- Graceful fallback when agents are not available

### Key Changes
```python
# ✅ Working imports
from ew_agents.election_watch_agents import (
    coordinator_agent,
    data_eng_agent,
    osint_agent,
    lexicon_agent,
    trend_analysis_agent
)
```

## 🎯 Integration Ready

Your API now provides:

✅ **REST Endpoints** - Standard HTTP API for integration  
✅ **Health Monitoring** - Service and agent status checks  
✅ **Content Analysis** - Process election-related content  
✅ **Finding Submission** - Accept new findings for analysis  
✅ **Agent Coordination** - Access to all 5 agents  
✅ **Auto Documentation** - OpenAPI/Swagger docs  
✅ **Error Handling** - Graceful degradation when components unavailable  

## 🏗️ Application Integration

### Frontend (JavaScript)
```javascript
const analyzeContent = async (content) => {
  const response = await fetch('http://localhost:8000/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      content,
      source_platform: 'web'
    })
  });
  return response.json();
};
```

### Python Client
```python
import requests

def submit_finding(content, platform):
    response = requests.post('http://localhost:8000/submit_finding', 
        json={
            'content_text': content,
            'source_platform': platform
        })
    return response.json()
```

## 🚀 Production Deployment

### Local Development
```bash
cd app/
python simple_agent_api.py
```

### Production with Gunicorn
```bash
cd app/
pip install gunicorn
gunicorn simple_agent_api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Environment Variables
```bash
export PORT=8000
export GOOGLE_CLOUD_PROJECT=your-project-id  # Optional for ADK
export MONGODB_ATLAS_URI=your-mongo-uri      # For data storage
```

## ✅ Success Indicators

1. **Service starts without errors**
2. **Health endpoint returns `"status": "healthy"`**
3. **Agents endpoint lists 5 available agents**
4. **Analysis endpoint processes content and returns structured results**
5. **Interactive docs available at `/docs`**

## 🎉 You're Ready!

Your ElectionWatch multi-agent system is now running as a **production-ready web API** that can:

1. **Accept findings** from external applications
2. **Process content** through your multi-agent system
3. **Return structured analysis** results
4. **Monitor system health** and agent availability
5. **Scale horizontally** with load balancers
6. **Integrate easily** with web/mobile applications

The API handles all the complexity of agent coordination while providing simple HTTP endpoints for integration! 