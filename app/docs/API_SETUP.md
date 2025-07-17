# ElectionWatch Agent API Service

A FastAPI-based web service for running the ElectionWatch multi-agent system with REST endpoints.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_api.txt
```

### 2. Start the Service
```bash
cd app/
python simple_agent_api.py
```

The service will be available at:
- **API**: http://localhost:8000
- **Documentation**: http://localhost:8000/docs

### 3. Test the Service
```bash
# Check health
curl http://localhost:8000/health

# Test analysis
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"content": "Test content", "source_platform": "twitter"}'
```

## 📡 API Endpoints

### Core Endpoints

#### `POST /analyze` - Analyze Content
Analyze content using the multi-agent system:
```json
{
  "content": "URGENT: BVAS machines hacked in Lagos!",
  "source_platform": "twitter",
  "request_type": "narrative_classification",
  "include_trends": true,
  "include_actors": true
}
```

#### `POST /submit_finding` - Submit New Finding
Submit new content for analysis and storage:
```json
{
  "content_text": "Breaking: Election irregularities detected",
  "source_platform": "facebook",
  "source_url": "https://facebook.com/post/123",
  "author_handle": "user123",
  "content_type": "post"
}
```

### Utility Endpoints

- `GET /health` - Service health check  
- `GET /agents` - List available agents
- `GET /status` - Detailed service status

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐
│   FastAPI App   │    │   Agent Tools   │
│                 │───▶│                 │
│ • REST API      │    │ • CoordinatorAgent │
│ • JSON Response │    │ • DataEngAgent  │
│ • Health Check  │    │ • OsintAgent    │
│ • Auto Docs     │    │ • LexiconAgent  │
└─────────────────┘    │ • TrendAnalysisAgent │
                       └─────────────────┘
```

## 🎯 Usage Examples

### Python Client
```python
import aiohttp

async def analyze_content():
    async with aiohttp.ClientSession() as session:
        data = {
            "content": "Suspicious election post here...",
            "source_platform": "twitter"
        }
        async with session.post('http://localhost:8000/analyze', json=data) as resp:
            result = await resp.json()
            print(f"Analysis: {result}")
```

### cURL
```bash
# Analyze content
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "URGENT: Vote rigging detected!",
    "source_platform": "twitter",
    "request_type": "narrative_classification"
  }'

# Submit finding
curl -X POST "http://localhost:8000/submit_finding" \
  -H "Content-Type: application/json" \
  -d '{
    "content_text": "Breaking election news...",
    "source_platform": "facebook"
  }'

# Check health
curl http://localhost:8000/health
```

## 🔧 Configuration

### Environment Variables
- `PORT`: Service port (default: 8000)
- `GOOGLE_CLOUD_PROJECT`: GCP project for ADK (optional)
- `GOOGLE_CLOUD_LOCATION`: GCP location (optional)
- `MONGODB_ATLAS_URI`: MongoDB connection (for data storage)
- `NEO4J_URI`: Neo4j connection (for graph operations)

### ADK Integration
The service automatically detects Google ADK availability:
- **With ADK**: Uses full Google Agent Engine integration
- **Without ADK**: Falls back to enhanced coordinator system

## 📊 Features

✅ **REST API** - Standard HTTP endpoints for integration  
✅ **Health Monitoring** - Service health and component status  
✅ **Multi-Agent System** - Access to 5 specialized agents  
✅ **Content Analysis** - Process election-related content  
✅ **Finding Submission** - Accept new findings for analysis  
✅ **CORS Enabled** - Ready for frontend integration  
✅ **OpenAPI Docs** - Auto-generated API documentation  
✅ **Production Ready** - Simple deployment and scaling  

## 🚀 Deployment

### Local Development
```bash
cd app/
python simple_agent_api.py
```

### Production
```bash
# Using gunicorn
cd app/
pip install gunicorn
gunicorn simple_agent_api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# Using Docker
cd app/
docker build -f Dockerfile -t electionwatch-api ..
docker run -p 8000:8000 electionwatch-api
```

### Cloud Deployment
The service is ready for deployment to:
- Google Cloud Run
- AWS Lambda (with Mangum adapter)
- Azure Container Instances
- Kubernetes

## 🔍 Troubleshooting

### Common Issues

1. **"No model found for CoordinatorAgent"**
   - Ensure model parameter is set in agent definitions
   - Check Google Cloud authentication if using ADK

2. **Import errors**
   - Install dependencies: `pip install -r requirements_api.txt`
   - Check Python path configuration

3. **Database connection errors**
   - Set proper environment variables for MongoDB/Neo4j
   - Tools will work in mock mode without real databases

### Logs
The service provides detailed logging:
```bash
cd app/
python simple_agent_api.py
# Look for:
# 🚀 Starting ElectionWatch Simple Agent API...
# ✅ Successfully imported ElectionWatch agents
# 🔧 Agent Status: Available
```

## 📚 Next Steps

1. **Frontend Integration**: Connect your web/mobile app to these endpoints
2. **Authentication**: Add authentication middleware for production
3. **Rate Limiting**: Implement rate limiting for API endpoints  
4. **Monitoring**: Add metrics and monitoring (Prometheus, etc.)
5. **Scaling**: Configure load balancing and auto-scaling
6. **Database**: Set up persistent storage for analysis results

## 🤝 Integration Examples

### React Frontend
```javascript
// Submit content for analysis
const analyzeContent = async (content) => {
  const response = await fetch('http://localhost:8000/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      content,
      source_platform: 'web',
      request_type: 'auto_detect'
    })
  });
  return response.json();
};
```

### Mobile App (React Native)
```javascript
// Submit finding from mobile
const submitFinding = async (text, platform) => {
  try {
    const response = await fetch('http://your-api.com/submit_finding', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        content_text: text,
        source_platform: platform
      })
    });
    return await response.json();
  } catch (error) {
    console.error('Submission failed:', error);
  }
};
```

This API service provides a complete solution for running your ElectionWatch multi-agent system as a production-ready web application! 