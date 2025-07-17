
# 🚀 ElectionWatch API Application

This folder contains the FastAPI application and all deployment configurations for the ElectionWatch multi-agent system.

## 📁 **Folder Structure**

```
app/
├── simple_agent_api.py      # Main FastAPI application
├── test_simple_api.py       # API tests
├── client_example.py        # Usage examples
├── requirements_api.txt     # Python dependencies
├── Dockerfile              # Container definition
├── docker-compose.yml      # Multi-service setup
├── nginx.conf              # Reverse proxy config
├── Procfile               # Heroku deployment
├── cloudrun-service.yaml  # Google Cloud Run config
├── deploy.sh              # Deployment script
├── env.template           # Environment variables template
└── README.md              # This file
```

## 🏃 **Quick Start**

### 1. **Local Development**
```bash
# From the app/ directory
cd app/

# Install dependencies
pip install -r requirements_api.txt

# Start the API
python simple_agent_api.py
```

The API will be available at:
- **API**: http://localhost:8000
- **Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### 2. **Testing**
```bash
# Run tests (make sure API is running first)
python test_simple_api.py
```

### 3. **Docker Development**
```bash
# From the app/ directory
cd app/

# Start with Docker Compose (includes databases)
docker-compose up --build

# Or just the API
docker build -f Dockerfile -t electionwatch-api ..
docker run -p 8000:8000 electionwatch-api
```

## 🌐 **Deployment Options**

### **Google Cloud Run** (Recommended)
```bash
# From the app/ directory
cd app/

# Make script executable
chmod +x deploy.sh

# Deploy
./deploy.sh YOUR-PROJECT-ID us-central1
```

### **Heroku**
```bash
# From the project root
git add .
git commit -m "Deploy to Heroku"
git push heroku main
```

### **Docker Production**
```bash
# From the app/ directory
cd app/

# Build for production
docker build -f Dockerfile -t electionwatch-api:prod ..

# Run in production mode
docker run -d \
  --name electionwatch-api \
  -p 8000:8000 \
  -e ENVIRONMENT=production \
  electionwatch-api:prod
```

## ⚙️ **Configuration**

### **Environment Variables**
Copy `env.template` to `.env` and configure:

```bash
# From the app/ directory
cp env.template .env
# Edit .env with your values
```

Required variables:
- `PORT=8000`
- `ENVIRONMENT=development|production`
- `LOG_LEVEL=INFO`

Optional (for enhanced features):
- `MONGODB_ATLAS_URI` - MongoDB connection
- `NEO4J_URI` - Neo4j connection  
- `GOOGLE_CLOUD_PROJECT` - GCP project ID

### **Database Setup**
The system works with or without databases. For full functionality:

1. **MongoDB** (data storage)
2. **Neo4j** (graph relationships)
3. **Redis** (caching - optional)

Use `docker-compose.yml` for local development databases.

## 🔧 **Development**

### **File Structure**
- API code is in the `app/` folder
- Agent definitions are in `../ew_agents/`
- Data and scripts are in `../data/` and `../scripts/`

### **Import Paths**
The API automatically adds the parent directory to the Python path to access `ew_agents`:

```python
# In simple_agent_api.py
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Can now import from ew_agents
from ew_agents.election_watch_agents import coordinator_agent
```

### **Running from Different Directories**

**From app/ directory:**
```bash
cd app/
python simple_agent_api.py
```

**From project root:**
```bash
python app/simple_agent_api.py
```

**With Docker:**
```bash
cd app/
docker-compose up
```

## 📊 **API Endpoints**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service information |
| `/health` | GET | Health check |
| `/agents` | GET | List available agents |
| `/analyze` | POST | Analyze content |
| `/submit_finding` | POST | Submit new findings |
| `/status` | GET | Detailed service status |
| `/docs` | GET | Interactive API documentation |

## 🧪 **Testing**

### **Manual Testing**
```bash
# Health check
curl http://localhost:8000/health

# Analyze content
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"content": "Test content", "source_platform": "twitter"}'
```

### **Automated Testing**
```bash
# Run the test suite
python test_simple_api.py
```

### **Load Testing**
```bash
# Install wrk
brew install wrk  # macOS

# Test load
wrk -t12 -c400 -d30s http://localhost:8000/health
```

## 🚨 **Troubleshooting**

### **Import Errors**
If you get import errors:
1. Ensure you're running from the `app/` directory
2. Check that `../ew_agents/` exists
3. Verify Python path configuration in `simple_agent_api.py`

### **Docker Issues**
1. **Build Context**: Always build from `app/` directory with `docker build -f Dockerfile .. `
2. **Permissions**: The Dockerfile creates an `appuser` for security
3. **Volumes**: Use `../` paths for accessing parent directories

### **Database Connections**
- MongoDB and Neo4j connections are optional
- System works in "mock mode" without databases
- Check connection strings in environment variables

## 📚 **Additional Documentation**

- **General Setup**: See `../QUICK_START.md`
- **Deployment**: See `../DEPLOYMENT_GUIDE.md`
- **API Details**: See `../API_SETUP.md`

## 🎯 **Next Steps**

1. **Start the API**: `python simple_agent_api.py`
2. **Test it**: `python test_simple_api.py`
3. **Deploy it**: `./deploy.sh YOUR-PROJECT-ID`
4. **Integrate it**: Use the `/analyze` and `/submit_finding` endpoints

Your ElectionWatch API is ready for development and deployment! 🎉 