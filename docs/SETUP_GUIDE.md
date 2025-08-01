# ElectionWatch Setup Guide

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- MongoDB Atlas account (or local MongoDB)
- OpenRouter API key (for AI report generation)
- Google Cloud account (for production deployment)

### 1. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd ml

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

```bash
# Copy the example environment file
cp env.example.txt .env

# Edit the .env file with your actual values
nano .env  # or use your preferred editor
```

#### Required Environment Variables

**For Basic Functionality:**
- `MONGODB_ATLAS_URI` - Your MongoDB Atlas connection string
- `OPEN_ROUTER_API_KEY` - OpenRouter API key for AI report generation
- `GOOGLE_CLOUD_PROJECT` - Your Google Cloud project ID

**For Full Features:**
- `GOOGLE_API_KEY` - Google API key for various services
- `HUGGING_FACE_HUB_TOKEN` - Hugging Face token for model access
- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` - Neo4j database credentials

### 3. Database Setup

#### MongoDB Atlas (Recommended)
1. Create a MongoDB Atlas cluster
2. Get your connection string
3. Add it to `MONGODB_ATLAS_URI` in your `.env` file
4. Whitelist your IP address in Atlas

#### Local MongoDB (Alternative)
1. Install MongoDB locally
2. Set `MONGODB_URI=mongodb://localhost:27017/election_watch`
3. Set `MONGODB_DEVELOPMENT_MODE=true`

### 4. API Keys Setup

#### OpenRouter API Key (Required for Reports)
1. Sign up at [OpenRouter](https://openrouter.ai/)
2. Get your API key
3. Add to `OPEN_ROUTER_API_KEY` in `.env`

#### Google Cloud Setup (For Production)
1. Create a Google Cloud project
2. Enable required APIs:
   - Cloud Run API
   - Secret Manager API
   - Cloud Build API
3. Create a service account with necessary permissions
4. Set `GOOGLE_CLOUD_PROJECT` in `.env`

### 5. Run the Application

```bash
# Start the FastAPI server
python3 main.py

# The application will be available at:
# http://localhost:8080
```

### 6. Verify Installation

Test the endpoints:
```bash
# Health check
curl http://localhost:8080/health

# MongoDB connection
curl http://localhost:8080/storage/stats

# Reports system
curl http://localhost:8080/api/reports/available
```

## 🌐 Production Deployment

### Google Cloud Run Deployment

```bash
# Make deploy script executable
chmod +x deploy.sh

# Run deployment
./deploy.sh
```

The deployment script will:
1. Build and push the Docker container
2. Set up Google Cloud Secrets Manager
3. Deploy to Cloud Run
4. Configure environment variables

### Environment Variables in Production

For production, use Google Cloud Secret Manager instead of `.env` files:

```bash
# Store secrets in Secret Manager
gcloud secrets create mongodb-atlas-uri --data-file=<(echo -n "your_mongodb_uri")
gcloud secrets create open-router-api-key --data-file=<(echo -n "your_api_key")

# Grant access to Cloud Run service account
gcloud secrets add-iam-policy-binding mongodb-atlas-uri \
    --member="serviceAccount:ew-agent-service@your-project.iam.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
```

## 🔧 Configuration Options

### Feature Flags

Control which features are enabled:

```bash
# Enable/disable reports system
REPORTS_SYSTEM_ENABLED=true

# Enable/disable AI report generation
AI_REPORT_GENERATION_ENABLED=true

# Enable/disable Cloud Run optimizations
CLOUD_RUN_MODE=false
```

### Performance Tuning

```bash
# Database connection pool
DB_POOL_SIZE=5

# Request timeout
REQUEST_TIMEOUT=30

# Maximum concurrent requests
MAX_CONCURRENT_REQUESTS=10
```

## 📊 Monitoring and Logging

### Log Levels
```bash
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR
```

### Health Checks
- `/health` - Application health status
- `/storage/stats` - Database connection status
- `/debug/env-check` - Environment variables status

## 🔒 Security Considerations

1. **Never commit `.env` files** to version control
2. **Use Secret Manager** in production
3. **Rotate API keys** regularly
4. **Enable CORS** only for trusted domains
5. **Use HTTPS** in production

## 🐛 Troubleshooting

### Common Issues

**MongoDB Connection Failed:**
- Check your connection string
- Verify IP whitelist in Atlas
- Set `MONGODB_DEVELOPMENT_MODE=true` for local testing

**LLM API Errors:**
- Verify your OpenRouter API key
- Check API quota limits
- Ensure `OPEN_ROUTER_API_KEY` is set

**Cloud Run Deployment Issues:**
- Verify Google Cloud project permissions
- Check service account roles
- Ensure all required APIs are enabled

### Debug Mode

Enable debug mode for detailed logging:
```bash
DEBUG=true
LOG_LEVEL=DEBUG
```

## 📚 Additional Resources

- [API Documentation](docs/API_GUIDE.md)
- [MongoDB Atlas Setup](https://docs.atlas.mongodb.com/)
- [Google Cloud Run Documentation](https://cloud.google.com/run/docs)
- [OpenRouter API Documentation](https://openrouter.ai/docs)

## 🤝 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the API documentation
3. Check application logs
4. Verify environment configuration 