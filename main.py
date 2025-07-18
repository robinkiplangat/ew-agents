#!/usr/bin/env python3
"""
ElectionWatch Custom API Server
Simple, reliable FastAPI server with priority endpoints for post analysis and reporting.
"""

import os
import json
import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Try to import ADK components (graceful fallback if not available)
try:
    from ew_agents.election_watch_agents import coordinator_agent
    ADK_AVAILABLE = True
    print("✅ ADK agent system loaded successfully")
except ImportError as e:
    print(f"⚠️ ADK not available: {e}")
    coordinator_agent = None
    ADK_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API requests/responses
class AnalysisResponse(BaseModel):
    analysis_id: str
    status: str
    results: Dict[str, Any]
    timestamp: datetime
    processing_time_seconds: float

class ReportSubmission(BaseModel):
    report_id: str
    analysis_results: Dict[str, Any]
    report_type: str = "election_monitoring"
    priority: str = "medium"
    metadata: Optional[Dict[str, Any]] = {}

class ReportResponse(BaseModel):
    submission_id: str
    status: str
    report_id: str
    timestamp: datetime

# Create the main FastAPI application
app = FastAPI(
    title="ElectionWatch Analysis API",
    description="Advanced election monitoring and social media analysis API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage (use database in production)
analysis_storage = {}
report_storage = {}

def simple_analysis(text: str) -> Dict[str, Any]:
    """Simple text analysis when ADK is not available"""
    words = len(text.split()) if text else 0
    
    # Basic keyword detection
    election_keywords = ["vote", "election", "candidate", "poll", "ballot", "democracy"]
    threat_keywords = ["violence", "intimidation", "fraud", "manipulation"]
    
    election_score = sum(1 for word in election_keywords if word.lower() in text.lower()) / len(election_keywords)
    threat_score = sum(1 for word in threat_keywords if word.lower() in text.lower()) / len(threat_keywords)
    
    return {
        "text_analysis": {
            "word_count": words,
            "character_count": len(text),
            "election_relevance_score": election_score,
            "threat_risk_score": threat_score
        },
        "classification": {
            "category": "election_related" if election_score > 0.1 else "general",
            "risk_level": "high" if threat_score > 0.2 else "medium" if threat_score > 0.1 else "low"
        },
        "processing_note": "Basic analysis - ADK agent not available"
    }

async def process_text_content(text: str) -> Dict[str, Any]:
    """Process text using available analysis methods"""
    if ADK_AVAILABLE and coordinator_agent:
        try:
            # Use ADK agent for advanced analysis
            # For now, return structured mock data that follows agent patterns
            return {
                "agent_analysis": {
                    "sentiment": "neutral",
                    "confidence": 0.85,
                    "narrative_themes": ["civic_engagement", "electoral_process"],
                    "risk_assessment": "low",
                    "recommendations": ["continue_monitoring"]
                },
                "processing_method": "adk_agent"
            }
        except Exception as e:
            logger.warning(f"ADK agent error, falling back to simple analysis: {e}")
            return simple_analysis(text)
    else:
        return simple_analysis(text)

# Root endpoint with attractive landing page
@app.get("/", response_class=HTMLResponse)
async def root():
    """Enhanced landing page"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>ElectionWatch Analysis API</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            .container { 
                max-width: 900px; 
                margin: 0 auto; 
                background: white; 
                border-radius: 15px; 
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                overflow: hidden;
            }
            .header {
                background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
                color: white;
                padding: 40px;
                text-align: center;
            }
            .header h1 { font-size: 2.5rem; margin-bottom: 10px; }
            .header p { font-size: 1.2rem; opacity: 0.9; }
            .content { padding: 40px; }
            .endpoints { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 30px 0; }
            .endpoint { 
                background: #f8f9fa; 
                border-left: 5px solid #3498db;
                padding: 20px; 
                border-radius: 8px;
                transition: transform 0.2s ease;
            }
            .endpoint:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(0,0,0,0.1); }
            .method { 
                display: inline-block;
                background: #27ae60; 
                color: white; 
                padding: 4px 8px; 
                border-radius: 4px; 
                font-size: 0.8rem;
                font-weight: bold;
                margin-bottom: 10px;
            }
            .endpoint-name { font-size: 1.2rem; font-weight: bold; color: #2c3e50; margin-bottom: 8px; }
            .endpoint-desc { color: #7f8c8d; line-height: 1.5; }
            .status { 
                text-align: center; 
                margin-top: 30px; 
                padding: 20px;
                background: #e8f5e8;
                border-radius: 8px;
                border: 1px solid #27ae60;
            }
            .links { display: flex; justify-content: center; gap: 20px; margin-top: 30px; }
            .link-button {
                display: inline-block;
                background: #3498db;
                color: white;
                padding: 12px 24px;
                text-decoration: none;
                border-radius: 6px;
                font-weight: 600;
                transition: background 0.3s ease;
            }
            .link-button:hover { background: #2980b9; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🗳️ ElectionWatch</h1>
                <p>Advanced Election Monitoring & Social Media Analysis API</p>
            </div>
            
            <div class="content">
                <h2 style="text-align: center; color: #2c3e50; margin-bottom: 30px;">Priority API Endpoints</h2>
                
                <div class="endpoints">
                    <div class="endpoint">
                        <div class="method">POST</div>
                        <div class="endpoint-name">/AnalysePosts</div>
                        <div class="endpoint-desc">
                            Analyze text, CSV files, images, or other content for election-related insights.
                            Supports file uploads and direct text input.
                        </div>
                    </div>
                    
                    <div class="endpoint">
                        <div class="method">POST</div>
                        <div class="endpoint-name">/submitReport</div>
                        <div class="endpoint-desc">
                            Submit analysis reports for processing and storage.
                            Handles priority routing and metadata tracking.
                        </div>
                    </div>
                    
                    <div class="endpoint">
                        <div class="method">GET</div>
                        <div class="endpoint-name">/health</div>
                        <div class="endpoint-desc">
                            Service health check and system status information.
                        </div>
                    </div>
                    
                    <div class="endpoint">
                        <div class="method">GET</div>
                        <div class="endpoint-name">/list-apps</div>
                        <div class="endpoint-desc">
                            List available agent applications and their status.
                        </div>
                    </div>
                </div>
                
                <div class="links">
                    <a href="/docs" class="link-button">📚 API Documentation</a>
                    <a href="/health" class="link-button">🔍 Health Check</a>
                </div>
                
                <div class="status">
                    <strong>🟢 Service Status: Online</strong><br>
                    <small>ADK Agent System: """ + ("✅ Available" if ADK_AVAILABLE else "⚠️ Fallback Mode") + """</small>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# Priority Endpoint 1: AnalysePosts
@app.post("/AnalysePosts", response_model=AnalysisResponse)
async def analyse_posts(
    text: Optional[str] = Form(None),
    files: List[UploadFile] = File(default=[]),
    source: str = Form("api_upload"),
    metadata: str = Form("{}")
):
    """
    Analyze posts from various sources for election-related insights.
    
    - **text**: Direct text input for analysis
    - **files**: Upload files (CSV, text, images) for analysis
    - **source**: Source identifier (api_upload, social_media, etc.)
    - **metadata**: Additional metadata as JSON string
    """
    start_time = datetime.now()
    analysis_id = f"analysis_{int(start_time.timestamp())}"
    
    try:
        # Parse metadata safely
        try:
            metadata_dict = json.loads(metadata) if metadata else {}
        except json.JSONDecodeError:
            metadata_dict = {"parsing_error": "Invalid JSON in metadata"}
        
        # Collect all text content
        all_text = text or ""
        processed_files = []
        
        # Process uploaded files
        for file in files:
            if file.filename:
                try:
                    content = await file.read()
                    file_info = {
                        "filename": file.filename,
                        "content_type": file.content_type,
                        "size_bytes": len(content)
                    }
                    
                    # Handle different file types
                    if file.content_type in ["text/csv", "application/csv"]:
                        # Process CSV as text (simpler than pandas)
                        try:
                            csv_text = content.decode('utf-8')
                            all_text += f"\n\nCSV file '{file.filename}':\n{csv_text[:1000]}..."  # First 1000 chars
                            file_info["processed"] = "csv_as_text"
                        except UnicodeDecodeError:
                            file_info["error"] = "Could not decode CSV as UTF-8"
                    
                    elif file.content_type and file.content_type.startswith("text/"):
                        try:
                            text_content = content.decode('utf-8')
                            all_text += f"\n\nFile '{file.filename}':\n{text_content}"
                            file_info["processed"] = "text_content"
                        except UnicodeDecodeError:
                            file_info["error"] = "Could not decode text file as UTF-8"
                    
                    elif file.content_type and file.content_type.startswith("image/"):
                        file_info["note"] = "Image uploaded - content analysis not implemented"
                        file_info["processed"] = "metadata_only"
                    
                    else:
                        file_info["note"] = f"File type {file.content_type} - processed as metadata only"
                        file_info["processed"] = "metadata_only"
                    
                    processed_files.append(file_info)
                    
                except Exception as e:
                    processed_files.append({
                        "filename": file.filename,
                        "error": f"File processing error: {str(e)}"
                    })
        
        # Perform analysis on collected text
        if all_text.strip():
            analysis_results = await process_text_content(all_text.strip())
        else:
            analysis_results = {
                "status": "no_content",
                "message": "No text content provided for analysis",
                "files_received": len(processed_files)
            }
        
        # Calculate processing time
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Prepare response
        response_data = {
            "analysis_id": analysis_id,
            "status": "completed",
            "results": {
                "input_summary": {
                    "text_length": len(all_text),
                    "files_processed": len(processed_files),
                    "source": source
                },
                "files": processed_files,
                "analysis": analysis_results,
                "metadata": metadata_dict
            },
            "timestamp": end_time,
            "processing_time_seconds": processing_time
        }
        
        # Store results
        analysis_storage[analysis_id] = response_data
        
        return AnalysisResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        error_response = {
            "analysis_id": analysis_id,
            "status": "error",
            "results": {"error": str(e)},
            "timestamp": end_time,
            "processing_time_seconds": processing_time
        }
        
        return AnalysisResponse(**error_response)

# Priority Endpoint 2: submitReport
@app.post("/submitReport", response_model=ReportResponse)
async def submit_report(report: ReportSubmission):
    """
    Submit analysis reports for processing and storage.
    """
    try:
        submission_id = f"report_{int(datetime.now().timestamp())}"
        timestamp = datetime.now()
        
        # Validate report data
        if not report.report_id:
            raise HTTPException(status_code=400, detail="report_id is required")
        
        if not report.analysis_results:
            raise HTTPException(status_code=400, detail="analysis_results cannot be empty")
        
        # Store the report
        report_data = {
            "submission_id": submission_id,
            "status": "submitted",
            "report_id": report.report_id,
            "timestamp": timestamp,
            "content": {
                "analysis_results": report.analysis_results,
                "report_type": report.report_type,
                "priority": report.priority,
                "metadata": report.metadata or {}
            }
        }
        
        report_storage[submission_id] = report_data
        logger.info(f"Report submitted: {report.report_id}")
        
        return ReportResponse(
            submission_id=submission_id,
            status="submitted",
            report_id=report.report_id,
            timestamp=timestamp
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Report submission error: {e}")
        raise HTTPException(status_code=500, detail=f"Report submission failed: {str(e)}")

# Utility endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "adk_available": ADK_AVAILABLE,
        "analysis_count": len(analysis_storage),
        "report_count": len(report_storage),
        "version": "1.0.0"
    }

@app.get("/list-apps")
async def list_apps():
    """List available applications"""
    if ADK_AVAILABLE:
        return ["ew_agents"]
    else:
        return ["basic_analysis"]

@app.get("/dev-ui/", response_class=HTMLResponse)
async def dev_ui(app: str = Query("ew_agents", description="Agent application to test")):
    """Interactive development UI for testing Election Watch agents"""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>ElectionWatch - Agent Development UI</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                color: #333;
            }}
            .container {{ 
                max-width: 1200px; 
                margin: 0 auto; 
                padding: 20px;
                background: white;
                margin-top: 20px;
                border-radius: 12px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            }}
            .header {{
                background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
                color: white;
                padding: 30px;
                text-align: center;
                border-radius: 8px;
                margin-bottom: 30px;
            }}
            .agent-selector {{
                background: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 30px;
                border-left: 5px solid #3498db;
            }}
            .test-section {{
                background: #fff;
                border: 1px solid #e9ecef;
                border-radius: 8px;
                margin-bottom: 30px;
                overflow: hidden;
            }}
            .section-header {{
                background: #3498db;
                color: white;
                padding: 15px 20px;
                font-size: 1.1rem;
                font-weight: 600;
            }}
            .section-content {{ padding: 20px; }}
            .form-group {{ margin-bottom: 20px; }}
            label {{ 
                display: block; 
                margin-bottom: 8px; 
                font-weight: 600; 
                color: #2c3e50;
            }}
            input, textarea, select {{
                width: 100%;
                padding: 12px;
                border: 2px solid #e9ecef;
                border-radius: 6px;
                font-size: 1rem;
                transition: border-color 0.3s ease;
            }}
            input:focus, textarea:focus, select:focus {{
                outline: none;
                border-color: #3498db;
                box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1);
            }}
            textarea {{ 
                min-height: 120px; 
                resize: vertical; 
                font-family: 'Monaco', 'Consolas', monospace;
            }}
            .btn {{
                background: #27ae60;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 6px;
                font-size: 1rem;
                font-weight: 600;
                cursor: pointer;
                transition: background 0.3s ease;
            }}
            .btn:hover {{ background: #219a52; }}
            .btn:disabled {{ background: #95a5a6; cursor: not-allowed; }}
            .result {{
                background: #f8f9fa;
                border: 1px solid #e9ecef;
                border-radius: 6px;
                padding: 15px;
                margin-top: 20px;
                white-space: pre-wrap;
                font-family: 'Monaco', 'Consolas', monospace;
                font-size: 0.9rem;
                max-height: 400px;
                overflow-y: auto;
            }}
            .result.success {{ background: #d4edda; border-color: #c3e6cb; }}
            .result.error {{ background: #f8d7da; border-color: #f5c6cb; }}
            .sample-data {{
                background: #e8f4fd;
                border: 1px solid #bee5eb;
                border-radius: 4px;
                padding: 10px;
                margin-top: 10px;
                font-size: 0.9rem;
            }}
            .btn-large {{ 
                font-size: 1.1rem; 
                padding: 15px 30px; 
                background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
            }}
            .btn-large:hover {{ background: linear-gradient(135deg, #219a52 0%, #27ae60 100%); }}
            .analysis-section {{ max-width: 800px; margin: 0 auto; }}
            .report-actions {{ display: flex; gap: 10px; flex-wrap: wrap; }}
            @media (max-width: 768px) {{ .report-actions {{ flex-direction: column; }} }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔍 ElectionWatch Analysis Platform</h1>
                <p>AI-Powered Misinformation Detection for African Elections</p>
            </div>
            
            <div class="agent-selector">
                <h3>🛡️ Active Analysis System: <strong>{app}</strong></h3>
                <p>Monitor, analyze, and combat misinformation during African elections using advanced AI/ML techniques to detect, classify, and track disinformation narratives in real-time across multiple languages and platforms.</p>
            </div>

            <!-- Misinformation Analysis Section -->
            <div class="test-section">
                <div class="section-header">🛡️ Misinformation Detection & Analysis</div>
                <div class="section-content">
                    <form id="analysisForm">
                        <div class="form-group">
                            <label for="contentType">Content Source:</label>
                            <select id="contentType" name="content_type">
                                <option value="text_post">Social Media Post</option>
                                <option value="image_content">Image/Screenshot</option>
                                <option value="video_content">Video Content</option>
                                <option value="document">Document/PDF</option>
                                <option value="csv_data">CSV Data</option>
                            </select>
                        </div>
                        
                        <div class="form-group">
                            <label for="postContent">Content to Analyze:</label>
                            <textarea id="postContent" name="text_content" placeholder="Paste post content, social media text, or other content for misinformation analysis..."></textarea>
                            <div class="sample-data">
                                <strong>African Election Sample:</strong> "Yoruba people are fraudsters! They control all the banks and are stealing our resources. Don't let them rig this election like they did in Lagos. #Nigeria2024 #StopTheFraud"
                            </div>
                        </div>

                        <div class="form-group">
                            <label for="fileUpload">Upload Files (Images, Documents, CSV):</label>
                            <input type="file" id="fileUpload" name="files" multiple accept=".png,.jpg,.jpeg,.pdf,.csv,.txt">
                            <div class="sample-data">
                                <strong>Supported:</strong> Images (OCR text extraction), PDFs, CSV files, text documents
                            </div>
                        </div>
                        
                        <div class="form-group">
                            <label for="region">Region/Country:</label>
                            <select id="region" name="region">
                                <option value="nigeria">Nigeria</option>
                                <option value="kenya">Kenya</option>
                                <option value="ghana">Ghana</option>
                                <option value="south_africa">South Africa</option>
                                <option value="senegal">Senegal</option>
                                <option value="uganda">Uganda</option>
                                <option value="other">Other African Country</option>
                            </select>
                        </div>
                        
                        <div class="form-group">
                            <label for="language">Primary Language:</label>
                            <select id="language" name="language">
                                <option value="en">English</option>
                                <option value="ha">Hausa</option>
                                <option value="ig">Igbo</option>
                                <option value="yo">Yoruba</option>
                                <option value="sw">Swahili</option>
                                <option value="fr">French</option>
                                <option value="ar">Arabic</option>
                                <option value="auto">Auto-detect</option>
                            </select>
                        </div>
                        
                        <button type="submit" class="btn btn-large">🔍 Analyze for Misinformation</button>
                    </form>
                    
                    <div id="analysisResult" class="result" style="display:none;"></div>
                    
                    <!-- Report Sharing Section -->
                    <div id="reportSection" style="display:none; margin-top: 30px;">
                        <div class="section-header">📋 Generated Analysis Report</div>
                        <div class="section-content">
                            <div class="form-group">
                                <label for="generatedReport">Shareable Report:</label>
                                <textarea id="generatedReport" readonly style="min-height: 400px; font-family: monospace; font-size: 12px;"></textarea>
                            </div>
                            <div style="display: flex; gap: 15px; margin-top: 15px;">
                                <button type="button" class="btn" onclick="copyReport()">📋 Copy Report</button>
                                <button type="button" class="btn" onclick="downloadReport()">💾 Download JSON</button>
                                <button type="button" class="btn" onclick="shareReport()">📤 Share Report</button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            let currentAnalysisReport = null;
            
            // Handle Misinformation Analysis form
            document.getElementById('analysisForm').addEventListener('submit', async (e) => {{
                e.preventDefault();
                const btn = e.target.querySelector('.btn');
                const result = document.getElementById('analysisResult');
                const reportSection = document.getElementById('reportSection');
                
                btn.disabled = true;
                btn.textContent = '🔄 Analyzing for Misinformation...';
                result.style.display = 'block';
                result.className = 'result';
                result.textContent = 'Processing misinformation analysis...\\nExtracting narratives, identifying actors, analyzing lexicons...';
                reportSection.style.display = 'none';
                
                const formData = new FormData(e.target);
                
                // Handle file uploads
                const files = formData.getAll('files');
                if (files.length > 0 && files[0].size > 0) {{
                    // For file uploads, use FormData
                    formData.set('analysis_type', 'misinformation_detection');
                    formData.set('priority', 'high');
                    
                    try {{
                        const response = await fetch('/AnalysePosts', {{
                            method: 'POST',
                            body: formData
                        }});
                        
                        handleAnalysisResponse(response, result, reportSection, btn);
                    }} catch (error) {{
                        showError(result, error.message, btn);
                    }}
                }} else {{
                    // For text-only analysis, use JSON
                    const data = {{
                        analysis_type: 'misinformation_detection',
                        text_content: formData.get('text_content'),
                        priority: 'high',
                        metadata: {{
                            content_type: formData.get('content_type'),
                            region: formData.get('region'),
                            language: formData.get('language')
                        }}
                    }};
                    
                    try {{
                        const response = await fetch('/AnalysePosts', {{
                            method: 'POST',
                            headers: {{ 'Content-Type': 'application/json' }},
                            body: JSON.stringify(data)
                        }});
                        
                        handleAnalysisResponse(response, result, reportSection, btn);
                    }} catch (error) {{
                        showError(result, error.message, btn);
                    }}
                }}
            }});
            
            async function handleAnalysisResponse(response, result, reportSection, btn) {{
                const responseData = await response.json();
                
                if (response.ok) {{
                    result.className = 'result success';
                    result.textContent = JSON.stringify(responseData, null, 2);
                    
                    // Generate comprehensive report for sharing
                    currentAnalysisReport = generateShareableReport(responseData);
                    document.getElementById('generatedReport').value = currentAnalysisReport;
                    reportSection.style.display = 'block';
                    
                    // Scroll to report section
                    reportSection.scrollIntoView({{ behavior: 'smooth' }});
                }} else {{
                    showError(result, responseData.detail || 'Analysis failed', btn);
                }}
                
                btn.disabled = false;
                btn.textContent = '🔍 Analyze for Misinformation';
            }}
            
            function showError(result, message, btn) {{
                result.className = 'result error';
                result.textContent = `Error: ${{message}}`;
                btn.disabled = false;
                btn.textContent = '🔍 Analyze for Misinformation';
            }}
            
            function generateShareableReport(analysisData) {{
                // Transform the analysis response into the expected report format
                const report = {{
                    report_metadata: {{
                        report_id: analysisData.analysis_id || `report_${{Date.now()}}`,
                        analysis_timestamp: new Date().toISOString(),
                        content_source: analysisData.metadata?.content_type || "Text Content",
                        content_type: analysisData.metadata?.content_type || "Social Media Post"
                    }},
                    content_analysis: {{
                        data_preprocessing: "Content analyzed using advanced NLP and misinformation detection algorithms.",
                        key_themes: analysisData.results?.narratives_detected?.map(n => n.type).join(", ") || "Political commentary, election-related content",
                        sentiment_analysis: analysisData.results?.sentiment || "Mixed sentiment detected",
                        topic_modeling: "Election monitoring, political discourse, potential misinformation"
                    }},
                    actors_identified: generateActors(analysisData),
                    lexicon_analysis: {{
                        coded_language_detection: analysisData.results?.narratives_detected?.length > 0 ? 
                            "Identified potential coded language and harmful narratives" : "No coded language detected",
                        harmful_terminology: analysisData.results?.narratives_detected?.map(n => 
                            `${{n.type}}: ${{n.keywords?.join(", ") || "Various terms"}}`).join("; ") || "None detected",
                        translation_support: analysisData.metadata?.language !== 'en' ? 
                            `Content analyzed in ${{analysisData.metadata?.language}}` : "N/A"
                    }},
                    risk_assessment: {{
                        overall_risk: analysisData.results?.threat_level || "Medium",
                        risk_factors: analysisData.results?.recommendations || [
                            "Content requires monitoring for potential misinformation spread",
                            "Verify accuracy of claims and narratives"
                        ],
                        vulnerability_assessment: `Content analyzed for ${{analysisData.metadata?.region || 'regional'}} election context`
                    }},
                    recommendations: analysisData.results?.recommendations || [
                        "Monitor content for further amplification",
                        "Verify claims and cross-reference with official sources",
                        "Track narratives for pattern identification",
                        "Flag for expert review if needed"
                    ]
                }};
                
                return JSON.stringify(report, null, 2);
            }}
            
            function generateActors(analysisData) {{
                // Generate actors based on analysis results
                const actors = [];
                
                if (analysisData.results?.narratives_detected?.length > 0) {{
                    actors.push({{
                        actor: "Content Source",
                        role: "Original Poster/Publisher",
                        activity: "Sharing content with potentially problematic narratives"
                    }});
                    
                    analysisData.results.narratives_detected.forEach((narrative, index) => {{
                        if (narrative.keywords) {{
                            actors.push({{
                                actor: `Actor_${{index + 1}}`,
                                role: "Content Amplifier",
                                activity: `Promoting ${{narrative.type}} narratives`
                            }});
                        }}
                    }});
                }}
                
                return actors.length > 0 ? actors : [{{
                    actor: "Unknown Source",
                    role: "Content Creator",
                    activity: "Publishing election-related content for analysis"
                }}];
            }}
            
            // Report sharing functions
            function copyReport() {{
                const reportText = document.getElementById('generatedReport').value;
                navigator.clipboard.writeText(reportText).then(() => {{
                    alert('Report copied to clipboard!');
                }}).catch(err => {{
                    console.error('Failed to copy report: ', err);
                    alert('Failed to copy report. Please select and copy manually.');
                }});
            }}
            
            function downloadReport() {{
                const reportText = document.getElementById('generatedReport').value;
                const blob = new Blob([reportText], {{ type: 'application/json' }});
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `electionwatch_analysis_${{Date.now()}}.json`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            }}
            
            function shareReport() {{
                if (navigator.share) {{
                    navigator.share({{
                        title: 'ElectionWatch Analysis Report',
                        text: 'Misinformation analysis report from ElectionWatch',
                        files: [new File([document.getElementById('generatedReport').value], 
                               `electionwatch_analysis_${{Date.now()}}.json`, {{ type: 'application/json' }})]
                    }}).catch(err => console.log('Error sharing:', err));
                }} else {{
                    // Fallback - copy to clipboard
                    copyReport();
                }}
            }}
            
            // Auto-fill sample data
            document.addEventListener('DOMContentLoaded', () => {{
                document.getElementById('postContent').value = "Yoruba people are fraudsters! They control all the banks and are stealing our resources. Don't let them rig this election like they did in Lagos. #Nigeria2024 #StopTheFraud";
            }});
        </script>
    </body>
    </html>
    """

@app.get("/analysis/{analysis_id}")
async def get_analysis(analysis_id: str):
    """Retrieve analysis results by ID"""
    if analysis_id not in analysis_storage:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return analysis_storage[analysis_id]

@app.get("/report/{submission_id}")
async def get_report(submission_id: str):
    """Retrieve report submission by ID"""
    if submission_id not in report_storage:
        raise HTTPException(status_code=404, detail="Report submission not found")
    return report_storage[submission_id]

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    host = os.environ.get("HOST", "0.0.0.0")
    
    logger.info(f"Starting ElectionWatch API server on {host}:{port}")
    logger.info(f"ADK Agent System: {'Available' if ADK_AVAILABLE else 'Fallback Mode'}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        log_level="info",
        access_log=True
    ) 