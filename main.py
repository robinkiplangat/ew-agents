#!/usr/bin/env python3
"""
ElectionWatch Standard FastAPI + ADK Integration
==============================================================
"""

# Load environment variables first, before any other imports
from dotenv import load_dotenv
load_dotenv()

import os
import json
import asyncio
import uvicorn
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import sys # Added for sys.path.append

# Set up logging
logger = logging.getLogger(__name__)

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import aiohttp
import base64
from io import BytesIO
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors

from google.adk.cli.fast_api import get_fast_api_app

# Get the directory where main.py is located
AGENT_DIR = os.path.dirname(os.path.abspath(__file__))

# ===== REPORT GENERATION FUNCTIONS =====

async def format_report_with_ai(llm_response: str, analysis_data: Dict[str, Any]) -> str:
    """
    Format the LLM response into a clean, professional report using AI via OpenRouter.
    """
    try:
        # Try to get API key from environment variable first
        openrouter_api_key = os.getenv("OPEN_ROUTER_API_KEY")
        
        # If not found, try to get from Google Cloud Secrets Manager
        if not openrouter_api_key:
            try:
                from google.cloud import secretmanager
                client = secretmanager.SecretManagerServiceClient()
                name = f"projects/{os.getenv('GOOGLE_CLOUD_PROJECT', 'ew-agents-v02')}/secrets/open-router-api-key/versions/latest"
                response = client.access_secret_version(request={"name": name})
                openrouter_api_key = response.payload.data.decode("UTF-8")
                logger.info("Retrieved OPEN_ROUTER_API_KEY from Google Cloud Secrets Manager")
            except Exception as secret_error:
                logger.warning(f"Could not retrieve OPEN_ROUTER_API_KEY from Secrets Manager: {secret_error}")
        
        if not openrouter_api_key:
            logger.warning("OPEN_ROUTER_API_KEY not found, returning raw response")
            return llm_response
        
        # Extract key information from analysis data
        analysis_id = analysis_data.get("structured_report", {}).get("report_metadata", {}).get("report_id", "Unknown")
        analysis_type = analysis_data.get("metadata", {}).get("analysis_type", "election_analysis")
        risk_level = analysis_data.get("structured_report", {}).get("risk_level", "unknown")
        date_analyzed = analysis_data.get("structured_report", {}).get("date_analyzed", "Unknown")
        
        # Create a comprehensive prompt for professional report generation
        prompt = f"""
        You are a senior election security analyst tasked with creating a professional, executive-level report. 
        
        Transform the following raw analysis data into a clean, aesthetically appealing, and insightful report.
        
        
        **REPORT REQUIREMENTS:**
        1. Create a professional executive summary
        2. Extract and highlight key insights and findings
        3. Provide clear assessments with visual indicators
        4. Offer actionable recommendations
        5. Include relevant technical details in an accessible format
        6. Use professional language suitable for all stakeholders
        
        **ANALYSIS CONTEXT:**
        - Analysis ID: {analysis_id}
        - Analysis Type: {analysis_type}
        - Risk Level: {risk_level}
        - Date Analyzed: {date_analyzed}
        
        **RAW ANALYSIS DATA:**
        {llm_response}
        
                            **OUTPUT FORMAT:**
                    Return the report in clean HTML format with the following structure:
                    
                    <div class="report-container">
                        <div class="ai-notice">
                            🤖 AI-GENERATED REPORT - This report was generated using artificial intelligence
                        </div>
                        <div class="header">
                            <h1>ElectionWatch Security Analysis Report</h1>
                            <div class="metadata">
                                <p><strong>Report ID:</strong> {analysis_id}</p>
                                <p><strong>Analysis Date:</strong> {date_analyzed}</p>
                                <p><strong>Risk Level:</strong> <span class="risk-{risk_level}">{risk_level.upper()}</span></p>
                            </div>
                        </div>
                        
                        <div class="executive-summary">
                            <h2>EXECUTIVE SUMMARY</h2>
                            <!-- Concise overview of key findings -->
                        </div>
                        
                        <div class="key-findings">
                            <h2>KEY FINDINGS</h2>
                            <!-- Bullet points of main discoveries -->
                        </div>
                        
                        <div class="risk-assessment">
                            <h2>RISK ASSESSMENT</h2>
                            <!-- Detailed risk analysis with severity levels -->
                        </div>
                        
                        <div class="recommendations">
                            <h2>RECOMMENDATIONS</h2>
                            <!-- Actionable next steps -->
                        </div>
                        
                        <div class="technical-details">
                            <h2>TECHNICAL ANALYSIS</h2>
                            <!-- Detailed technical findings -->
                        </div>
                    </div>
        
        **STYLING GUIDELINES:**
        - Use professional color scheme (blues, grays, whites)
        - Include appropriate CSS classes for styling
        - Make risk levels visually distinct (red for high, yellow for medium, green for low)
        - Use bullet points and numbered lists for clarity
        - Ensure the report is scannable and easy to read
        - Include relevant icons or visual indicators where appropriate
        
        **CONTENT GUIDELINES:**
        - Extract the insights from the raw data
        - Present information in a logical, flowing manner
        - Use clear, professional language
        - Focus on actionable intelligence
        - Highlight critical security concerns
        - Provide context for technical findings
        
        Create a report that would be suitable for presentation to election officials, security teams, and government stakeholders.
        """
        
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {openrouter_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "google/gemini-2.5-flash",
                "messages": [
                    {
                        "role": "system", 
                        "content": "You are a senior OSINT and election security analyst with expertise in creating executive-level reports. You excel at transforming raw analysis data into clear, actionable intelligence reports that are both comprehensive and accessible to stakeholders."
                    },
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2,  # Lower temperature for more consistent formatting
                "max_tokens": 4000   # Increased token limit for comprehensive reports
            }
            
            async with session.post("https://openrouter.ai/api/v1/chat/completions", 
                                  headers=headers, json=payload) as response:
                
                if response.status == 200:
                    result = await response.json()
                    formatted_report = result["choices"][0]["message"]["content"]
                    
                    # Add CSS styling to make the report more visually appealing
                    css_styles = """
                    <style>
                    .ai-notice {
                        background: #e3f2fd;
                        border: 1px solid #2196f3;
                        border-radius: 5px;
                        padding: 10px;
                        margin-bottom: 20px;
                        text-align: center;
                        color: #1976d2;
                        font-weight: bold;
                    }
                    .report-container {
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        max-width: 1200px;
                        margin: 0 auto;
                        padding: 20px;
                        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                        min-height: 100vh;
                    }
                    .header {
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        padding: 30px;
                        border-radius: 10px;
                        margin-bottom: 30px;
                        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                    }
                    .header h1 {
                        margin: 0 0 20px 0;
                        font-size: 2.5em;
                        font-weight: 300;
                    }
                    .metadata {
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                        gap: 15px;
                        margin-top: 20px;
                    }
                    .metadata p {
                        margin: 5px 0;
                        font-size: 1.1em;
                    }
                    .risk-high { color: #dc3545; font-weight: bold; background: #ffe6e6; padding: 5px 10px; border-radius: 5px; }
                    .risk-medium { color: #ffc107; font-weight: bold; background: #fff3cd; padding: 5px 10px; border-radius: 5px; }
                    .risk-low { color: #28a745; font-weight: bold; background: #d4edda; padding: 5px 10px; border-radius: 5px; }
                    .risk-unknown { color: #6c757d; font-weight: bold; background: #f8f9fa; padding: 5px 10px; border-radius: 5px; }
                    
                    .executive-summary, .key-findings, .risk-assessment, .recommendations, .technical-details {
                        background: white;
                        padding: 25px;
                        margin-bottom: 25px;
                        border-radius: 10px;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
                        border-left: 5px solid #667eea;
                    }
                    
                    h2 {
                        color: #2c3e50;
                        border-bottom: 2px solid #ecf0f1;
                        padding-bottom: 10px;
                        margin-bottom: 20px;
                        font-size: 1.8em;
                    }
                    
                    ul, ol {
                        padding-left: 25px;
                    }
                    
                    li {
                        margin-bottom: 8px;
                        line-height: 1.6;
                    }
                    
                    .highlight {
                        background: #fff3cd;
                        padding: 15px;
                        border-radius: 5px;
                        border-left: 4px solid #ffc107;
                        margin: 15px 0;
                    }
                    
                    .critical {
                        background: #f8d7da;
                        padding: 15px;
                        border-radius: 5px;
                        border-left: 4px solid #dc3545;
                        margin: 15px 0;
                    }
                    
                    .success {
                        background: #d4edda;
                        padding: 15px;
                        border-radius: 5px;
                        border-left: 4px solid #28a745;
                        margin: 15px 0;
                    }
                    </style>
                    """
                    
                    # Clean up markdown code blocks from the LLM response
                    cleaned_report = formatted_report.replace("```html", "").replace("```", "")
                                
                    # Combine the CSS with the cleaned report
                    final_report = css_styles + cleaned_report
                    return final_report
                    
                else:
                    logger.error(f"OpenRouter API error: {response.status}")
                    return llm_response
                    
    except Exception as e:
        logger.error(f"Error formatting report with AI: {e}")
        return llm_response

def create_html_template() -> str:
    """
    Create the HTML template for the reports viewing page.
    This function returns the contents of the view_reports.html template.
    """
    try:
        # Try to read the template file
        template_path = Path("ew_agents/templates/view_reports.html")
        if template_path.exists():
            return template_path.read_text(encoding='utf-8')
        else:
            # Return a basic fallback template if file doesn't exist
            return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ElectionWatch - View Reports</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 1000px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .header { text-align: center; margin-bottom: 40px; }
        .form-group { margin-bottom: 20px; }
        select, button { width: 100%; padding: 12px; margin: 5px 0; border: 1px solid #ddd; border-radius: 5px; }
        button { background: #007bff; color: white; cursor: pointer; }
        button:hover { background: #0056b3; }
        .report-section { margin-top: 30px; display: none; }
        .loading { text-align: center; padding: 20px; }
        .error { background: #f8d7da; color: #721c24; padding: 15px; border-radius: 5px; margin: 10px 0; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 ElectionWatch Reports</h1>
            <p>View and download analysis reports</p>
        </div>
        
        <div class="form-group">
            <label for="reportSelect">Available Reports:</label>
            <select id="reportSelect">
                <option value="">-- Select a report --</option>
            </select>
        </div>
        
        <button onclick="viewReport()">View Report</button>
        
        <div id="reportSection" class="report-section">
            <div id="loading" class="loading" style="display: none;">
                <h3>🔄 Generating Report...</h3>
            </div>
            <div id="error" class="error" style="display: none;"></div>
            <div id="reportContent" style="display: none;"></div>
            <div id="reportActions" style="display: none;">
                <button onclick="downloadPDF()">📄 Download PDF</button>
                <button onclick="printReport()">🖨️ Print Report</button>
            </div>
        </div>
    </div>
    
    <script>
        let currentAnalysisId = null;
        
        window.onload = function() { loadAvailableReports(); };
        
        async function loadAvailableReports() {
            try {
                const response = await fetch('/api/reports/available');
                const data = await response.json();
                const select = document.getElementById('reportSelect');
                select.innerHTML = '<option value="">-- Select a report --</option>';
                data.reports.forEach(report => {
                    const option = document.createElement('option');
                    option.value = report.analysis_id;
                    option.textContent = `${report.analysis_id} - ${report.date_analyzed}`;
                    select.appendChild(option);
                });
            } catch (error) {
                console.error('Error loading reports:', error);
            }
        }
        
        async function viewReport() {
            const analysisId = document.getElementById('reportSelect').value;
            if (!analysisId) return;
            
            currentAnalysisId = analysisId;
            document.getElementById('loading').style.display = 'block';
            document.getElementById('reportSection').style.display = 'block';
            
            try {
                const response = await fetch(`/api/reports/generate/${analysisId}`);
                const data = await response.json();
                
                if (data.success) {
                    document.getElementById('reportContent').innerHTML = data.formatted_report;
                    document.getElementById('reportContent').style.display = 'block';
                    document.getElementById('reportActions').style.display = 'block';
                } else {
                    document.getElementById('error').textContent = data.error;
                    document.getElementById('error').style.display = 'block';
                }
            } catch (error) {
                document.getElementById('error').textContent = 'Failed to generate report';
                document.getElementById('error').style.display = 'block';
            }
            document.getElementById('loading').style.display = 'none';
        }
        
        async function downloadPDF() {
            if (!currentAnalysisId) return;
            try {
                const response = await fetch(`/api/reports/download/${currentAnalysisId}`);
                if (response.ok) {
                    const blob = await response.blob();
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `electionwatch_report_${currentAnalysisId}.pdf`;
                    document.body.appendChild(a);
                    a.click();
                    window.URL.revokeObjectURL(url);
                    document.body.removeChild(a);
                }
            } catch (error) {
                console.error('Error downloading PDF:', error);
            }
        }
        
        function printReport() {
            const content = document.getElementById('reportContent').innerHTML;
            const printWindow = window.open('', '_blank');
            printWindow.document.write(`
                <html>
                    <head><title>ElectionWatch Report</title></head>
                    <body style="font-family: Arial, sans-serif; margin: 20px;">
                        <h1>ElectionWatch Analysis Report</h1>
                        ${content}
                    </body>
                </html>
            `);
            printWindow.document.close();
            printWindow.print();
        }
    </script>
</body>
</html>
            """
    except Exception as e:
        logger.error(f"Error creating HTML template: {e}")
        return "<h1>Error loading template</h1><p>Please try again later.</p>"

def generate_pdf_report(report_content: str, analysis_id: str) -> BytesIO:
    """
    Generate a PDF report from the formatted content with enhanced aesthetics.
    """
    try:
        # Create a buffer for the PDF
        buffer = BytesIO()
        
        # Create the PDF document with margins
        doc = SimpleDocTemplate(buffer, pagesize=A4, 
                              leftMargin=50, rightMargin=50, 
                              topMargin=50, bottomMargin=50)
        styles = getSampleStyleSheet()
        
        # Create enhanced custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.darkblue,
            alignment=1,  # Center alignment
            fontName='Helvetica-Bold'
        )
        
        subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=styles['Normal'],
            fontSize=12,
            spaceAfter=20,
            textColor=colors.grey,
            alignment=1,  # Center alignment
            fontName='Helvetica'
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            spaceAfter=15,
            spaceBefore=20,
            textColor=colors.darkblue,
            fontName='Helvetica-Bold',
            leftIndent=0
        )
        
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=11,
            spaceAfter=8,
            textColor=colors.black,
            fontName='Helvetica',
            leftIndent=0
        )
        
        highlight_style = ParagraphStyle(
            'CustomHighlight',
            parent=styles['Normal'],
            fontSize=11,
            spaceAfter=8,
            textColor=colors.black,
            fontName='Helvetica',
            leftIndent=20,
            backColor=colors.lightgrey
        )
        
        # Build the PDF content
        story = []
        
        # Professional header with icon-like elements
        story.append(Paragraph("🔍 ELECTION WATCH ANALYSIS", title_style))
        story.append(Paragraph("OSINT Report", subtitle_style))
        
        # Metadata section with clean formatting
        story.append(Paragraph(f"📋 Report ID: {analysis_id}", normal_style))
        story.append(Paragraph(f"📅 Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", normal_style))
        story.append(Spacer(1, 30))
        
        # Parse HTML content and convert to PDF with better structure
        import re
        
        # Remove CSS styles
        content_without_css = re.sub(r'<style>.*?</style>', '', report_content, flags=re.DOTALL)
        
        # Extract structured content from HTML
        sections = []
        
        # Find main sections
        section_patterns = [
            (r'<h1[^>]*>(.*?)</h1>', 'h1'),
            (r'<h2[^>]*>(.*?)</h2>', 'h2'),
            (r'<div class="([^"]*)"[^>]*>(.*?)</div>', 'div'),
            (r'<p[^>]*>(.*?)</p>', 'p')
        ]
        
        # Extract text content while preserving structure
        content_text = re.sub(r'<[^>]+>', '', content_without_css)
        
        # Split into lines and process with better formatting
        lines = content_text.split('\n')
        current_section = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Skip CSS remnants and empty lines
            if not line or line.startswith('{') or line.startswith('}') or line.startswith('.'):
                continue
            
            # Detect section headers (common patterns from LLM output)
            if any(keyword in line.lower() for keyword in ['executive summary', 'key findings', 'risk assessment', 'recommendations', 'technical analysis', 'technical details']):
                if current_section:  # Add spacing between sections
                    story.append(Spacer(1, 15))
                current_section = line
                story.append(Paragraph(f"📊 {line.upper()}", heading_style))
                story.append(Spacer(1, 10))
            elif line.startswith('•') or line.startswith('-') or line.startswith('*'):
                # Bullet points
                story.append(Paragraph(f"  {line}", normal_style))
            elif len(line) > 100:  # Long paragraphs
                story.append(Paragraph(line, normal_style))
                story.append(Spacer(1, 5))
            elif line.isupper() and len(line) < 50:  # Small caps headers
                story.append(Paragraph(line, heading_style))
            else:
                # Regular content
                story.append(Paragraph(line, normal_style))
        
        # Add footer
        story.append(Spacer(1, 30))
        story.append(Paragraph("─" * 50, normal_style))
        story.append(Paragraph("Generated by ElectionWatch Intelligence Platform", subtitle_style))
        story.append(Paragraph("For Internal use only | Not For Public Use", subtitle_style))
        
        # Build the PDF
        doc.build(story)
        buffer.seek(0)
        return buffer
        
    except Exception as e:
        logger.error(f"Error generating PDF: {e}")
        # Return a simple error PDF
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        story = [Paragraph("Error generating PDF report", styles['Heading1'])]
        doc.build(story)
        buffer.seek(0)
        return buffer

# Configuration following ADK standards
SESSION_SERVICE_URI = "sqlite:///./sessions.db"
ALLOWED_ORIGINS = [
    "http://localhost", 
    "http://localhost:8080", 
    "http://localhost:3000",
    "*"  # Remove in production, add specific domains
]

# Enable web interface for testing and debugging
SERVE_WEB_INTERFACE = True

# MongoDB storage integration via ew_agents
from ew_agents.mongodb_storage import (
    storage,
    store_analysis,
    get_analysis,
    store_report,
    get_report,
    get_stats
)

# Storage function aliases for compatibility
async def store_analysis_result(analysis_id: str, analysis_data: Dict[str, Any]) -> bool:
    """Store analysis result using ew_agents storage."""
    return await store_analysis(analysis_id, analysis_data)



async def get_analysis_result(analysis_id: str) -> Optional[Dict[str, Any]]:
    """Get analysis result using ew_agents storage."""
    return await get_analysis(analysis_id)

async def store_report_submission(submission_id: str, report_data: Dict[str, Any]) -> bool:
    """Store report submission using ew_agents storage."""
    return await store_report(submission_id, report_data)

async def get_report_submission(submission_id: str) -> Optional[Dict[str, Any]]:
    """Get report submission using ew_agents storage."""
    return await get_report(submission_id)

async def get_storage_stats() -> Dict[str, Any]:
    """Get storage statistics using ew_agents storage."""
    return await get_stats()

# Pydantic models for request/response validation
class AnalysisMetadata(BaseModel):
    content_type: str = Field(..., description="Type of content being analyzed")
    region: str = Field(default="general", description="Geographic region")
    language: str = Field(default="auto", description="Content language")

class AnalysisRequest(BaseModel):
    analysis_type: str = Field(default="misinformation_detection", description="Type of analysis to perform")
    text_content: Optional[str] = Field(None, description="Text content to analyze")
    priority: str = Field(default="medium", description="Analysis priority level")
    metadata: Optional[AnalysisMetadata] = Field(default_factory=AnalysisMetadata)

class ReportMetadata(BaseModel):
    analyst: str = Field(default="system", description="Analyst identifier")
    region: str = Field(..., description="Geographic region")
    time_period: Optional[str] = Field(None, description="Analysis time period")
    sources_analyzed: int = Field(default=1, description="Number of sources analyzed")

class SupportingData(BaseModel):
    analysis_ids: List[str] = Field(default_factory=list, description="Related analysis IDs")
    confidence_scores: List[float] = Field(default_factory=list, description="Confidence scores")
    trend_indicators: List[str] = Field(default_factory=list, description="Trend indicators")

class ReportSubmission(BaseModel):
    report_id: str = Field(..., description="Unique report identifier")
    report_type: str = Field(..., description="Type of report")
    findings: str = Field(..., description="Analysis findings")
    threat_level: str = Field(..., description="Assessed threat level")
    metadata: ReportMetadata = Field(..., description="Report metadata")
    supporting_data: Optional[SupportingData] = Field(default_factory=SupportingData)

def create_app():
    """Create and configure the FastAPI application using ADK standards."""
    
    # Use ADK's built-in FastAPI app factory
    app = get_fast_api_app(
        agents_dir=AGENT_DIR,
        session_service_uri=SESSION_SERVICE_URI,
        allow_origins=ALLOWED_ORIGINS,
        web=SERVE_WEB_INTERFACE,
    )
    
    templates = Jinja2Templates(directory="ew_agents/templates")
    
    # ===== ELECTIONWATCH CUSTOM ENDPOINTS =====
    
    @app.post("/run_analysis")
    async def run_analysis(
        text: str = Form(None),
        files: List[UploadFile] = File(default=[]),
        analysis_type: str = Form("misinformation_detection"),
        priority: str = Form("medium"),
        source: str = Form("api_upload"),
        metadata: str = Form("{}")
    ):
        """
        Test endpoint that outputs raw LLM response and structured report
        for debugging the data pipeline.
        
        Returns:
        - LLM_Response: Raw OSINT agent analysis
        - Report: Structured JSON following data/outputs/response_*.json format
        """
        start_time = datetime.now()
        analysis_id = f"analysis_{int(start_time.timestamp())}"
        
        try:
            # Parse metadata safely
            try:
                metadata_dict = json.loads(metadata) if metadata else {}
            except json.JSONDecodeError:
                metadata_dict = {"parsing_error": "Invalid JSON in metadata"}
            
            # Initialize processed_files list
            processed_files = []
            
            # Process uploaded files (optimized version)
            for file in files:
                if file.filename:
                    try:
                        content = await file.read()
                        file_info = {
                            "filename": file.filename,
                            "content_type": file.content_type,
                            "size_bytes": len(content)
                        }
                        
                        if file.content_type in ["text/csv", "application/csv"]:
                            csv_text = content.decode('utf-8')
                            # Use optimized CSV processing instead of raw text concatenation
                            file_info["processed"] = "csv_structured"
                            file_info["csv_content"] = csv_text  # Store for structured processing
                        elif file.content_type and file.content_type.startswith("text/"):
                            text_content = content.decode('utf-8')
                            # Use optimized text processing
                            file_info["processed"] = "text_optimized"
                            file_info["text_content"] = text_content
                        else:
                            file_info["processed"] = "metadata_only"
                        
                        processed_files.append(file_info)
                        
                    except Exception as e:
                        processed_files.append({
                            "filename": file.filename,
                            "error": f"File processing error: {str(e)}"
                        })
            
            # Create optimized analysis content
            if text:
                all_text = text
            elif processed_files:
                # Use structured processing instead of raw concatenation
                analysis_content = {
                    "files": processed_files,
                    "metadata": metadata_dict,
                    "processing_optimized": True,
                    "platform_detection": True
                }
                all_text = json.dumps(analysis_content, indent=2)
            else:
                all_text = ""
            
            # Run ADK analysis (same as original)
            if all_text.strip():
                try:
                    from ew_agents.agent import root_agent
                    from google.adk.runners import InMemoryRunner
                    from google.genai import types
                    
                    runner = InMemoryRunner(root_agent)
                    
                    user_content = types.Content(
                        role="user",
                        parts=[
                            types.Part(text=f"Analyze this content for election-related misinformation, narratives, and risks: {all_text} Metadata: {json.dumps(metadata_dict)}")]
                    )
                    
                    import uuid
                    session_id = f"test_{uuid.uuid4().hex[:8]}"
                    user_id = "test_user"
                    
                    # Create session first if needed (ADK should auto-create, but let's be explicit)
                    try:
                        session = await runner.session_service.create_session(
                            app_name=runner.app_name,
                            user_id=user_id,
                            session_id=session_id
                        )
                        logger.info(f"Created session: {session_id}")
                    except Exception as e:
                        logger.info(f"Session creation note: {e}, proceeding with run_async")
                    
                    # Run the agent
                    analysis_events = []
                    async for event in runner.run_async(
                        user_id=user_id,
                        session_id=session_id,
                        new_message=user_content
                    ):
                        analysis_events.append(event)
                        logger.info(f"ADK Event: {event.author if hasattr(event, 'author') else 'unknown'}")
                    
                    # Extract final response from events with better debugging
                    llm_response = "ElectionWatch Analysis Completed"
                    logger.info(f"Total events received: {len(analysis_events)}")
                    
                    for i, event in enumerate(reversed(analysis_events)):  # Check from last to first
                        logger.info(f"Checking event {len(analysis_events) - i}: {type(event).__name__}")
                        
                        # Check for content in the event
                        if hasattr(event, 'content') and event.content:
                            logger.info(f"Event has content: {type(event.content)}")
                            if hasattr(event.content, 'parts') and event.content.parts:
                                llm_response = event.content.parts[0].text
                                logger.info(f"Found response in content.parts: {llm_response[:100]}...")
                                break
                            else:
                                llm_response = str(event.content)
                                logger.info(f"Found response in content: {llm_response[:100]}...")
                                break
                        
                        # Check for final response flag
                        elif hasattr(event, 'is_final_response') and event.is_final_response():
                            logger.info("Found final response event")
                            if hasattr(event, 'content') and event.content:
                                if hasattr(event.content, 'parts') and event.content.parts:
                                    llm_response = event.content.parts[0].text
                                else:
                                    llm_response = str(event.content)
                                logger.info(f"Final response: {llm_response[:100]}...")
                                break
                        
                        # Check for text attribute directly
                        elif hasattr(event, 'text') and event.text:
                            llm_response = event.text
                            logger.info(f"Found response in text: {llm_response[:100]}...")
                            break
                        
                        # Check for message attribute
                        elif hasattr(event, 'message') and event.message:
                            llm_response = str(event.message)
                            logger.info(f"Found response in message: {llm_response[:100]}...")
                            break
                    
                    # If we still have the default response, try to get the last meaningful event
                    if llm_response == "ElectionWatch Analysis Completed" and analysis_events:
                        last_event = analysis_events[-1]
                        logger.info(f"Using last event as fallback: {type(last_event).__name__}")
                        llm_response = str(last_event)
                    
                    logger.info(f"Final analysis result: {llm_response[:200]}...")
                    logger.info(f"Full response length: {len(llm_response)}")
                    if len(llm_response) > 200:
                        logger.info(f"Response continues: ...{llm_response[200:400]}...")
                    
                    end_time = datetime.now()
                    processing_time = (end_time - start_time).total_seconds()
                    
                    # Check if LLM response is already valid JSON and store it
                    agent_json_response = None
                    try:
                        parsed_json = json.loads(llm_response)
                        logger.info("✅ Agent returned valid JSON, will store and return")
                        agent_json_response = parsed_json
                    except json.JSONDecodeError:
                        logger.info("Agent response not in JSON format, extracting data dynamically")
                        
                        # Try to extract JSON from the response if it contains JSON-like content
                        import re
                        json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
                        if json_match:
                            try:
                                extracted_json = json.loads(json_match.group())
                                logger.info("✅ Extracted JSON from response")
                                agent_json_response = extracted_json
                            except json.JSONDecodeError:
                                logger.info("Failed to parse extracted JSON")
                    
                    # Dynamic report structure - start empty and populate based on LLM content
                    report = {
                        "report_metadata": {
                            "report_id": analysis_id,
                            "analysis_timestamp": end_time.isoformat(),
                            "report_type": "analysis",
                            "content_type": metadata_dict.get("content_type", "text_post"),
                            "analysis_depth": "standard",
                            "content_source": source,
                            "processing_time_seconds": processing_time
                        },
                        "narrative_classification": {},
                        "actors": [],
                        "lexicon_terms": [],
                        "risk_level": "low",  # Default, will be updated based on content
                        "date_analyzed": end_time.isoformat(),
                        "recommendations": [],
                        "analysis_insights": {
                            "content_statistics": {
                                "word_count": len(all_text.split()),
                                "character_count": len(all_text),
                                "language_detected": "en"
                            },
                            "key_findings": "",  # Will be extracted from LLM
                            "risk_factors": [],  # Will be extracted from LLM
                            "confidence_level": "medium",
                            "llm_response": llm_response
                        }
                    }
                    
                    # Apply the same data extraction logic as the fixed /AnalysePosts endpoint
                    # Check for workflow completion
                    key_indicators = [
                        "ElectionWatch analysis workflow",
                        "DataEngAgent: Extraction complete",
                        "OsintAgent: Analysis complete",
                        # "✓ OsintAgent: Analysis complete",
                        "Narrative Classification",
                        "Political Actors and Their Roles"
                    ]
                    workflow_completed = any(phrase in llm_response for phrase in key_indicators)
                    
                    # ENHANCED: Use Knowledge Base Integration instead of hardcoded string matching
                    logger.info('🔍 Using knowledge base for narrative classification and lexicon extraction')
                    
                    # Initialize search variables
                    narrative_search = None
                    lexicon_search = None
                    
                    try:
                        # Import knowledge base functions
                        from ew_agents.knowledge_retrieval import search_knowledge, analyze_content
                        
                        # Perform semantic search for narrative classification
                        narrative_search = await search_knowledge(llm_response, collections=['narratives'])
                        logger.info(f'📚 Narrative search: {len(narrative_search.get("narratives", {}).get("source_nodes", []))} matches found')
                        
                        # Perform semantic search for lexicon terms  
                        lexicon_search = await search_knowledge(llm_response, collections=['hate_speech_lexicon'])
                        logger.info(f'📖 Lexicon search: {len(lexicon_search.get("hate_speech_lexicon", {}).get("source_nodes", []))} matches found')
                        
                        # Extract narrative classification from knowledge base results
                        if narrative_search and narrative_search.get('narratives', {}).get('source_nodes'):
                            best_narrative = narrative_search['narratives']['source_nodes'][0]
                            narrative_meta = best_narrative.get('metadata', {})
                            confidence = best_narrative.get('score', 0.5)
                            
                            report['narrative_classification'] = {
                                "theme": narrative_meta.get('category', 'general_political'),
                                "threat_level": "medium" if confidence > 0.7 else "low",
                                "details": narrative_meta.get('scenario', 'Content analyzed using knowledge base'),
                                "confidence_score": confidence,
                                "alternative_themes": narrative_meta.get('tags', [])[:3],  # Take first 3 tags
                                "threat_indicators": narrative_meta.get('key_indicators_for_ai', [])[:5]  # Take first 5 indicators
                            }
                            logger.info(f'✅ Narrative classified: {narrative_meta.get("category", "unknown")} (confidence: {confidence:.2f})')
                        else:
                            # Fallback narrative classification based on LLM analysis content
                            report['narrative_classification'] = {
                                "theme": "candidate_support_campaigning" if "Candidate Support" in llm_response else "general_political",
                                "threat_level": "low_medium" if "threat" in llm_response.lower() else "low",
                                "details": "Content analyzed using ElectionWatch AI system with LLM analysis",
                                "confidence_score": 0.6,
                                "alternative_themes": ["political_engagement", "voter_mobilization"],
                                "threat_indicators": []
                            }
                            logger.info('⚠️ No narrative matches found, using LLM-based classification')
                        
                        # Extract lexicon terms from knowledge base results and LLM analysis
                        lexicon_terms = []
                        
                        # Add knowledge base lexicon matches
                        if lexicon_search and lexicon_search.get('hate_speech_lexicon', {}).get('source_nodes'):
                            for lexicon_node in lexicon_search['hate_speech_lexicon']['source_nodes'][:3]:  # Top 3 matches
                                lexicon_meta = lexicon_node.get('metadata', {})
                                lexicon_terms.append({
                                    "term": lexicon_meta.get('term', 'unknown'),
                                    "category": lexicon_meta.get('category', 'general'),
                                    "context": "knowledge_base_match",
                                    "confidence_score": lexicon_node.get('score', 0.5),
                                    "language": lexicon_meta.get('language', 'en'),
                                    "severity": lexicon_meta.get('severity', 'medium'),
                                    "definition": lexicon_meta.get('definition', 'Term from knowledge base')
                                })
                        
                        # Add terms extracted from LLM analysis
                        if 'PVC' in llm_response:
                            lexicon_terms.append({
                                "term": "PVC",
                                "category": "voter_mobilization",
                                "context": "llm_extraction",
                                "confidence_score": 0.9,
                                "language": "en",
                                "severity": "low",
                                "definition": "Permanent Voter Card - essential for voting"
                            })
                        
                        if 'Obidients' in llm_response or '#Obidients' in llm_response:
                            lexicon_terms.append({
                                "term": "Obidients",
                                "category": "political_movement",
                                "context": "llm_extraction",
                                "confidence_score": 0.8,
                                "language": "en",
                                "severity": "low",
                                "definition": "Supporters of Peter Obi and Labour Party"
                            })
                        
                        if 'Labour Party' in llm_response:
                            lexicon_terms.append({
                                "term": "Labour Party",
                                "category": "political_party",
                                "context": "llm_extraction",
                                "confidence_score": 0.9,
                                "language": "en",
                                "severity": "low",
                                "definition": "Political party in Nigerian elections"
                            })
                        
                        # If no terms found, add default
                        if not lexicon_terms:
                            lexicon_terms = [{
                                "term": "election content",
                                "category": "general",
                                "context": "default",
                                "confidence_score": 0.6,
                                "language": "en",
                                "severity": "low",
                                "definition": "General election-related content"
                            }]
                        
                        report['lexicon_terms'] = lexicon_terms
                        logger.info(f'✅ Extracted {len(lexicon_terms)} lexicon terms')
                        
                        # Add LLM response to analysis_insights
                        report['analysis_insights']["llm_response"] = llm_response
                        logger.info('✅ Added LLM response to analysis_insights')
                        
                        # Extract key findings from knowledge base
                        if narrative_search and narrative_search.get('narratives', {}).get('response'):
                            report['analysis_insights']['key_findings'] = narrative_search['narratives']['response'][:500]
                        else:
                            report['analysis_insights']['key_findings'] = f"Analysis of {len(all_text.split())} words of content for election-related patterns"
                        
                        # Generate recommendations based on narrative classification
                        recommendations = []
                        if report['narrative_classification']['threat_level'] in ['medium', 'high']:
                            recommendations.extend([
                                "Enhanced monitoring recommended based on knowledge base analysis",
                                "Cross-reference with similar historical patterns",
                                "Monitor for escalation indicators"
                            ])
                        else:
                            recommendations.append("Standard monitoring protocols apply")
                        
                        if lexicon_terms and len(lexicon_terms) > 2:
                            recommendations.append(f"Track usage patterns of {len(lexicon_terms)} identified terms")
                        
                        report['recommendations'] = recommendations[:5]  # Limit to 5 recommendations
                        
                        # Set risk level based on narrative classification
                        report['risk_level'] = report['narrative_classification']['threat_level']
                        
                        logger.info('✅ Knowledge base integration completed successfully')
                        
                    except Exception as kb_error:
                        logger.warning(f'⚠️ Knowledge base integration failed: {kb_error}')
                        logger.info('🔄 Falling back to basic analysis without knowledge base enhancement')
                        
                        # Basic fallback analysis
                        report['narrative_classification'] = {
                            "theme": "general_political",
                            "threat_level": "low",
                            "details": "Content analyzed using basic ElectionWatch AI system",
                            "confidence_score": 0.5,
                            "alternative_themes": ["political_engagement"],
                            "threat_indicators": []
                        }
                        
                        report['lexicon_terms'] = [{
                            "term": "election content",
                            "category": "general",
                            "context": "fallback",
                            "confidence_score": 0.5,
                            "language": "en",
                            "severity": "low",
                            "definition": "General election-related content"
                        }]
                        
                        report['recommendations'] = ["Standard monitoring protocols apply"]
                        report['risk_level'] = "low"
                
                    # Store the analysis result
                    try:
                        await store_analysis_result(analysis_id, report)
                        logger.info(f'✅ Analysis stored with ID: {analysis_id}')
                    except Exception as store_error:
                        logger.warning(f'⚠️ Failed to store analysis: {store_error}')
                    
                    # Return the final report
                    return report
                    
                    # Handle agent JSON responses
                    if agent_json_response:
                        # Store the agent JSON response
                        try:
                            await store_analysis_result(analysis_id, agent_json_response)
                            logger.info(f'✅ Agent JSON analysis stored with ID: {analysis_id}')
                        except Exception as store_error:
                            logger.warning(f'⚠️ Failed to store agent JSON analysis: {store_error}')
                        
                        # Return the agent JSON response
                        return agent_json_response
                    
                except Exception as e:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Agent processing error: {str(e)}"
                    )
            else:
                raise HTTPException(
                    status_code=400,
                    detail="No content provided for analysis"
                )
                
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Analysis processing error: {str(e)}"
            )
    

    @app.post("/submitReport")
    async def submit_report(report: ReportSubmission):
        """
        Submit a manual report for storage and analysis.
        """
        try:
            # Store the report submission
            success = await store_report_submission(report.report_id, report.dict())
            
            if success:
                return {"status": "success", "message": "Report submitted successfully"}
            else:
                raise HTTPException(status_code=500, detail="Failed to store report")
                
        except Exception as e:
            return {
                "error": "Failed to submit report",
                "details": str(e)
            }

    # ===== REPORT VIEWING ENDPOINTS =====
    
    @app.get("/view_reports")
    async def view_reports():
        """
        Serve the reports viewing page with dropdown of available reports.
        """
        try:
            html_content = create_html_template()
            return HTMLResponse(content=html_content, status_code=200)
        except Exception as e:
            logger.error(f"Error serving reports page: {e}")
            return HTMLResponse(
                content="<h1>Error loading reports page</h1><p>Please try again later.</p>",
                status_code=500
            )

    @app.get("/api/reports/available")
    async def get_available_reports():
        """
        Get list of available reports from MongoDB analysis_results collection for the dropdown.
        """
        try:
            # Get the actual analysis data directly from storage
            from ew_agents.mongodb_storage import storage
            analyses = await storage.list_recent_analyses(limit=50)
            
            reports = []
            for analysis in analyses:
                # Check if this analysis has LLM response data
                # The list_recent_analyses returns full documents, so we need to look in data.llm_response
                data = analysis.get("data", {})
                llm_response = data.get("llm_response", "")
                
                # If not found in data, look in analysis_insights
                if not llm_response:
                    analysis_insights = data.get("analysis_insights", {})
                    llm_response = analysis_insights.get("llm_response", "")
                
                if llm_response and len(llm_response.strip()) > 10:  # Only include analyses with substantial LLM responses
                    # Get timestamp
                    date_analyzed = data.get("date_analyzed", analysis.get("created_at", "Unknown"))
                    if date_analyzed and date_analyzed != "Unknown":
                        try:
                            if isinstance(date_analyzed, str):
                                dt = datetime.fromisoformat(date_analyzed.replace('Z', '+00:00'))
                                date_str = dt.strftime("%Y-%m-%d %H:%M")
                            else:
                                date_str = date_analyzed.strftime("%Y-%m-%d %H:%M")
                        except:
                            date_str = "Unknown"
                    else:
                        date_str = "Unknown"
                    
                    reports.append({
                        "analysis_id": analysis.get("analysis_id", "unknown"),
                        "date_analyzed": date_str,
                        "analysis_type": data.get("analysis_type", "misinformation_analysis"),
                        "risk_level": data.get("risk_level", "unknown"),
                        "content_preview": llm_response[:100] + "..." if len(llm_response) > 100 else llm_response
                    })
            
            # Sort by date (newest first)
            reports.sort(key=lambda x: x["date_analyzed"], reverse=True)
            
            return {
                "success": True,
                "reports": reports,
                "total_count": len(reports)
            }
            
        except Exception as e:
            logger.error(f"Error getting available reports: {e}")
            return {
                "success": False,
                "error": str(e),
                "reports": [],
                "total_count": 0
            }

    @app.get("/api/reports/generate/{analysis_id}")
    async def generate_formatted_report(analysis_id: str):
        """
        Generate a formatted report using AI for the specified analysis.
        """
        try:
            # Get the analysis data from MongoDB storage
            from ew_agents.mongodb_storage import storage
            analysis_data = await storage.get_analysis_result(analysis_id)
            
            if not analysis_data:
                return {
                    "success": False,
                    "error": f"Analysis with ID {analysis_id} not found"
                }
            
            # Extract LLM response from the analysis data
            # The MongoDB storage returns the data field directly (not wrapped)
            # So analysis_data IS the data object containing llm_response
            llm_response = analysis_data.get("llm_response", "")
            
            # If not found at top level, look in analysis_insights
            if not llm_response:
                analysis_insights = analysis_data.get("analysis_insights", {})
                llm_response = analysis_insights.get("llm_response", "")
            
            # Debug logging
            logger.info(f"Analysis data keys: {list(analysis_data.keys()) if analysis_data else 'None'}")
            logger.info(f"LLM response found: {bool(llm_response)}")
            logger.info(f"LLM response length: {len(llm_response) if llm_response else 0}")
            
            if not llm_response:
                return {
                    "success": False,
                    "error": "No LLM response found in analysis data"
                }
            
            # Format the report using AI
            formatted_report = await format_report_with_ai(llm_response, analysis_data)
            
            return {
                "success": True,
                "formatted_report": formatted_report,
                "analysis_id": analysis_id,
                "original_data": analysis_data
            }
            
        except Exception as e:
            logger.error(f"Error generating formatted report: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    @app.get("/api/reports/download/{analysis_id}")
    async def download_pdf_report(analysis_id: str):
        """
        Download a PDF version of the formatted report.
        """
        try:
            # Get the analysis data from MongoDB storage
            from ew_agents.mongodb_storage import storage
            analysis_data = await storage.get_analysis_result(analysis_id)
            
            if not analysis_data:
                raise HTTPException(status_code=404, detail=f"Analysis with ID {analysis_id} not found")
            
            # Extract LLM response from the analysis data
            # The MongoDB storage returns the data field directly (not wrapped)
            # So analysis_data IS the data object containing llm_response
            llm_response = analysis_data.get("llm_response", "")
            
            # If not found at top level, look in analysis_insights
            if not llm_response:
                analysis_insights = analysis_data.get("analysis_insights", {})
                llm_response = analysis_insights.get("llm_response", "")
            
            # Debug logging
            logger.info(f"Analysis data keys: {list(analysis_data.keys()) if analysis_data else 'None'}")
            logger.info(f"LLM response found: {bool(llm_response)}")
            logger.info(f"LLM response length: {len(llm_response) if llm_response else 0}")
            
            if not llm_response:
                raise HTTPException(status_code=400, detail="No LLM response found in analysis data")
            
            # Format the report using AI
            formatted_report = await format_report_with_ai(llm_response, analysis_data)
            
            # Generate PDF
            pdf_buffer = generate_pdf_report(formatted_report, analysis_id)
            
            # Return the PDF as a response with proper headers
            from fastapi.responses import Response
            
            return Response(
                content=pdf_buffer.getvalue(),
                media_type="application/pdf",
                headers={
                    "Content-Disposition": f"attachment; filename=electionwatch_report_{analysis_id}.pdf"
                }
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error downloading PDF report: {e}")
            raise HTTPException(status_code=500, detail=f"Error generating PDF: {str(e)}")

    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}
    
    @app.get("/debug/agent-test")
    async def test_agent_response():
        """Test endpoint to check agent response format"""
        try:
            from ew_agents.agent import root_agent
            from google.adk.runners import InMemoryRunner
            from google.genai import types
            
            runner = InMemoryRunner(root_agent)
            
            user_content = types.Content(
                role="user",
                parts=[types.Part(text="Analyze this simple text: 'Election day is tomorrow'")]
            )
            
            import uuid
            session_id = f"test_{uuid.uuid4().hex[:8]}"
            user_id = "test_user"
            
            # Run the agent
            analysis_events = []
            async for event in runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=user_content
            ):
                analysis_events.append(event)
                logger.info(f"Test Event: {type(event).__name__}")
            
            # Extract response
            llm_response = "No response found"
            for event in reversed(analysis_events):
                if hasattr(event, 'content') and event.content and hasattr(event.content, 'parts') and event.content.parts:
                    llm_response = event.content.parts[0].text
                    break
            
            return {
                "success": True,
                "events_count": len(analysis_events),
                "event_types": [type(event).__name__ for event in analysis_events],
                "response": llm_response,
                "response_length": len(llm_response)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    return app

if __name__ == "__main__":
    # Create the app
    app = create_app()
    
    # Run with Uvicorn
    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8080)),
        log_level="info"
    )