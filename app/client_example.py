#!/usr/bin/env python3
"""
ElectionWatch Agent API Client Example

This script demonstrates how to interact with the ElectionWatch agent API
to submit findings and get analysis results.
"""

import asyncio
import json
import aiohttp
import websockets
from datetime import datetime
from typing import Dict, Any, Optional

# Configuration
API_BASE_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000/ws"

class ElectionWatchClient:
    """Client for interacting with the ElectionWatch Agent API"""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def health_check(self) -> Dict[str, Any]:
        """Check if the API service is healthy"""
        async with self.session.get(f"{self.base_url}/health") as response:
            return await response.json()
    
    async def analyze_content(self, content: str, source_platform: str = "unknown", 
                            request_type: str = "auto_detect") -> Dict[str, Any]:
        """Analyze content using the multi-agent system"""
        data = {
            "content": content,
            "source_platform": source_platform,
            "request_type": request_type,
            "include_trends": True,
            "include_actors": True
        }
        
        async with self.session.post(f"{self.base_url}/analyze", json=data) as response:
            return await response.json()
    
    async def submit_finding(self, content_text: str, source_platform: str,
                           source_url: Optional[str] = None, author_handle: Optional[str] = None) -> Dict[str, Any]:
        """Submit a new finding for analysis"""
        data = {
            "content_text": content_text,
            "source_platform": source_platform,
            "source_url": source_url,
            "author_handle": author_handle,
            "content_type": "post"
        }
        
        async with self.session.post(f"{self.base_url}/submit_finding", json=data) as response:
            return await response.json()
    
    async def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get the status of a specific workflow"""
        async with self.session.get(f"{self.base_url}/workflows/{workflow_id}/status") as response:
            return await response.json()
    
    async def list_workflows(self) -> Dict[str, Any]:
        """List all workflows"""
        async with self.session.get(f"{self.base_url}/workflows") as response:
            return await response.json()
    
    async def get_templates(self) -> Dict[str, Any]:
        """Get available workflow templates"""
        async with self.session.get(f"{self.base_url}/templates") as response:
            return await response.json()

async def monitor_workflow_progress(workflow_id: str):
    """Monitor workflow progress via WebSocket"""
    try:
        async with websockets.connect(WS_URL) as websocket:
            # Subscribe to workflow updates
            await websocket.send(json.dumps({
                "type": "subscribe_workflow",
                "workflow_id": workflow_id
            }))
            
            print(f"📡 Monitoring workflow: {workflow_id}")
            
            # Listen for updates
            async for message in websocket:
                data = json.loads(message)
                event_type = data.get("type")
                
                if event_type == "workflow_status":
                    status = data.get("status", {})
                    print(f"   Status: {status.get('status', 'unknown')}")
                elif event_type == "progress_update":
                    event = data.get("event", {})
                    print(f"   📊 Progress update: {event}")
                elif event_type == "analysis_complete":
                    print(f"   ✅ Analysis completed!")
                    break
                
    except Exception as e:
        print(f"❌ WebSocket error: {e}")

async def demo_content_analysis():
    """Demo: Analyze suspicious content"""
    print("\n🧪 DEMO: Content Analysis")
    print("=" * 50)
    
    async with ElectionWatchClient() as client:
        # Check health first
        health = await client.health_check()
        print(f"🏥 Service Health: {health['status']}")
        print(f"📊 ADK Available: {health['adk_available']}")
        
        # Analyze content
        content = "URGENT: BVAS machines hacked in Lagos! Don't trust the results! Share to save democracy!"
        print(f"\n📝 Analyzing: {content}")
        
        result = await client.analyze_content(content, "twitter", "narrative_classification")
        
        print(f"\n📊 Analysis Results:")
        print(f"   Status: {result['status']}")
        print(f"   Workflow ID: {result['workflow_id']}")
        print(f"   Execution Time: {result.get('execution_time', 0):.2f}s")
        
        if result['status'] == 'completed' and result.get('analysis'):
            analysis = result['analysis']
            
            # Executive summary
            if 'executive_summary' in analysis:
                summary = analysis['executive_summary']
                print(f"\n📋 Executive Summary:")
                print(f"   Request: {summary.get('request_analysis', 'N/A')}")
                
                if summary.get('key_findings'):
                    print(f"   🔍 Key Findings:")
                    for finding in summary['key_findings'][:3]:
                        print(f"      • {finding.get('finding', 'N/A')} (confidence: {finding.get('confidence', 0):.1%})")
            
            # Risk assessment
            if 'risk_assessment' in analysis:
                risk = analysis['risk_assessment']
                print(f"\n⚠️  Risk Assessment:")
                print(f"   Level: {risk.get('overall_risk_level', 'unknown')}")
                print(f"   Score: {risk.get('risk_score', 0)}")
                print(f"   Recommendation: {risk.get('recommendation', 'N/A')}")

async def demo_finding_submission():
    """Demo: Submit new findings"""
    print("\n🧪 DEMO: Finding Submission")
    print("=" * 50)
    
    async with ElectionWatchClient() as client:
        # Submit finding
        finding_content = "Breaking: Massive irregularities found in Ogun State elections. Military blocking access to polling units."
        
        print(f"📥 Submitting finding: {finding_content[:60]}...")
        
        result = await client.submit_finding(
            content_text=finding_content,
            source_platform="facebook",
            source_url="https://facebook.com/post/123456",
            author_handle="concerned_citizen_ng"
        )
        
        print(f"📊 Submission Result:")
        print(f"   Status: {result['status']}")
        print(f"   Message: {result['message']}")
        print(f"   Platform: {result['source_platform']}")

async def demo_workflow_monitoring():
    """Demo: Monitor workflow progress"""
    print("\n🧪 DEMO: Workflow Monitoring")
    print("=" * 50)
    
    async with ElectionWatchClient() as client:
        # Start an analysis
        content = "Analyzing this for monitoring demo: Election rigging in progress!"
        
        print(f"🚀 Starting analysis for monitoring...")
        result = await client.analyze_content(content, "twitter")
        
        if result.get('workflow_id'):
            workflow_id = result['workflow_id']
            print(f"📋 Workflow ID: {workflow_id}")
            
            # Monitor progress
            await monitor_workflow_progress(workflow_id)
            
            # Check final status
            final_status = await client.get_workflow_status(workflow_id)
            print(f"\n📊 Final Status:")
            print(f"   Status: {final_status['status']}")
            print(f"   Progress: {final_status['progress_percent']:.1f}%")

async def demo_workflow_management():
    """Demo: Workflow management"""
    print("\n🧪 DEMO: Workflow Management")
    print("=" * 50)
    
    async with ElectionWatchClient() as client:
        # List existing workflows
        workflows = await client.list_workflows()
        print(f"📋 Total Workflows: {workflows['total_workflows']}")
        
        if workflows['workflows']:
            print(f"   Recent workflows:")
            for wf in workflows['workflows'][:3]:
                print(f"      • {wf['workflow_id']}: {wf['status']} ({wf['task_count']} tasks)")
        
        # List available templates
        templates = await client.get_templates()
        print(f"\n🎯 Available Templates: {len(templates['templates'])}")
        for name, info in templates['templates'].items():
            print(f"   • {name}: {info['description']} ({info['task_count']} tasks)")

async def main():
    """Run all demos"""
    print("🚀 ELECTIONWATCH AGENT API CLIENT DEMO")
    print("=" * 60)
    print("Make sure the API service is running on http://localhost:8000")
    print("Start it with: python run_agent_service.py")
    
    try:
        # Run demos
        await demo_content_analysis()
        await demo_finding_submission()
        await demo_workflow_monitoring()
        await demo_workflow_management()
        
        print(f"\n🎉 All demos completed successfully!")
        print(f"💡 You can now:")
        print(f"   • Visit http://localhost:8000/docs for interactive API documentation")
        print(f"   • Use the WebSocket at ws://localhost:8000/ws for real-time updates")
        print(f"   • Integrate these endpoints into your application")
        
    except aiohttp.ClientConnectorError:
        print("❌ Cannot connect to API service!")
        print("   Make sure to start the service first:")
        print("   python run_agent_service.py")
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 