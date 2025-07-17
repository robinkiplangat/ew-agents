#!/usr/bin/env python3
"""
Test script for ElectionWatch Agent Gateway API

Tests the complete flow of the gateway API calling deployed agents.
"""

import requests
import json
import time
import sys
from typing import Dict, Any

class GatewayAPITester:
    """Test client for the ElectionWatch Gateway API"""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def test_health(self) -> bool:
        """Test the health endpoint"""
        print("🏥 Testing health endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            
            data = response.json()
            print(f"   Status: {data.get('status')}")
            print(f"   Gateway Mode: {data.get('gateway_mode')}")
            print(f"   Agent Client: {data.get('agent_client_available')}")
            
            return data.get('status') == 'healthy'
        except Exception as e:
            print(f"   ❌ Health check failed: {e}")
            return False
    
    def test_agents_list(self) -> bool:
        """Test the agents listing endpoint"""
        print("\n🤖 Testing agents list endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/agents")
            response.raise_for_status()
            
            data = response.json()
            print(f"   Mode: {data.get('mode')}")
            print(f"   Agents found: {len(data.get('agents', []))}")
            
            for agent in data.get('agents', []):
                print(f"   - {agent['name']}: {agent['status']} ({agent.get('available', False)})")
            
            return len(data.get('agents', [])) > 0
        except Exception as e:
            print(f"   ❌ Agent list failed: {e}")
            return False
    
    def test_analysis(self, content: str, source_platform: str = "test") -> bool:
        """Test the content analysis endpoint"""
        print(f"\n📝 Testing analysis endpoint...")
        print(f"   Content: {content[:50]}...")
        
        try:
            payload = {
                "content": content,
                "source_platform": source_platform,
                "request_type": "comprehensive"
            }
            
            response = self.session.post(
                f"{self.base_url}/analyze",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            data = response.json()
            print(f"   Status: {data.get('status')}")
            print(f"   Content Preview: {data.get('content_preview', '')[:50]}...")
            
            analysis = data.get('analysis', {})
            if analysis:
                print(f"   Summary: {analysis.get('summary', 'No summary')[:100]}...")
                
                risk = analysis.get('risk_assessment', {})
                if risk:
                    print(f"   Risk Level: {risk.get('level')} (confidence: {risk.get('confidence', 0):.2f})")
                
                themes = analysis.get('content_analysis', {}).get('themes', [])
                if themes:
                    print(f"   Themes: {', '.join(themes)}")
                
                recommendations = analysis.get('recommendations', [])
                if recommendations:
                    print(f"   Recommendations: {len(recommendations)} items")
            
            return data.get('status') == 'completed'
        except Exception as e:
            print(f"   ❌ Analysis failed: {e}")
            return False
    
    def test_status(self) -> bool:
        """Test the detailed status endpoint"""
        print("\n📊 Testing status endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/status")
            response.raise_for_status()
            
            data = response.json()
            print(f"   Service: {data.get('service')}")
            print(f"   Version: {data.get('version')}")
            print(f"   Uptime: {data.get('uptime')}")
            
            capabilities = data.get('capabilities', {})
            print(f"   Capabilities:")
            for cap, status in capabilities.items():
                print(f"     - {cap}: {status}")
            
            return True
        except Exception as e:
            print(f"   ❌ Status check failed: {e}")
            return False
    
    def run_comprehensive_test(self) -> Dict[str, bool]:
        """Run all tests and return results"""
        print("🧪 ElectionWatch Gateway API Comprehensive Test")
        print("=" * 50)
        
        results = {}
        
        # Test 1: Health check
        results['health'] = self.test_health()
        
        # Test 2: Agent listing
        results['agents_list'] = self.test_agents_list()
        
        # Test 3: Simple analysis
        results['analysis_simple'] = self.test_analysis(
            "This is a test message about election processes.",
            "test"
        )
        
        # Test 4: Election-related analysis
        results['analysis_election'] = self.test_analysis(
            "There are concerns about vote rigging in the upcoming election. "
            "Some people are spreading false information about BVAS systems.",
            "twitter"
        )
        
        # Test 5: Status check
        results['status'] = self.test_status()
        
        print("\n" + "=" * 50)
        print("📋 Test Results Summary:")
        
        passed = 0
        total = len(results)
        
        for test_name, passed_test in results.items():
            status = "✅ PASS" if passed_test else "❌ FAIL"
            print(f"   {test_name}: {status}")
            if passed_test:
                passed += 1
        
        print(f"\n🎯 Overall: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All tests passed! Gateway API is working correctly.")
        else:
            print("⚠️  Some tests failed. Check the API setup.")
        
        return results

def main():
    """Main test function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test ElectionWatch Gateway API")
    parser.add_argument(
        "--url", 
        default="http://localhost:8080",
        help="Base URL for the API (default: http://localhost:8080)"
    )
    parser.add_argument(
        "--wait",
        type=int,
        default=5,
        help="Seconds to wait for server startup (default: 5)"
    )
    
    args = parser.parse_args()
    
    print(f"🔗 Testing API at: {args.url}")
    
    if args.wait > 0:
        print(f"⏳ Waiting {args.wait} seconds for server startup...")
        time.sleep(args.wait)
    
    tester = GatewayAPITester(args.url)
    results = tester.run_comprehensive_test()
    
    # Exit with error code if any tests failed
    if not all(results.values()):
        sys.exit(1)
    
    print("\n✅ All tests completed successfully!")

if __name__ == "__main__":
    main() 