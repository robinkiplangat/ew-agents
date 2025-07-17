#!/usr/bin/env python3
"""
Simple test script for ElectionWatch Simple Agent API

Tests the basic functionality of the simple_agent_api.py service.

Note: Run this from the app/ directory or ensure the API is running
"""

import requests
import json
import time
import sys

# Configuration
BASE_URL = "http://localhost:8000"
TIMEOUT = 30

def test_health_check():
    """Test the health endpoint"""
    print("🔍 Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data['status']}")
            print(f"   Agents available: {data['agents_available']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_service_info():
    """Test the root endpoint"""
    print("\n🔍 Testing service info...")
    try:
        response = requests.get(f"{BASE_URL}/", timeout=TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Service info: {data['service']} v{data['version']}")
            print(f"   Status: {data['status']}")
            return True
        else:
            print(f"❌ Service info failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Service info error: {e}")
        return False

def test_agents_list():
    """Test the agents endpoint"""
    print("\n🔍 Testing agents list...")
    try:
        response = requests.get(f"{BASE_URL}/agents", timeout=TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            agents = data['agents']
            print(f"✅ Found {len(agents)} agents:")
            for agent in agents:
                status = "✅" if agent['available'] else "❌"
                print(f"   {status} {agent['name']}")
            return True
        else:
            print(f"❌ Agents list failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Agents list error: {e}")
        return False

def test_content_analysis():
    """Test the analyze endpoint"""
    print("\n🔍 Testing content analysis...")
    try:
        test_content = {
            "content": "URGENT: BVAS machines showing errors in Lagos! Voters being turned away!",
            "source_platform": "twitter",
            "request_type": "narrative_classification"
        }
        
        response = requests.post(
            f"{BASE_URL}/analyze", 
            json=test_content,
            timeout=TIMEOUT
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Analysis completed: {data['status']}")
            print(f"   Content preview: {data['content_preview'][:50]}...")
            if data.get('analysis'):
                analysis = data['analysis']
                print(f"   Risk level: {analysis.get('risk_assessment', {}).get('level', 'unknown')}")
                print(f"   Content type: {analysis.get('content_type', 'unknown')}")
            return True
        else:
            print(f"❌ Analysis failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        return False

def test_finding_submission():
    """Test the submit_finding endpoint"""
    print("\n🔍 Testing finding submission...")
    try:
        test_finding = {
            "content_text": "Irregularities observed at polling unit 001 in Ikeja",
            "source_platform": "whatsapp",
            "source_url": "https://chat.whatsapp.com/example",
            "author_handle": "observer123"
        }
        
        response = requests.post(
            f"{BASE_URL}/submit_finding",
            json=test_finding,
            timeout=TIMEOUT
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Finding submitted: {data['status']}")
            print(f"   Finding ID: {data.get('finding_id', 'N/A')}")
            print(f"   Platform: {data['source_platform']}")
            return True
        else:
            print(f"❌ Finding submission failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Finding submission error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting ElectionWatch Simple API Tests")
    print("=" * 50)
    
    # Check if service is running
    print("⏳ Checking if service is running...")
    try:
        requests.get(BASE_URL, timeout=5)
        print("✅ Service is running")
    except:
        print("❌ Service is not running. Please start it with:")
        print("   python simple_agent_api.py")
        sys.exit(1)
    
    # Run tests
    tests = [
        test_health_check,
        test_service_info,
        test_agents_list,
        test_content_analysis,
        test_finding_submission
    ]
    
    results = []
    for test in tests:
        time.sleep(1)  # Brief pause between tests
        results.append(test())
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Your API is working correctly.")
        return 0
    else:
        print(f"❌ {total - passed} test(s) failed. Check the output above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 