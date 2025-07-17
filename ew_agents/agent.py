"""
ElectionWatch Multi-Agent System

This module contains the coordinator agent that orchestrates specialist agents
for election monitoring and misinformation detection.
"""

import sys
import os

# Handle imports for both local development and deployed environments
try:
    from google.adk.agents.llm_agent import LlmAgent
except ImportError:
    from google.adk.agents.llm_agent import Agent as LlmAgent

# Add multiple possible paths to handle both local and deployed environments
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)

# Add paths for different deployment scenarios
sys.path.insert(0, current_dir)     # Current directory
sys.path.insert(0, parent_dir)      # Parent directory (ml/)
sys.path.insert(0, project_root)    # Project root
sys.path.insert(0, '.')             # Current working directory
sys.path.insert(0, '/app')          # Common container path

# Try different import approaches for the coordinator agent
coordinator_agent = None

try:
    # Try relative import first (package context)
    from .election_watch_agents import coordinator_agent  
except ImportError:
    try:
        # Try direct import (deployed environment)
        from election_watch_agents import coordinator_agent
    except ImportError:
        try:
            # Try absolute import with ew_agents prefix
            from ew_agents.election_watch_agents import coordinator_agent
        except ImportError:
            # Fallback: create a simple mock agent for testing
            print("Warning: Could not import coordinator_agent, creating mock agent")
            
            class MockCoordinatorAgent:
                def __init__(self):
                    self.name = "MockCoordinatorAgent"
                    self.description = "Mock agent for testing deployment"
                
                def process(self, request):
                    return {"status": "mock_response", "message": "Agent deployment successful but using mock mode"}
            
            coordinator_agent = MockCoordinatorAgent()

# The ADK will look for an agent named 'root_agent' in this file
# to start the execution.
root_agent = coordinator_agent