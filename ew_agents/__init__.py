"""ElectionWatch Agent Package

This package contains the ElectionWatch multi-agent system built on Google ADK.
"""

from . import agent
from .agent import root_agent

__all__ = ['agent', 'root_agent']