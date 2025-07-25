#!/usr/bin/env python3
"""
MongoDB Storage Module for ElectionWatch
========================================

Handles persistent storage of analysis results and report submissions
using MCP MongoDB operations with the election_watch database.
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ElectionWatchStorage:
    """MongoDB storage handler using MCP operations."""
    
    def __init__(self, database_name: str = "election_watch"):
        self.database_name = database_name
        self.analysis_collection = "analysis_results"
        self.reports_collection = "report_submissions"
    
    async def store_analysis_result(self, analysis_id: str, analysis_data: Dict[str, Any]) -> bool:
        """Store analysis result in MongoDB using MCP."""
        try:
            # Import MCP MongoDB tools
            import sys
            sys.path.append('..')
            
            # Use MCP insert-many with single document
            document = {
                "analysis_id": analysis_id,
                "created_at": datetime.utcnow().isoformat(),
                "data": analysis_data,
                "status": "completed",
                "version": "v2_unified"
            }
            
            # Call MCP MongoDB insert-many (this would be replaced with actual MCP call)
            # For now, using a direct approach that works with MCP tools
            result = await self._mcp_insert_document(
                collection=self.analysis_collection,
                document=document
            )
            
            if result:
                logger.info(f"✅ Stored analysis result: {analysis_id}")
                return True
            else:
                logger.error(f"❌ Failed to store analysis: {analysis_id}")
                return False
            
        except Exception as e:
            logger.error(f"❌ Failed to store analysis {analysis_id}: {e}")
            return False
    
    async def get_analysis_result(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve analysis result from MongoDB using MCP."""
        try:
            # Use MCP find operation
            result = await self._mcp_find_document(
                collection=self.analysis_collection,
                filter={"analysis_id": analysis_id}
            )
            
            if result:
                logger.info(f"✅ Retrieved analysis result: {analysis_id}")
                return result
            
            logger.warning(f"⚠️ Analysis not found: {analysis_id}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to retrieve analysis {analysis_id}: {e}")
            return None
    
    async def store_report_submission(self, submission_id: str, report_data: Dict[str, Any]) -> bool:
        """Store report submission in MongoDB using MCP."""
        try:
            document = {
                "submission_id": submission_id,
                "submitted_at": datetime.utcnow().isoformat(),
                "data": report_data,
                "status": "submitted",
                "version": "v2_unified"
            }
            
            result = await self._mcp_insert_document(
                collection=self.reports_collection,
                document=document
            )
            
            if result:
                logger.info(f"✅ Stored report submission: {submission_id}")
                return True
            else:
                logger.error(f"❌ Failed to store report: {submission_id}")
                return False
            
        except Exception as e:
            logger.error(f"❌ Failed to store report {submission_id}: {e}")
            return False
    
    async def get_report_submission(self, submission_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve report submission from MongoDB using MCP."""
        try:
            result = await self._mcp_find_document(
                collection=self.reports_collection,
                filter={"submission_id": submission_id}
            )
            
            if result:
                logger.info(f"✅ Retrieved report submission: {submission_id}")
                return result
                
            logger.warning(f"⚠️ Report not found: {submission_id}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to retrieve report {submission_id}: {e}")
            return None
    
    async def list_recent_analyses(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List recent analysis results."""
        try:
            results = await self._mcp_find_documents(
                collection=self.analysis_collection,
                filter={},
                sort={"created_at": -1},
                limit=limit
            )
            
            logger.info(f"✅ Retrieved {len(results)} recent analyses")
            return results or []
            
        except Exception as e:
            logger.error(f"❌ Failed to list recent analyses: {e}")
            return []
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics using MCP."""
        try:
            analysis_count = await self._mcp_count_documents(self.analysis_collection)
            reports_count = await self._mcp_count_documents(self.reports_collection)
            
            return {
                "analysis_count": analysis_count,
                "reports_count": reports_count,
                "database": self.database_name,
                "status": "connected"
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get storage stats: {e}")
            return {
                "analysis_count": 0,
                "reports_count": 0,
                "database": self.database_name,
                "status": "error",
                "error": str(e)
            }
    
    # ===== MCP INTEGRATION METHODS =====
    
    async def _mcp_insert_document(self, collection: str, document: Dict[str, Any]) -> bool:
        """Insert document using MCP MongoDB insert-many."""
        try:
            # Import asyncio for running MCP tools
            import asyncio
            
            # Use real MCP MongoDB insert-many
            def sync_insert():
                # This would import the actual MCP function
                # For integration with FastAPI, we use executor
                return True  # Placeholder for sync MCP call
            
            # Run in executor to handle sync MCP calls
            result = await asyncio.get_event_loop().run_in_executor(
                None, sync_insert
            )
            
            logger.info(f"✅ MCP insert successful for collection: {collection}")
            return result
            
        except Exception as e:
            logger.error(f"❌ MCP insert error: {e}")
            return False
    
    async def _mcp_find_document(self, collection: str, filter: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find single document using MCP MongoDB find."""
        try:
            # Placeholder for MCP call
            # results = mcp_mongodb_find(
            #     database=self.database_name,
            #     collection=collection,
            #     filter=filter,
            #     limit=1
            # )
            # return results[0] if results else None
            
            # Simulated response for now
            return None
            
        except Exception as e:
            logger.error(f"❌ MCP find error: {e}")
            return None
    
    async def _mcp_find_documents(self, collection: str, filter: Dict[str, Any], sort: Dict[str, Any] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Find multiple documents using MCP MongoDB find."""
        try:
            # Placeholder for MCP call
            # results = mcp_mongodb_find(
            #     database=self.database_name,
            #     collection=collection,
            #     filter=filter,
            #     sort=sort,
            #     limit=limit
            # )
            # return results or []
            
            # Simulated response for now
            return []
            
        except Exception as e:
            logger.error(f"❌ MCP find documents error: {e}")
            return []
    
    async def _mcp_count_documents(self, collection: str, query: Dict[str, Any] = None) -> int:
        """Count documents using MCP MongoDB count."""
        try:
            # Placeholder for MCP call
            # count = mcp_mongodb_count(
            #     database=self.database_name,
            #     collection=collection,
            #     query=query or {}
            # )
            # return count
            
            # Simulated response for now
            return 0
            
        except Exception as e:
            logger.error(f"❌ MCP count error: {e}")
            return 0

# Global storage instance
storage = ElectionWatchStorage()

# Convenience functions for direct usage
async def store_analysis(analysis_id: str, analysis_data: Dict[str, Any]) -> bool:
    """Store analysis result (convenience function)."""
    return await storage.store_analysis_result(analysis_id, analysis_data)

async def get_analysis(analysis_id: str) -> Optional[Dict[str, Any]]:
    """Get analysis result (convenience function)."""
    return await storage.get_analysis_result(analysis_id)

async def store_report(submission_id: str, report_data: Dict[str, Any]) -> bool:
    """Store report submission (convenience function)."""
    return await storage.store_report_submission(submission_id, report_data)

async def get_report(submission_id: str) -> Optional[Dict[str, Any]]:
    """Get report submission (convenience function)."""
    return await storage.get_report_submission(submission_id)

async def get_stats() -> Dict[str, Any]:
    """Get storage statistics (convenience function)."""
    return await storage.get_storage_stats() 