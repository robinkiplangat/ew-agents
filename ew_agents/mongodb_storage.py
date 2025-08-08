#!/usr/bin/env python3
"""
MongoDB Storage Module for ElectionWatch
========================================

Handles persistent storage of analysis results and report submissions
using MCP MongoDB operations with the election_watch database.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import os

# Use pymongo for direct MongoDB access
try:
    from pymongo import MongoClient
    from pymongo.errors import PyMongoError
    import pymongo
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False
    print("Warning: pymongo not available. Install with: pip install pymongo")

# Import Secret Manager utilities
try:
    from .secret_manager import get_mongodb_uri
    SECRET_MANAGER_AVAILABLE = True
except ImportError:
    SECRET_MANAGER_AVAILABLE = False
    get_mongodb_uri = None

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ElectionWatchStorage:
    """MongoDB storage handler with seamless MCP integration."""
    
    def __init__(self, database_name: str = "election_watch", mongo_uri: Optional[str] = None):
        self.database_name = database_name
        self.analysis_collection = "analysis_results"
        self.reports_collection = "report_submissions"
        
        # Get MongoDB URI from Secret Manager or environment variables
        if mongo_uri:
            self.mongo_uri = mongo_uri
        elif SECRET_MANAGER_AVAILABLE and get_mongodb_uri:
            # Try to get from Secret Manager first
            self.mongo_uri = get_mongodb_uri()
            if not self.mongo_uri:
                # Fallback to environment variables
                self.mongo_uri = os.getenv("MONGODB_ATLAS_URI") or os.getenv("MONGODB_URI")
        else:
            # Fallback to environment variables only
            self.mongo_uri = os.getenv("MONGODB_ATLAS_URI") or os.getenv("MONGODB_URI")
        
        # Check if we're in development mode (for SSL certificate handling)
        self.development_mode = os.getenv("MONGODB_DEVELOPMENT_MODE", "false").lower() == "true"
        
        # Check if we're running on Cloud Run
        self.cloud_run_mode = os.getenv("K_SERVICE") is not None or os.getenv("CLOUD_RUN_MODE", "false").lower() == "true"
        
        if self.cloud_run_mode:
            logger.info("☁️ Cloud Run environment detected - using optimized settings")
        elif self.development_mode:
            logger.info("🔧 Development mode detected - using relaxed SSL settings")
        else:
            logger.info("🏠 Local environment detected - using standard settings")
        
        # Debug logging (without exposing credentials)
        if self.mongo_uri:
            if "<username>" in self.mongo_uri:
                logger.warning("🔍 MongoDB URI contains placeholder values - check your .env file")
            else:
                # Show connection type without credentials
                if self.mongo_uri.startswith("mongodb+srv://"):
                    cluster_part = self.mongo_uri.split("@")[1] if "@" in self.mongo_uri else "unknown"
                    logger.info(f"🔍 Attempting MongoDB Atlas connection to: {cluster_part}")
                elif self.mongo_uri.startswith("mongodb://"):
                    logger.info("🔍 Attempting MongoDB connection (non-Atlas)")
                else:
                    logger.warning("🔍 Invalid MongoDB URI format")
        else:
            logger.error("🔍 No MongoDB URI found in environment variables")
        
        # Check if pymongo is available
        if not PYMONGO_AVAILABLE:
            logger.error("❌ pymongo not available - storage will be disabled")
            logger.info("💡 Install with: pip install pymongo")
            self.client = None
            self.db = None
            return
        
        logger.info(f"🔍 pymongo version available: {pymongo.version}")
        
        if PYMONGO_AVAILABLE:
            try:
                # For MongoDB Atlas, we need to handle SSL and authentication
                if self.mongo_uri.startswith("mongodb+srv://") or self.mongo_uri.startswith("mongodb://"):
                    if "<username>" in self.mongo_uri or "<password>" in self.mongo_uri or "<cluster>" in self.mongo_uri:
                        logger.error("❌ MongoDB Atlas URI not properly configured. Please set MONGODB_ATLAS_URI environment variable.")
                        logger.info("💡 Example: MONGODB_ATLAS_URI='mongodb+srv://username:password@cluster.mongodb.net/'")
                        logger.info("💡 Make sure to replace <username>, <password>, and <cluster> with actual values")
                        self.client = None
                        self.db = None
                        return
                    
                    logger.info("🔗 Creating MongoDB Atlas client...")
                    # Connect to MongoDB Atlas with proper SSL settings
                    try:
                        # First try with strict SSL settings (local development)
                        if not self.cloud_run_mode:
                            self.client = MongoClient(
                                self.mongo_uri,
                                tls=True,  # Enable TLS for Atlas
                                tlsAllowInvalidCertificates=False,
                                serverSelectionTimeoutMS=5000,  # 5 second timeout
                                connectTimeoutMS=10000,  # 10 second connection timeout
                                socketTimeoutMS=30000,   # 30 second socket timeout
                                maxPoolSize=10,          # Connection pool size
                                retryWrites=True         # Enable retryable writes
                            )
                            
                            # Test the connection immediately
                            self.client.admin.command('ping')
                            logger.info("✅ MongoDB Atlas connected with strict SSL settings")
                        else:
                            # Cloud Run optimized settings
                            self.client = MongoClient(
                                self.mongo_uri,
                                tls=True,  # Enable TLS for Atlas
                                tlsAllowInvalidCertificates=True,  # Allow invalid certificates for Cloud Run
                                serverSelectionTimeoutMS=10000,  # Longer timeout for Cloud Run
                                connectTimeoutMS=15000,  # Longer connection timeout
                                socketTimeoutMS=30000,   # 30 second socket timeout
                                maxPoolSize=5,           # Smaller pool for serverless
                                retryWrites=True,        # Enable retryable writes
                                maxIdleTimeMS=60000,     # Close idle connections after 1 minute
                                maxConnecting=1          # Single connection attempt for serverless
                            )
                            
                            # Test the connection immediately
                            self.client.admin.command('ping')
                            logger.info("✅ MongoDB Atlas connected with Cloud Run optimized settings")
                        
                    except Exception as ssl_error:
                        logger.warning(f"⚠️ Primary connection failed: {ssl_error}")
                        logger.info("🔄 Trying with relaxed SSL settings...")
                        
                        # Fallback with relaxed SSL settings for development
                        self.client = MongoClient(
                            self.mongo_uri,
                            tls=True,  # Enable TLS for Atlas
                            tlsAllowInvalidCertificates=True,  # Allow invalid certificates for dev
                            serverSelectionTimeoutMS=10000,  # Longer timeout
                            connectTimeoutMS=15000,  # Longer connection timeout
                            socketTimeoutMS=30000,   # 30 second socket timeout
                            maxPoolSize=10,          # Connection pool size
                            retryWrites=True         # Enable retryable writes
                        )
                        
                        # Test the fallback connection
                        try:
                            self.client.admin.command('ping')
                            logger.info("✅ MongoDB Atlas connected with relaxed SSL settings")
                        except Exception as fallback_error:
                            logger.error(f"❌ Fallback SSL connection also failed: {fallback_error}")
                            raise fallback_error
                else:
                    logger.info("🔗 Creating MongoDB client (local/custom)...")
                    # Fallback for local MongoDB (if someone overrides the URI)
                    self.client = MongoClient(self.mongo_uri)
                
                self.db = self.client[self.database_name]
                
                logger.info("🔗 Testing MongoDB connection...")
                # Test connection with Atlas-friendly ping
                try:
                    self.client.admin.command('ping')
                    logger.info(f"✅ MongoDB Atlas connected successfully: {self.database_name}")
                    
                    # Test database access
                    self.db.list_collection_names()
                    logger.info(f"✅ Database access confirmed: {self.database_name}")
                    
                except Exception as ping_error:
                    logger.error(f"❌ MongoDB connection test failed: {ping_error}")
                    logger.info("💡 Connection established but ping failed - this may be a permissions issue")
                    # Continue anyway as the connection might still work for basic operations
                
            except Exception as e:
                logger.error(f"❌ MongoDB Atlas connection failed: {type(e).__name__}: {e}")
                if "authentication failed" in str(e).lower():
                    logger.info("💡 Check your username and password in MONGODB_ATLAS_URI")
                elif "network" in str(e).lower() or "timeout" in str(e).lower():
                    logger.info("💡 Check your network connection and cluster availability")
                elif "ssl" in str(e).lower() or "tls" in str(e).lower():
                    logger.info("💡 TLS/SSL connection issue - check your Atlas cluster configuration")
                else:
                    logger.info("💡 Make sure your MONGODB_ATLAS_URI is set correctly for MongoDB Atlas")
                logger.info("💡 Example: MONGODB_ATLAS_URI='mongodb+srv://username:password@cluster.mongodb.net/'")
                self.client = None
                self.db = None
        else:
            self.client = None
            self.db = None
            logger.error("❌ pymongo not available - storage will be disabled")
            logger.info("💡 Install with: pip install pymongo")

    async def store_analysis_result(self, analysis_id: str, analysis_data: Dict[str, Any]) -> bool:
        """Store analysis result in MongoDB seamlessly."""
        if self.db is None:
            logger.warning("⚠️ MongoDB not available - skipping storage")
            return False
            
        try:
            # Create document with standardized structure
            document = {
                "analysis_id": analysis_id,
                "created_at": datetime.utcnow().isoformat(),
                "data": analysis_data,
                "status": "completed",
                "version": "v2_unified",
                "_id": analysis_id  # Use analysis_id as MongoDB _id for easy retrieval
            }
            
            # Insert into MongoDB
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
        """Retrieve analysis result from MongoDB seamlessly."""
        if self.db is None:
            logger.warning("⚠️ MongoDB not available - cannot retrieve")
            return None
            
        try:
            # Use analysis_id as _id for direct lookup
            result = await self._mcp_find_document(
                collection=self.analysis_collection,
                filter={"_id": analysis_id}
            )
            
            if result:
                logger.info(f"✅ Retrieved analysis result: {analysis_id}")
                return result.get("data", result)  # Return the data field or full document
            
            logger.warning(f"⚠️ Analysis not found: {analysis_id}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to retrieve analysis {analysis_id}: {e}")
            return None

    async def store_report_submission(self, submission_id: str, report_data: Dict[str, Any]) -> bool:
        """Store report submission in MongoDB seamlessly."""
        if self.db is None:
            logger.warning("⚠️ MongoDB not available - skipping storage")
            return False
            
        try:
            document = {
                "submission_id": submission_id,
                "submitted_at": datetime.utcnow().isoformat(),
                "data": report_data,
                "status": "submitted",
                "version": "v2_unified",
                "_id": submission_id  # Use submission_id as MongoDB _id
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
        """Retrieve report submission from MongoDB seamlessly."""
        if self.db is None:
            logger.warning("⚠️ MongoDB not available - cannot retrieve")
            return None
            
        try:
            result = await self._mcp_find_document(
                collection=self.reports_collection,
                filter={"_id": submission_id}
            )
            
            if result:
                logger.info(f"✅ Retrieved report submission: {submission_id}")
                return result.get("data", result)
                
            logger.warning(f"⚠️ Report not found: {submission_id}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to retrieve report {submission_id}: {e}")
            return None

    async def list_recent_analyses(self, limit: int = 10) -> List[Dict[str, Any]]:
        """List recent analysis results."""
        if self.db is None:
            return []
            
        try:
            results = await self._mcp_find_documents(
                collection=self.analysis_collection,
                filter={},
                sort={"created_at": -1},
                limit=limit
            )
            
            logger.info(f"✅ Retrieved {len(results)} recent analyses")
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to list analyses: {e}")
            return []

    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about stored data."""
        if self.db is None:
            return {"error": "MongoDB not available"}
            
        try:
            analysis_count = await self._mcp_count_documents(self.analysis_collection)
            reports_count = await self._mcp_count_documents(self.reports_collection)
            
            stats = {
                "database": self.database_name,
                "collections": {
                    "analysis_results": analysis_count,
                    "report_submissions": reports_count
                },
                "total_documents": analysis_count + reports_count,
                "status": "connected"
            }
            
            logger.info(f"✅ Retrieved storage stats: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"❌ Failed to get stats: {e}")
            return {"error": str(e)}

    async def test_connection(self) -> Dict[str, Any]:
        """Test MongoDB Atlas connection and provide diagnostics."""
        try:
            if self.db is None:
                return {
                    "status": "disconnected",
                    "error": "MongoDB client not initialized",
                    "uri_configured": bool(self.mongo_uri and "<username>" not in self.mongo_uri),
                    "pymongo_available": PYMONGO_AVAILABLE
                }
            
            # Test basic connection
            try:
                self.client.admin.command('ping')
                connection_status = "connected"
            except Exception as ping_error:
                connection_status = f"ping_failed: {str(ping_error)}"
            
            # Test database access
            try:
                collections = self.db.list_collection_names()
                db_access = "accessible"
            except Exception as db_error:
                db_access = f"access_failed: {str(db_error)}"
            
            # Test collection operations
            try:
                test_count = await self._mcp_count_documents(self.analysis_collection, {})
                collection_ops = "working"
            except Exception as coll_error:
                collection_ops = f"failed: {str(coll_error)}"
                test_count = None
            
            return {
                "status": "connected" if connection_status == "connected" else "partial",
                "connection": connection_status,
                "database_access": db_access,
                "collection_operations": collection_ops,
                "test_document_count": test_count,
                "database": self.database_name,
                "collections": collections if db_access == "accessible" else [],
                "uri_configured": bool(self.mongo_uri and "<username>" not in self.mongo_uri),
                "pymongo_available": PYMONGO_AVAILABLE
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "uri_configured": bool(self.mongo_uri and "<username>" not in self.mongo_uri),
                "pymongo_available": PYMONGO_AVAILABLE
            }

    async def _mcp_insert_document(self, collection: str, document: Dict[str, Any]) -> bool:
        """Insert document using pymongo with seamless error handling."""
        if self.db is None:
            return False
            
        try:
            # Use replace_one with upsert=True for seamless updates
            result = self.db[collection].replace_one(
                {"_id": document["_id"]},
                document,
                upsert=True
            )
            
            action = "updated" if result.matched_count > 0 else "inserted"
            logger.info(f"✅ MongoDB {action} document in {collection}: {document['_id']}")
            return True
            
        except PyMongoError as e:
            logger.error(f"❌ MongoDB insert error in {collection}: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Unexpected error in {collection}: {e}")
            return False

    async def _mcp_find_document(self, collection: str, filter: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find single document using pymongo."""
        if self.db is None:
            return None
            
        try:
            result = self.db[collection].find_one(filter)
            if result:
                # Convert ObjectId to string if present
                if "_id" in result and hasattr(result["_id"], "__str__"):
                    result["_id"] = str(result["_id"])
                logger.debug(f"✅ MongoDB find successful in {collection}")
            return result
        except PyMongoError as e:
            logger.error(f"❌ MongoDB find error in {collection}: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Unexpected find error in {collection}: {e}")
            return None

    async def _mcp_find_documents(self, collection: str, filter: Dict[str, Any], sort: Optional[Dict[str, Any]] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Find multiple documents using pymongo."""
        if self.db is None:
            return []
            
        try:
            cursor = self.db[collection].find(filter)
            if sort:
                cursor = cursor.sort(list(sort.items()))
            if limit:
                cursor = cursor.limit(limit)
                
            results = list(cursor)
            
            # Convert ObjectIds to strings
            for result in results:
                if "_id" in result and hasattr(result["_id"], "__str__"):
                    result["_id"] = str(result["_id"])
                    
            logger.debug(f"✅ MongoDB find_many successful in {collection}, count: {len(results)}")
            return results
            
        except PyMongoError as e:
            logger.error(f"❌ MongoDB find_many error in {collection}: {e}")
            return []
        except Exception as e:
            logger.error(f"❌ Unexpected find_many error in {collection}: {e}")
            return []

    async def _mcp_count_documents(self, collection: str, query: Optional[Dict[str, Any]] = None) -> int:
        """Count documents using pymongo."""
        if self.db is None:
            return 0
            
        try:
            count = self.db[collection].count_documents(query or {})
            logger.debug(f"✅ MongoDB count for {collection}: {count}")
            return count
        except PyMongoError as e:
            logger.error(f"❌ MongoDB count error in {collection}: {e}")
            return 0
        except Exception as e:
            logger.error(f"❌ Unexpected count error in {collection}: {e}")
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
    return await storage.get_collection_stats()

async def test_mongodb_connection() -> Dict[str, Any]:
    """Test MongoDB Atlas connection (convenience function)."""
    return await storage.test_connection() 