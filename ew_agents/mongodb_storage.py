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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ElectionWatchStorage:
    """MongoDB storage handler with seamless MCP integration."""
    
    def __init__(self, database_name: str = "election_watch", mongo_uri: str = None):
        self.database_name = database_name
        self.analysis_collection = "analysis_results"
        self.reports_collection = "report_submissions"
        
        # Get MongoDB URI from environment (.env file) - prefer MONGODB_ATLAS_URI
        self.mongo_uri = mongo_uri or os.getenv(
            "MONGODB_ATLAS_URI", 
            os.getenv("MONGODB_URI", "mongodb+srv://ew_ml:moHsc5i6gYFrLsvL@ewcluster1.fpkzpxg.mongodb.net/")
        )
        
        # Check if we're in development mode (for SSL certificate handling)
        self.development_mode = os.getenv("MONGODB_DEVELOPMENT_MODE", "false").lower() == "true"
        
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
                    # Connect to MongoDB Atlas with Cloud Run compatible SSL settings
                    try:
                        # Use Cloud Run compatible SSL settings
                        self.client = MongoClient(
                            self.mongo_uri,
                            tls=True,  # Enable TLS for Atlas
                            tlsAllowInvalidCertificates=False,
                            tlsInsecure=False,
                            serverSelectionTimeoutMS=30000,  # 30 second timeout
                            connectTimeoutMS=30000,  # 30 second connection timeout
                            socketTimeoutMS=30000,   # 30 second socket timeout
                            maxPoolSize=10,          # Connection pool size
                            retryWrites=True,        # Enable retryable writes
                            # Cloud Run specific SSL settings
                            ssl=True,
                            ssl_cert_reqs='CERT_NONE',  # Don't verify certificates in Cloud Run
                            ssl_ca_certs=None,          # Don't use CA certs
                            directConnection=False      # Use replica set discovery
                        )
                        
                        # Test the connection immediately
                        self.client.admin.command('ping')
                        logger.info("✅ MongoDB Atlas connected with Cloud Run compatible settings")
                        
                    except Exception as ssl_error:
                        logger.warning(f"⚠️ Cloud Run SSL connection failed: {ssl_error}")
                        logger.info("🔄 Trying with minimal SSL settings...")
                        
                        # Fallback with minimal SSL settings for Cloud Run
                        self.client = MongoClient(
                            self.mongo_uri,
                            tls=True,  # Enable TLS for Atlas
                            tlsAllowInvalidCertificates=True,  # Allow invalid certificates
                            tlsInsecure=True,  # Insecure mode for Cloud Run
                            serverSelectionTimeoutMS=30000,  # Longer timeout
                            connectTimeoutMS=30000,  # Longer connection timeout
                            socketTimeoutMS=30000,   # 30 second socket timeout
                            maxPoolSize=10,          # Connection pool size
                            retryWrites=True,        # Enable retryable writes
                            # Minimal SSL settings
                            ssl=True,
                            ssl_cert_reqs='CERT_NONE',
                            ssl_ca_certs=None,
                            directConnection=False
                        )
                        
                        # Test the fallback connection
                        try:
                            self.client.admin.command('ping')
                            logger.info("✅ MongoDB Atlas connected with minimal SSL settings")
                        except Exception as fallback_error:
                            logger.error(f"❌ Minimal SSL connection also failed: {fallback_error}")
                            # Try one more time with completely disabled SSL verification
                            try:
                                logger.info("🔄 Trying with SSL verification completely disabled...")
                                self.client = MongoClient(
                                    self.mongo_uri,
                                    tls=False,  # Disable TLS completely
                                    ssl=False,  # Disable SSL completely
                                    serverSelectionTimeoutMS=30000,
                                    connectTimeoutMS=30000,
                                    socketTimeoutMS=30000,
                                    maxPoolSize=10,
                                    retryWrites=True,
                                    directConnection=False
                                )
                                self.client.admin.command('ping')
                                logger.info("✅ MongoDB Atlas connected with SSL disabled")
                            except Exception as final_error:
                                logger.error(f"❌ All connection attempts failed: {final_error}")
                                raise final_error
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
            
            return {
                "status": "connected" if connection_status == "connected" else "partial",
                "connection": connection_status,
                "database_access": db_access,
                "collection_operations": collection_ops,
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

    async def _mcp_find_documents(self, collection: str, filter: Dict[str, Any], sort: Dict[str, Any] = None, limit: int = 10) -> List[Dict[str, Any]]:
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

    async def _mcp_count_documents(self, collection: str, query: Dict[str, Any] = None) -> int:
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