"""
Google Secret Manager integration for secure configuration management.
Provides utilities to retrieve sensitive configuration values from Google Secret Manager.
"""

import os
import logging
from typing import Optional
from google.cloud import secretmanager
from google.api_core import exceptions

logger = logging.getLogger(__name__)

class SecretManagerClient:
    """Client for interacting with Google Secret Manager."""
    
    def __init__(self, project_id: Optional[str] = None):
        """
        Initialize the Secret Manager client.
        
        Args:
            project_id: Google Cloud project ID. If None, will use GOOGLE_CLOUD_PROJECT env var.
        """
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        if not self.project_id:
            raise ValueError("Project ID must be provided or GOOGLE_CLOUD_PROJECT environment variable must be set")
        
        try:
            self.client = secretmanager.SecretManagerServiceClient()
            logger.info(f"✅ Secret Manager client initialized for project: {self.project_id}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Secret Manager client: {e}")
            raise
    
    def get_secret(self, secret_id: str, version_id: str = "latest") -> Optional[str]:
        """
        Retrieve a secret from Secret Manager.
        
        Args:
            secret_id: The ID of the secret (without project prefix)
            version_id: The version of the secret (default: "latest")
            
        Returns:
            The secret value as a string, or None if not found
        """
        try:
            # Build the resource name of the secret version
            name = f"projects/{self.project_id}/secrets/{secret_id}/versions/{version_id}"
            
            # Access the secret version
            response = self.client.access_secret_version(request={"name": name})
            
            # Return the secret payload as a string
            secret_value = response.payload.data.decode("UTF-8")
            logger.info(f"✅ Successfully retrieved secret: {secret_id}")
            return secret_value
            
        except exceptions.NotFound:
            logger.warning(f"⚠️ Secret not found: {secret_id}")
            return None
        except exceptions.PermissionDenied:
            logger.error(f"❌ Permission denied accessing secret: {secret_id}")
            return None
        except Exception as e:
            logger.error(f"❌ Error retrieving secret {secret_id}: {e}")
            return None
    
    def create_secret(self, secret_id: str, secret_value: str) -> bool:
        """
        Create a new secret in Secret Manager.
        
        Args:
            secret_id: The ID for the new secret
            secret_value: The value to store in the secret
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Build the parent resource name
            parent = f"projects/{self.project_id}"
            
            # Create the secret
            secret = {"replication": {"automatic": {}}}
            secret_name = f"{parent}/secrets/{secret_id}"
            
            try:
                self.client.create_secret(
                    request={
                        "parent": parent,
                        "secret_id": secret_id,
                        "secret": secret,
                    }
                )
                logger.info(f"✅ Created secret: {secret_id}")
            except exceptions.AlreadyExists:
                logger.info(f"ℹ️ Secret already exists: {secret_id}")
            
            # Add the secret version
            self.client.add_secret_version(
                request={
                    "parent": secret_name,
                    "payload": {"data": secret_value.encode("UTF-8")},
                }
            )
            
            logger.info(f"✅ Added version to secret: {secret_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating secret {secret_id}: {e}")
            return False

def get_mongodb_uri() -> Optional[str]:
    """
    Get MongoDB URI from Secret Manager or environment variable.
    
    Returns:
        MongoDB connection string, or None if not available
    """
    # First try to get from Secret Manager
    try:
        secret_client = SecretManagerClient()
        mongodb_uri = secret_client.get_secret("mongodb-atlas-uri")
        if mongodb_uri:
            logger.info("✅ Retrieved MongoDB URI from Secret Manager")
            return mongodb_uri
    except Exception as e:
        logger.warning(f"⚠️ Could not access Secret Manager: {e}")
    
    # Fallback to environment variable (for local development)
    mongodb_uri = os.getenv("MONGODB_ATLAS_URI")
    if mongodb_uri:
        logger.info("✅ Retrieved MongoDB URI from environment variable")
        return mongodb_uri
    
    logger.error("❌ MongoDB URI not found in Secret Manager or environment variables")
    return None

def setup_secret_manager_secrets() -> bool:
    """
    Set up required secrets in Secret Manager.
    This function can be called during deployment to ensure secrets exist.
    
    Returns:
        True if all secrets are set up successfully, False otherwise
    """
    try:
        secret_client = SecretManagerClient()
        
        # Check if MongoDB URI secret exists
        mongodb_uri = secret_client.get_secret("mongodb-atlas-uri")
        if not mongodb_uri:
            logger.warning("⚠️ MongoDB URI secret not found in Secret Manager")
            logger.info("💡 You can create it manually or set MONGODB_ATLAS_URI environment variable")
            return False
        
        logger.info("✅ All required secrets are available in Secret Manager")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error setting up Secret Manager secrets: {e}")
        return False 