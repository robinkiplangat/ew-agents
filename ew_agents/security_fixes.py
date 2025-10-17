"""
Security Fixes and Enhancements
==============================

This module contains security fixes and enhancements for the ElectionWatch system,
addressing critical vulnerabilities identified in the system evaluation.
"""

import ssl
import logging
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse
import certifi

logger = logging.getLogger(__name__)


class SecureConnectionManager:
    """
    Manages secure connections with proper SSL/TLS configuration.
    Replaces insecure SSL contexts with secure alternatives.
    """
    
    def __init__(self):
        self.default_ssl_context = self._create_secure_ssl_context()
    
    def _create_secure_ssl_context(self) -> ssl.SSLContext:
        """
        Creates a secure SSL context with proper certificate validation.
        Replaces the insecure ssl.create_default_context() with check_hostname=False.
        """
        try:
            # Create secure SSL context with proper certificate validation
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            
            # Ensure proper verification settings
            ssl_context.check_hostname = True
            ssl_context.verify_mode = ssl.CERT_REQUIRED
            
            # Set secure cipher suites
            ssl_context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20')
            
            # Set minimum TLS version to 1.2
            ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
            
            logger.info("✅ Secure SSL context created with proper certificate validation")
            return ssl_context
            
        except Exception as e:
            logger.error(f"❌ Failed to create secure SSL context: {e}")
            # Fallback to default but still secure context
            return ssl.create_default_context()
    
    def get_secure_connection_config(self, url: str) -> Dict[str, Any]:
        """
        Returns secure connection configuration for a given URL.
        
        Args:
            url: The URL to create secure connection for
            
        Returns:
            Dict with secure connection parameters
        """
        parsed_url = urlparse(url)
        
        return {
            "ssl_context": self.default_ssl_context,
            "verify_ssl": True,
            "cert_reqs": ssl.CERT_REQUIRED,
            "check_hostname": True,
            "url": url,
            "scheme": parsed_url.scheme,
            "hostname": parsed_url.hostname,
            "port": parsed_url.port or (443 if parsed_url.scheme == 'https' else 80)
        }
    
    def validate_mongodb_connection_string(self, connection_string: str) -> Dict[str, Any]:
        """
        Validates and secures MongoDB connection string.
        Removes insecure tlsAllowInvalidCertificates option.
        
        Args:
            connection_string: MongoDB connection string
            
        Returns:
            Dict with validation results and secure connection string
        """
        try:
            # Check for insecure options
            insecure_options = [
                "tlsAllowInvalidCertificates=true",
                "sslAllowInvalidCertificates=true",
                "tlsInsecure=true",
                "sslInsecure=true"
            ]
            
            found_insecure = []
            secure_connection = connection_string
            
            for option in insecure_options:
                if option in connection_string:
                    found_insecure.append(option)
                    # Remove insecure option
                    secure_connection = secure_connection.replace(option, "")
                    # Clean up any double ampersands or question marks
                    secure_connection = secure_connection.replace("&&", "&").replace("??", "?")
                    secure_connection = secure_connection.rstrip("&?").rstrip("?&")
            
            if found_insecure:
                logger.warning(f"⚠️ Removed insecure MongoDB options: {found_insecure}")
                logger.info("✅ MongoDB connection string secured")
            
            return {
                "secure": len(found_insecure) == 0,
                "original_connection": connection_string,
                "secure_connection": secure_connection,
                "removed_options": found_insecure,
                "validation_passed": True
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to validate MongoDB connection string: {e}")
            return {
                "secure": False,
                "original_connection": connection_string,
                "secure_connection": connection_string,
                "error": str(e),
                "validation_passed": False
            }


class SecurityAuditor:
    """
    Performs security audits on the system configuration.
    """
    
    def __init__(self):
        self.connection_manager = SecureConnectionManager()
    
    def audit_environment_security(self) -> Dict[str, Any]:
        """
        Performs a comprehensive security audit of the environment.
        
        Returns:
            Dict with security audit results
        """
        audit_results = {
            "ssl_tls_configuration": self._audit_ssl_configuration(),
            "mongodb_security": self._audit_mongodb_security(),
            "api_security": self._audit_api_security(),
            "overall_security_score": 0,
            "critical_issues": [],
            "recommendations": []
        }
        
        # Calculate overall security score
        scores = []
        for category, result in audit_results.items():
            if isinstance(result, dict) and "score" in result:
                scores.append(result["score"])
        
        if scores:
            audit_results["overall_security_score"] = sum(scores) / len(scores)
        
        # Generate recommendations
        audit_results["recommendations"] = self._generate_security_recommendations(audit_results)
        
        return audit_results
    
    def _audit_ssl_configuration(self) -> Dict[str, Any]:
        """Audits SSL/TLS configuration"""
        try:
            ssl_context = self.connection_manager.default_ssl_context
            
            score = 100  # Start with perfect score
            
            # Check for insecure settings
            if not ssl_context.check_hostname:
                score -= 50
            if ssl_context.verify_mode != ssl.CERT_REQUIRED:
                score -= 50
            if ssl_context.minimum_version < ssl.TLSVersion.TLSv1_2:
                score -= 25
            
            return {
                "score": max(0, score),
                "check_hostname_enabled": ssl_context.check_hostname,
                "verify_mode": str(ssl_context.verify_mode),
                "minimum_tls_version": str(ssl_context.minimum_version),
                "secure": score >= 80
            }
        except Exception as e:
            return {
                "score": 0,
                "error": str(e),
                "secure": False
            }
    
    def _audit_mongodb_security(self) -> Dict[str, Any]:
        """Audits MongoDB connection security"""
        import os
        
        connection_string = os.getenv("MONGODB_ATLAS_URI") or os.getenv("MONGODB_URI", "")
        
        if not connection_string:
            return {
                "score": 0,
                "error": "No MongoDB connection string found",
                "secure": False
            }
        
        validation = self.connection_manager.validate_mongodb_connection_string(connection_string)
        
        score = 100 if validation["secure"] else 50
        
        return {
            "score": score,
            "secure": validation["secure"],
            "removed_insecure_options": validation.get("removed_options", []),
            "connection_validated": validation["validation_passed"]
        }
    
    def _audit_api_security(self) -> Dict[str, Any]:
        """Audits API security configuration"""
        import os
        
        score = 100
        
        # Check for secure headers
        cors_origins = os.getenv("ALLOWED_ORIGINS", "*")
        if cors_origins == "*":
            score -= 30
        
        # Check for HTTPS in production
        environment = os.getenv("ENVIRONMENT", "development")
        if environment == "production":
            https_enabled = os.getenv("HTTPS_ENABLED", "false").lower() == "true"
            if not https_enabled:
                score -= 40
        
        return {
            "score": max(0, score),
            "cors_configured": cors_origins != "*",
            "https_enabled": os.getenv("HTTPS_ENABLED", "false").lower() == "true",
            "environment": environment,
            "secure": score >= 80
        }
    
    def _generate_security_recommendations(self, audit_results: Dict[str, Any]) -> List[str]:
        """Generates security recommendations based on audit results"""
        recommendations = []
        
        # SSL/TLS recommendations
        ssl_audit = audit_results.get("ssl_tls_configuration", {})
        if not ssl_audit.get("secure", False):
            recommendations.append("🔒 Fix SSL/TLS configuration - enable certificate validation")
        
        # MongoDB recommendations
        mongo_audit = audit_results.get("mongodb_security", {})
        if not mongo_audit.get("secure", False):
            recommendations.append("🔒 Remove insecure MongoDB connection options")
        
        # API recommendations
        api_audit = audit_results.get("api_security", {})
        if not api_audit.get("secure", False):
            recommendations.append("🔒 Configure CORS properly for production")
            recommendations.append("🔒 Enable HTTPS in production environment")
        
        # General recommendations
        if audit_results.get("overall_security_score", 0) < 80:
            recommendations.append("🔒 Implement comprehensive security testing")
            recommendations.append("🔒 Add rate limiting and input validation")
        
        return recommendations


# Global instances for easy access
secure_connection_manager = SecureConnectionManager()
security_auditor = SecurityAuditor()


def get_secure_ssl_context() -> ssl.SSLContext:
    """Get a secure SSL context for external connections"""
    return secure_connection_manager.default_ssl_context


def validate_and_secure_mongodb_connection(connection_string: str) -> str:
    """Validate and secure MongoDB connection string"""
    validation = secure_connection_manager.validate_mongodb_connection_string(connection_string)
    return validation["secure_connection"]


def perform_security_audit() -> Dict[str, Any]:
    """Perform comprehensive security audit"""
    return security_auditor.audit_environment_security() 