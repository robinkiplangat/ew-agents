"""
Performance Monitoring and Thread Safety
======================================

This module provides performance monitoring and thread-safe operations
for the ElectionWatch system, addressing the thread-safety issues
identified in the system evaluation.
"""

import time
import threading
import logging
import psutil
import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import weakref

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Structured performance metrics"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    disk_io_read: float
    disk_io_write: float
    network_io_sent: float
    network_io_recv: float
    active_threads: int
    active_connections: int


@dataclass
class OperationMetrics:
    """Metrics for individual operations"""
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None


class ThreadSafeModelManager:
    """
    Thread-safe model manager for global model instances.
    Addresses the thread-safety issues identified in the evaluation.
    """
    
    def __init__(self):
        self._models = {}
        self._locks = {}
        self._initialization_locks = {}
        self._model_configs = {}
    
    def get_model(self, model_name: str, model_config: Optional[Dict[str, Any]] = None):
        """
        Thread-safe model retrieval with lazy loading.
        
        Args:
            model_name: Name of the model to retrieve
            model_config: Configuration for model initialization
            
        Returns:
            The requested model instance
        """
        # Create initialization lock for this model if it doesn't exist
        if model_name not in self._initialization_locks:
            self._initialization_locks[model_name] = threading.Lock()
        
        # Use double-checked locking pattern
        if model_name not in self._models:
            with self._initialization_locks[model_name]:
                if model_name not in self._models:
                    self._models[model_name] = self._initialize_model(model_name, model_config)
        
        return self._models[model_name]
    
    def _initialize_model(self, model_name: str, model_config: Optional[Dict[str, Any]] = None):
        """Initialize a model with the given configuration"""
        try:
            logger.info(f"🔄 Initializing model: {model_name}")
            
            if model_name == "multimodal":
                # Initialize multimodal model
                from transformers import AutoProcessor, AutoModel
                processor = AutoProcessor.from_pretrained("microsoft/git-base")
                model = AutoModel.from_pretrained("microsoft/git-base")
                return {"processor": processor, "model": model}
            
            elif model_name == "whisper":
                # Initialize Whisper model
                import whisper
                model = whisper.load_model("base")
                return model
            
            elif model_name == "sentiment":
                # Initialize sentiment analysis model
                from transformers import pipeline
                model = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
                return model
            
            else:
                logger.warning(f"⚠️ Unknown model type: {model_name}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize model {model_name}: {e}")
            return None
    
    def cleanup_model(self, model_name: str):
        """Clean up a model instance"""
        if model_name in self._models:
            try:
                # Clean up model resources
                model = self._models[model_name]
                if hasattr(model, 'close'):
                    model.close()
                elif hasattr(model, 'model') and hasattr(model['model'], 'close'):
                    model['model'].close()
                
                del self._models[model_name]
                logger.info(f"✅ Cleaned up model: {model_name}")
                
            except Exception as e:
                logger.error(f"❌ Failed to cleanup model {model_name}: {e}")
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all loaded models"""
        return {
            "loaded_models": list(self._models.keys()),
            "total_models": len(self._models),
            "initialization_locks": list(self._initialization_locks.keys())
        }


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system.
    """
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics_history = deque(maxlen=max_history)
        self.operation_history = deque(maxlen=max_history)
        self._monitoring = False
        self._monitor_thread = None
        self._lock = threading.Lock()
        
        # Performance thresholds
        self.thresholds = {
            "cpu_warning": 80.0,
            "cpu_critical": 95.0,
            "memory_warning": 85.0,
            "memory_critical": 95.0,
            "operation_timeout_ms": 30000  # 30 seconds
        }
    
    def start_monitoring(self, interval_seconds: int = 5):
        """Start performance monitoring"""
        if self._monitoring:
            logger.warning("⚠️ Monitoring already active")
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self._monitor_thread.start()
        logger.info(f"✅ Performance monitoring started (interval: {interval_seconds}s)")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("✅ Performance monitoring stopped")
    
    def _monitor_loop(self, interval_seconds: int):
        """Main monitoring loop"""
        while self._monitoring:
            try:
                metrics = self._collect_metrics()
                self._check_thresholds(metrics)
                
                with self._lock:
                    self.metrics_history.append(metrics)
                
                time.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"❌ Monitoring error: {e}")
                time.sleep(interval_seconds)
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current system metrics"""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk_io = psutil.disk_io_counters()
            network_io = psutil.net_io_counters()
            
            # Process metrics
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            # Thread and connection metrics
            active_threads = threading.active_count()
            active_connections = len(psutil.net_connections())
            
            return PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_mb=memory_mb,
                disk_io_read=disk_io.read_bytes if disk_io else 0,
                disk_io_write=disk_io.write_bytes if disk_io else 0,
                network_io_sent=network_io.bytes_sent if network_io else 0,
                network_io_recv=network_io.bytes_recv if network_io else 0,
                active_threads=active_threads,
                active_connections=active_connections
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to collect metrics: {e}")
            return PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_mb=0.0,
                disk_io_read=0.0,
                disk_io_write=0.0,
                network_io_sent=0.0,
                network_io_recv=0.0,
                active_threads=0,
                active_connections=0
            )
    
    def _check_thresholds(self, metrics: PerformanceMetrics):
        """Check performance thresholds and log warnings"""
        warnings = []
        
        if metrics.cpu_percent > self.thresholds["cpu_critical"]:
            warnings.append(f"🚨 CRITICAL: CPU usage {metrics.cpu_percent:.1f}%")
        elif metrics.cpu_percent > self.thresholds["cpu_warning"]:
            warnings.append(f"⚠️ WARNING: CPU usage {metrics.cpu_percent:.1f}%")
        
        if metrics.memory_percent > self.thresholds["memory_critical"]:
            warnings.append(f"🚨 CRITICAL: Memory usage {metrics.memory_percent:.1f}%")
        elif metrics.memory_percent > self.thresholds["memory_warning"]:
            warnings.append(f"⚠️ WARNING: Memory usage {metrics.memory_percent:.1f}%")
        
        for warning in warnings:
            logger.warning(warning)
    
    def track_operation(self, operation_name: str):
        """Context manager for tracking operations"""
        return OperationTracker(self, operation_name)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        with self._lock:
            if not self.metrics_history:
                return {"error": "No metrics available"}
            
            recent_metrics = list(self.metrics_history)[-10:]  # Last 10 metrics
            
            # Calculate averages
            avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
            avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
            avg_memory_mb = sum(m.memory_mb for m in recent_metrics) / len(recent_metrics)
            
            # Calculate trends
            if len(recent_metrics) >= 2:
                cpu_trend = recent_metrics[-1].cpu_percent - recent_metrics[0].cpu_percent
                memory_trend = recent_metrics[-1].memory_percent - recent_metrics[0].memory_percent
            else:
                cpu_trend = 0.0
                memory_trend = 0.0
            
            return {
                "current_metrics": {
                    "cpu_percent": recent_metrics[-1].cpu_percent if recent_metrics else 0.0,
                    "memory_percent": recent_metrics[-1].memory_percent if recent_metrics else 0.0,
                    "memory_mb": recent_metrics[-1].memory_mb if recent_metrics else 0.0,
                    "active_threads": recent_metrics[-1].active_threads if recent_metrics else 0,
                    "active_connections": recent_metrics[-1].active_connections if recent_metrics else 0
                },
                "average_metrics": {
                    "avg_cpu_percent": avg_cpu,
                    "avg_memory_percent": avg_memory,
                    "avg_memory_mb": avg_memory_mb
                },
                "trends": {
                    "cpu_trend": cpu_trend,
                    "memory_trend": memory_trend
                },
                "thresholds": self.thresholds,
                "monitoring_active": self._monitoring,
                "metrics_count": len(self.metrics_history),
                "operations_count": len(self.operation_history)
            }
    
    def get_operation_stats(self) -> Dict[str, Any]:
        """Get operation statistics"""
        with self._lock:
            if not self.operation_history:
                return {"error": "No operation history available"}
            
            operations = list(self.operation_history)
            
            # Group by operation name
            operation_stats = defaultdict(list)
            for op in operations:
                if op.duration_ms is not None:
                    operation_stats[op.operation_name].append(op.duration_ms)
            
            # Calculate statistics
            stats = {}
            for op_name, durations in operation_stats.items():
                stats[op_name] = {
                    "count": len(durations),
                    "avg_duration_ms": sum(durations) / len(durations),
                    "min_duration_ms": min(durations),
                    "max_duration_ms": max(durations),
                    "success_rate": sum(1 for op in operations if op.operation_name == op_name and op.success) / len([op for op in operations if op.operation_name == op_name])
                }
            
            return {
                "operation_stats": dict(stats),
                "total_operations": len(operations),
                "successful_operations": sum(1 for op in operations if op.success),
                "failed_operations": sum(1 for op in operations if not op.success)
            }


class OperationTracker:
    """Context manager for tracking individual operations"""
    
    def __init__(self, monitor: PerformanceMonitor, operation_name: str):
        self.monitor = monitor
        self.operation_name = operation_name
        self.start_time = None
        self.metrics = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.metrics = self.monitor._collect_metrics()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = datetime.now()
        duration_ms = (end_time - self.start_time).total_seconds() * 1000
        
        # Calculate memory usage
        end_metrics = self.monitor._collect_metrics()
        memory_usage_mb = end_metrics.memory_mb - self.metrics.memory_mb if self.metrics else 0.0
        
        # Create operation metrics
        operation_metrics = OperationMetrics(
            operation_name=self.operation_name,
            start_time=self.start_time,
            end_time=end_time,
            duration_ms=duration_ms,
            success=exc_type is None,
            error_message=str(exc_val) if exc_val else None,
            memory_usage_mb=memory_usage_mb,
            cpu_usage_percent=end_metrics.cpu_percent - self.metrics.cpu_percent if self.metrics else 0.0
        )
        
        # Add to operation history
        with self.monitor._lock:
            self.monitor.operation_history.append(operation_metrics)
        
        # Log slow operations
        if duration_ms > self.monitor.thresholds["operation_timeout_ms"]:
            logger.warning(f"⚠️ Slow operation detected: {self.operation_name} took {duration_ms:.1f}ms")


class ResourceManager:
    """
    Resource management for cleanup and optimization.
    """
    
    def __init__(self):
        self._resources = weakref.WeakSet()
        self._cleanup_callbacks = []
    
    def register_resource(self, resource, cleanup_callback=None):
        """Register a resource for cleanup"""
        self._resources.add(resource)
        if cleanup_callback:
            self._cleanup_callbacks.append((resource, cleanup_callback))
    
    def cleanup_resources(self):
        """Clean up all registered resources"""
        logger.info("🧹 Starting resource cleanup")
        
        # Call cleanup callbacks
        for resource, callback in self._cleanup_callbacks:
            try:
                callback(resource)
            except Exception as e:
                logger.error(f"❌ Cleanup callback failed: {e}")
        
        # Clear callbacks
        self._cleanup_callbacks.clear()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        logger.info("✅ Resource cleanup completed")
    
    def get_resource_count(self) -> int:
        """Get number of registered resources"""
        return len(self._resources)


# Global instances
model_manager = ThreadSafeModelManager()
performance_monitor = PerformanceMonitor()
resource_manager = ResourceManager()


# Utility functions
def get_model_safely(model_name: str, model_config: Optional[Dict[str, Any]] = None):
    """Thread-safe model retrieval"""
    return model_manager.get_model(model_name, model_config)


def track_performance(operation_name: str):
    """Decorator for tracking operation performance"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with performance_monitor.track_operation(operation_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def get_performance_metrics() -> Dict[str, Any]:
    """Get current performance metrics"""
    return performance_monitor.get_performance_summary()


def get_operation_stats() -> Dict[str, Any]:
    """Get operation statistics"""
    return performance_monitor.get_operation_stats()


def start_performance_monitoring(interval_seconds: int = 5):
    """Start performance monitoring"""
    performance_monitor.start_monitoring(interval_seconds)


def stop_performance_monitoring():
    """Stop performance monitoring"""
    performance_monitor.stop_monitoring()


def cleanup_resources():
    """Clean up all resources"""
    resource_manager.cleanup_resources()
    model_manager._models.clear() 