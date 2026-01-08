"""Monitoring and metrics collection for the RAG learning system."""

import asyncio  # Async programming support for concurrent operations
import time  # Time utilities for performance measurement
from collections import defaultdict, deque
from datetime import datetime, timedelta  # Time utilities for performance measurement
from typing import Any, Callable, Dict, List, Optional, Union  # Type hints for better code documentation
from dataclasses import dataclass, field

from .logging import get_logger  # Structured logging for debugging and monitoring
from .utils import Timer, AsyncTimer

logger = get_logger('monitoring')


@dataclass
class MetricPoint:
    """A single metric data point."""
    timestamp: datetime
    value: Union[int, float]
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricSummary:
    """Summary statistics for a metric."""
    count: int
    sum: float
    min: float
    max: float
    avg: float
    p50: float
    p95: float
    p99: float


class MetricsCollector:
    """Collect and store metrics for the RAG system."""
    
    def __init__(self, max_points_per_metric: int = 10000):
        """
          Init   function implementation.
        """
        self.max_points_per_metric = max_points_per_metric
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_points_per_metric))
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = defaultdict(float)
        self._histograms: Dict[str, List[float]] = defaultdict(list)
    
    def counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        key = self._make_key(name, labels)
        self._counters[key] += value
        self._add_point(name, self._counters[key], labels)
        
        logger.debug(f"Counter {name} incremented", metric_name=name, value=value, labels=labels)
    
    def gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set a gauge metric value."""
        key = self._make_key(name, labels)
        self._gauges[key] = value
        self._add_point(name, value, labels)
        
        logger.debug(f"Gauge {name} set", metric_name=name, value=value, labels=labels)
    
    def histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Add a value to a histogram metric."""
        key = self._make_key(name, labels)
        self._histograms[key].append(value)
        
        # Keep only recent values to prevent memory issues
        if len(self._histograms[key]) > 1000:
            self._histograms[key] = self._histograms[key][-1000:]
        
        self._add_point(name, value, labels)
        
        logger.debug(f"Histogram {name} recorded", metric_name=name, value=value, labels=labels)
    
    def timing(self, name: str, duration: float, labels: Optional[Dict[str, str]] = None):
        """Record a timing metric (convenience method for histogram)."""
        self.histogram(f"{name}_duration_seconds", duration, labels)
    
    def _make_key(self, name: str, labels: Optional[Dict[str, str]] = None) -> str:
        """Create a unique key for metric with labels."""
        if not labels:
            return name
        
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}[{label_str}]"
    
    def _add_point(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Add a metric point to the time series."""
        point = MetricPoint(
            timestamp=datetime.utcnow(),
            value=value,
            labels=labels or {}
        )
        self._metrics[name].append(point)
    
    def get_metric_summary(self, name: str, since: Optional[datetime] = None) -> Optional[MetricSummary]:
        """Get summary statistics for a metric."""
        if name not in self._metrics:
            return None
        
        points = self._metrics[name]
        if since:
            points = [p for p in points if p.timestamp >= since]
        
        if not points:
            return None
        
        values = [p.value for p in points]
        values.sort()
        
        count = len(values)
        total = sum(values)
        
        return MetricSummary(
            count=count,
            sum=total,
            min=values[0],
            max=values[-1],
            avg=total / count,
            p50=values[int(count * 0.5)] if count > 0 else 0,
            p95=values[int(count * 0.95)] if count > 0 else 0,
            p99=values[int(count * 0.99)] if count > 0 else 0
        )
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all current metric values."""
        return {
            'counters': dict(self._counters),
            'gauges': dict(self._gauges),
            'histograms': {k: len(v) for k, v in self._histograms.items()}
        }
    
    def reset_metrics(self):
        """Reset all metrics."""
        self._metrics.clear()
        self._counters.clear()
        self._gauges.clear()
        self._histograms.clear()
        
        logger.info("All metrics reset")


class PerformanceMonitor:
    """Monitor system performance and resource usage."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        """
          Init   function implementation.
        """
        self.metrics = metrics_collector
        self._monitoring = False
        self._monitor_task = None
    
    async def start_monitoring(self, interval: float = 30.0):
        """Start performance monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop(interval))
        
        logger.info("Performance monitoring started", interval=interval)
    
    async def stop_monitoring(self):
        """Stop performance monitoring."""
        if not self._monitoring:
            return
        
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Performance monitoring stopped")
    
    async def _monitor_loop(self, interval: float):
        """Main monitoring loop."""
        while self._monitoring:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(interval)
    
    async def _collect_system_metrics(self):
        """Collect system performance metrics."""
        try:
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics.gauge('system_cpu_percent', cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.metrics.gauge('system_memory_percent', memory.percent)
            self.metrics.gauge('system_memory_used_bytes', memory.used)
            self.metrics.gauge('system_memory_available_bytes', memory.available)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.metrics.gauge('system_disk_percent', (disk.used / disk.total) * 100)
            self.metrics.gauge('system_disk_used_bytes', disk.used)
            self.metrics.gauge('system_disk_free_bytes', disk.free)
            
            # Process info
            process = psutil.Process()
            self.metrics.gauge('process_memory_rss_bytes', process.memory_info().rss)
            self.metrics.gauge('process_memory_vms_bytes', process.memory_info().vms)
            self.metrics.gauge('process_cpu_percent', process.cpu_percent())
            
        except ImportError:
            logger.warning("psutil not available, skipping system metrics")
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")


class RAGMetrics:
    """Specific metrics for RAG operations."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        """
          Init   function implementation.
        """
        self.metrics = metrics_collector
    
    def record_query(
        self,
        query_id: str,
        response_time: float,
        retrieval_time: float,
        generation_time: float,
        results_count: int,
        success: bool = True
    ):
        """Record RAG query metrics."""
        labels = {'query_id': query_id, 'success': str(success)}
        
        self.metrics.counter('rag_queries_total', labels=labels)
        self.metrics.histogram('rag_query_duration_seconds', response_time, labels)
        self.metrics.histogram('rag_retrieval_duration_seconds', retrieval_time, labels)
        self.metrics.histogram('rag_generation_duration_seconds', generation_time, labels)
        self.metrics.histogram('rag_results_count', results_count, labels)
        
        if not success:
            self.metrics.counter('rag_query_errors_total', labels=labels)
    
    def record_document_processing(
        self,
        document_id: str,
        processing_time: float,
        chunk_count: int,
        file_size: int,
        success: bool = True
    ):
        """Record document processing metrics."""
        labels = {'document_id': document_id, 'success': str(success)}
        
        self.metrics.counter('documents_processed_total', labels=labels)
        self.metrics.histogram('document_processing_duration_seconds', processing_time, labels)
        self.metrics.histogram('document_chunks_count', chunk_count, labels)
        self.metrics.histogram('document_size_bytes', file_size, labels)
        
        if not success:
            self.metrics.counter('document_processing_errors_total', labels=labels)
    
    def record_embedding(
        self,
        model_name: str,
        text_length: int,
        embedding_time: float,
        success: bool = True
    ):
        """Record embedding generation metrics."""
        labels = {'model': model_name, 'success': str(success)}
        
        self.metrics.counter('embeddings_generated_total', labels=labels)
        self.metrics.histogram('embedding_duration_seconds', embedding_time, labels)
        self.metrics.histogram('embedding_text_length', text_length, labels)
        
        if not success:
            self.metrics.counter('embedding_errors_total', labels=labels)
    
    def record_llm_call(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        call_time: float,
        success: bool = True
    ):
        """Record LLM API call metrics."""
        labels = {'provider': provider, 'model': model, 'success': str(success)}
        
        self.metrics.counter('llm_calls_total', labels=labels)
        self.metrics.histogram('llm_call_duration_seconds', call_time, labels)
        self.metrics.histogram('llm_prompt_tokens', prompt_tokens, labels)
        self.metrics.histogram('llm_completion_tokens', completion_tokens, labels)
        
        if not success:
            self.metrics.counter('llm_call_errors_total', labels=labels)


def timed_operation(metric_name: str, labels: Optional[Dict[str, str]] = None):
    """Decorator to time operations and record metrics."""
    def decorator(func: Callable):
        """
        Decorator function implementation.
        """
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                """
                Async async wrapper function implementation.
                """
                async with AsyncTimer() as timer:
                    try:
                        result = await func(*args, **kwargs)
                        metrics_collector.timing(metric_name, timer.duration, labels)
                        return result
                    except Exception as e:
                        error_labels = {**(labels or {}), 'error': type(e).__name__}
                        metrics_collector.counter(f"{metric_name}_errors_total", labels=error_labels)
                        raise
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                """
                Sync Wrapper function implementation.
                """
                with Timer() as timer:
                    try:
                        result = func(*args, **kwargs)
                        metrics_collector.timing(metric_name, timer.duration, labels)
                        return result
                    except Exception as e:
                        error_labels = {**(labels or {}), 'error': type(e).__name__}
                        metrics_collector.counter(f"{metric_name}_errors_total", labels=error_labels)
                        raise
            return sync_wrapper
    return decorator


class HealthChecker:
    """Health check system for monitoring service status."""
    
    def __init__(self):
        """
          Init   function implementation.
        """
        self._checks: Dict[str, Callable[[], bool]] = {}
        self._last_results: Dict[str, bool] = {}
        self._last_check_time: Dict[str, datetime] = {}
    
    def register_check(self, name: str, check_func: Callable[[], bool]):
        """Register a health check function."""
        self._checks[name] = check_func
        logger.info(f"Health check registered: {name}")
    
    def run_check(self, name: str) -> bool:
        """Run a specific health check."""
        if name not in self._checks:
            logger.warning(f"Unknown health check: {name}")
            return False
        
        try:
            result = self._checks[name]()
            self._last_results[name] = result
            self._last_check_time[name] = datetime.utcnow()
            
            logger.debug(f"Health check {name}: {'PASS' if result else 'FAIL'}")
            return result
        except Exception as e:
            logger.error(f"Health check {name} failed with exception: {e}")
            self._last_results[name] = False
            self._last_check_time[name] = datetime.utcnow()
            return False
    
    def run_all_checks(self) -> Dict[str, bool]:
        """Run all registered health checks."""
        results = {}
        for name in self._checks:
            results[name] = self.run_check(name)
        return results
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status."""
        all_results = self.run_all_checks()
        
        return {
            'healthy': all(all_results.values()),
            'checks': {
                name: {
                    'status': 'PASS' if result else 'FAIL',
                    'last_check': self._last_check_time.get(name, datetime.utcnow()).isoformat()
                }
                for name, result in all_results.items()
            },
            'timestamp': datetime.utcnow().isoformat()
        }


# Global instances
metrics_collector = MetricsCollector()
performance_monitor = PerformanceMonitor(metrics_collector)
rag_metrics = RAGMetrics(metrics_collector)
health_checker = HealthChecker()


def setup_monitoring():
    """Setup monitoring system."""
    logger.info("Monitoring system initialized")
    
    # Register basic health checks
    health_checker.register_check('metrics_collector', lambda: metrics_collector is not None)
    
    return {
        'metrics_collector': metrics_collector,
        'performance_monitor': performance_monitor,
        'rag_metrics': rag_metrics,
        'health_checker': health_checker
    }