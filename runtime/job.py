"""Job definition for ML inference workloads."""

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


@dataclass
class Job:
    """Represents a single ML inference job with scheduling metadata."""
    
    job_id: str
    class_type: str  # "interactive" or "batch"
    slo_ms: float  # Service Level Objective in milliseconds
    priority: int  # Lower number = higher priority
    workload_name: str  # e.g., "resnet18", "resnet50", "vit"
    backend: str  # "pytorch" or "onnx"
    precision: str  # "fp32" or "fp16"
    batch_size: int
    created_time: float  # Unix timestamp when job was created
    
    # Trace fields (filled during execution)
    enqueue_time: Optional[float] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    # Metrics (computed after execution)
    wait_ms: Optional[float] = None  # Time spent in queue
    service_ms: Optional[float] = None  # Actual inference time
    e2e_ms: Optional[float] = None  # End-to-end latency
    throughput_sps: Optional[float] = None  # Samples per second
    slo_violation: Optional[bool] = None  # True if e2e_ms > slo_ms
    
    # Resource tracking
    resource_cost_tokens: Optional[int] = None
    
    # Concurrency simulation
    lane_id: Optional[int] = None
    
    # Predictor
    predicted_service_ms: Optional[float] = None
    
    def compute_metrics(self) -> None:
        """Compute wait time, e2e latency, throughput, and SLO violation."""
        if self.start_time is None or self.end_time is None:
            return
        
        if self.enqueue_time is not None:
            self.wait_ms = (self.start_time - self.enqueue_time) * 1000
        
        if self.service_ms is None and self.start_time is not None and self.end_time is not None:
            self.service_ms = (self.end_time - self.start_time) * 1000
        
        if self.enqueue_time is not None and self.end_time is not None:
            self.e2e_ms = (self.end_time - self.enqueue_time) * 1000
        
        if self.service_ms is not None and self.service_ms > 0:
            self.throughput_sps = (self.batch_size / self.service_ms) * 1000
        
        if self.e2e_ms is not None and self.slo_ms is not None:
            self.slo_violation = self.e2e_ms > self.slo_ms
    
    def get_deadline(self) -> float:
        """Compute deadline as enqueue_time + slo_ms."""
        if self.enqueue_time is None:
            return float('inf')
        return self.enqueue_time + (self.slo_ms / 1000.0)
    
    def time_to_deadline(self, current_time: float) -> float:
        """Returns time remaining until deadline in seconds (can be negative)."""
        deadline = self.get_deadline()
        return deadline - current_time



