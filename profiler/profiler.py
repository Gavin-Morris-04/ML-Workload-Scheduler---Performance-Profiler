"""Profiling utilities for ML workloads."""

import torch
import time
from typing import Tuple
from workloads.base import Workload


class Profiler:
    """Profiles workload inference performance."""
    
    def __init__(self, warmup_iterations: int = 5, timed_iterations: int = 30):
        self.warmup_iterations = warmup_iterations
        self.timed_iterations = timed_iterations
    
    def profile(self, workload: Workload, batch_size: int) -> Tuple[float, float]:
        """
        Profile a workload and return mean service time and throughput.
        
        Args:
            workload: The workload to profile
            batch_size: Batch size to use
        
        Returns:
            Tuple of (mean_service_ms, throughput_samples_per_sec)
        """
        device = workload.device
        
        # Warmup
        for _ in range(self.warmup_iterations):
            _ = workload.infer(batch_size)
        
        # Actual timing
        times = []
        for _ in range(self.timed_iterations):
            _, inference_time = workload.infer(batch_size)
            times.append(inference_time * 1000.0)  # Convert to ms
        
        mean_service_ms = sum(times) / len(times)
        throughput_sps = (batch_size / (mean_service_ms / 1000.0)) if mean_service_ms > 0 else 0
        
        return mean_service_ms, throughput_sps


