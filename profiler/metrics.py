"""Metrics computation for job execution."""

from typing import List, Dict, Any
from runtime.job import Job


def compute_percentile(values: List[float], percentile: float) -> float:
    """Compute percentile value from a list."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = int(len(sorted_vals) * percentile / 100.0)
    idx = min(idx, len(sorted_vals) - 1)
    return sorted_vals[idx]


def compute_metrics(jobs: List[Job]) -> Dict[str, Any]:
    """Compute comprehensive metrics from completed jobs."""
    if not jobs:
        return {}
    
    interactive_jobs = [j for j in jobs if j.class_type == "interactive"]
    batch_jobs = [j for j in jobs if j.class_type == "batch"]
    
    def compute_stats(job_list: List[Job], prefix: str) -> Dict[str, Any]:
        if not job_list:
            return {}
        
        e2e_latencies = [j.e2e_ms for j in job_list if j.e2e_ms is not None]
        wait_times = [j.wait_ms for j in job_list if j.wait_ms is not None]
        service_times = [j.service_ms for j in job_list if j.service_ms is not None]
        slo_violations = sum(1 for j in job_list if j.slo_violation)
        
        total_samples = sum(j.batch_size for j in job_list)
        total_time = 0.0
        if job_list:
            end_times = [j.end_time for j in job_list if j.end_time is not None]
            start_times = [j.enqueue_time for j in job_list if j.enqueue_time is not None]
            if end_times and start_times:
                total_time = max(end_times) - min(start_times)
        
        return {
            f"{prefix}_count": len(job_list),
            f"{prefix}_total_samples": total_samples,
            f"{prefix}_avg_e2e_ms": sum(e2e_latencies) / len(e2e_latencies) if e2e_latencies else 0,
            f"{prefix}_p50_e2e_ms": compute_percentile(e2e_latencies, 50),
            f"{prefix}_p95_e2e_ms": compute_percentile(e2e_latencies, 95),
            f"{prefix}_p99_e2e_ms": compute_percentile(e2e_latencies, 99),
            f"{prefix}_avg_wait_ms": sum(wait_times) / len(wait_times) if wait_times else 0,
            f"{prefix}_avg_service_ms": sum(service_times) / len(service_times) if service_times else 0,
            f"{prefix}_slo_violations": slo_violations,
            f"{prefix}_slo_violation_rate": slo_violations / len(job_list) if job_list else 0,
            f"{prefix}_throughput_sps": (total_samples / total_time) if total_time > 0 else 0
        }
    
    all_stats = compute_stats(jobs, "all")
    interactive_stats = compute_stats(interactive_jobs, "interactive")
    batch_stats = compute_stats(batch_jobs, "batch")
    
    # Fairness metric: ratio of batch wait to interactive wait
    fairness = 0.0
    if interactive_stats.get("interactive_avg_wait_ms", 0) > 0:
        fairness = batch_stats.get("batch_avg_wait_ms", 0) / interactive_stats["interactive_avg_wait_ms"]
    
    return {
        **all_stats,
        **interactive_stats,
        **batch_stats,
        "fairness_ratio": fairness
    }



