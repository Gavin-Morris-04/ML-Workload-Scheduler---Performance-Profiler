"""Trace logging for job execution."""

import csv
import json
import os
from pathlib import Path
from typing import List, Dict, Any
from runtime.job import Job


class TraceLogger:
    """Logs job traces to CSV and generates summary statistics."""
    
    def __init__(self, run_id: str, scheduler_name: str, output_dir: str = "results"):
        self.run_id = run_id
        self.scheduler_name = scheduler_name
        self.output_dir = Path(output_dir) / run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.trace_file = self.output_dir / "trace.csv"
        self.summary_file = self.output_dir / "summary.json"
        
        # Open CSV file for writing
        self.csv_file = open(self.trace_file, 'w', newline='')
        self.writer = csv.DictWriter(self.csv_file, fieldnames=[
            'run_id', 'scheduler_name', 'job_id', 'class_type', 'workload_name',
            'backend', 'precision', 'batch_size', 'enqueue_time', 'start_time',
            'end_time', 'wait_ms', 'service_ms', 'e2e_ms', 'throughput_sps',
            'slo_ms', 'slo_violation', 'lane_id', 'predicted_service_ms',
            'resource_cost_tokens'
        ])
        self.writer.writeheader()
        
        self.completed_jobs: List[Job] = []
    
    def log_job(self, job: Job) -> None:
        """Log a completed job to the trace CSV."""
        job.compute_metrics()
        self.completed_jobs.append(job)
        
        self.writer.writerow({
            'run_id': self.run_id,
            'scheduler_name': self.scheduler_name,
            'job_id': job.job_id,
            'class_type': job.class_type,
            'workload_name': job.workload_name,
            'backend': job.backend,
            'precision': job.precision,
            'batch_size': job.batch_size,
            'enqueue_time': job.enqueue_time,
            'start_time': job.start_time,
            'end_time': job.end_time,
            'wait_ms': job.wait_ms,
            'service_ms': job.service_ms,
            'e2e_ms': job.e2e_ms,
            'throughput_sps': job.throughput_sps,
            'slo_ms': job.slo_ms,
            'slo_violation': 1 if job.slo_violation else 0,
            'lane_id': job.lane_id if job.lane_id is not None else '',
            'predicted_service_ms': job.predicted_service_ms if job.predicted_service_ms is not None else '',
            'resource_cost_tokens': job.resource_cost_tokens if job.resource_cost_tokens is not None else ''
        })
        self.csv_file.flush()
    
    def compute_summary(self) -> Dict[str, Any]:
        """Compute summary statistics from completed jobs."""
        if not self.completed_jobs:
            return {}
        
        # Separate by class type
        interactive_jobs = [j for j in self.completed_jobs if j.class_type == "interactive"]
        batch_jobs = [j for j in self.completed_jobs if j.class_type == "batch"]
        
        def compute_stats(jobs: List[Job], name: str) -> Dict[str, Any]:
            if not jobs:
                return {}
            
            e2e_latencies = [j.e2e_ms for j in jobs if j.e2e_ms is not None]
            wait_times = [j.wait_ms for j in jobs if j.wait_ms is not None]
            service_times = [j.service_ms for j in jobs if j.service_ms is not None]
            slo_violations = sum(1 for j in jobs if j.slo_violation)
            
            def percentile(values: List[float], p: float) -> float:
                if not values:
                    return 0.0
                sorted_vals = sorted(values)
                idx = int(len(sorted_vals) * p / 100.0)
                idx = min(idx, len(sorted_vals) - 1)
                return sorted_vals[idx]
            
            total_samples = sum(j.batch_size for j in jobs)
            total_time = max((j.end_time for j in jobs if j.end_time is not None), default=0) - \
                        min((j.enqueue_time for j in jobs if j.enqueue_time is not None), default=0)
            
            return {
                f"{name}_count": len(jobs),
                f"{name}_total_samples": total_samples,
                f"{name}_avg_e2e_ms": sum(e2e_latencies) / len(e2e_latencies) if e2e_latencies else 0,
                f"{name}_p50_e2e_ms": percentile(e2e_latencies, 50),
                f"{name}_p95_e2e_ms": percentile(e2e_latencies, 95),
                f"{name}_p99_e2e_ms": percentile(e2e_latencies, 99),
                f"{name}_avg_wait_ms": sum(wait_times) / len(wait_times) if wait_times else 0,
                f"{name}_avg_service_ms": sum(service_times) / len(service_times) if service_times else 0,
                f"{name}_slo_violations": slo_violations,
                f"{name}_slo_violation_rate": slo_violations / len(jobs) if jobs else 0,
                f"{name}_throughput_sps": (total_samples / total_time) if total_time > 0 else 0
            }
        
        stats = {
            'run_id': self.run_id,
            'scheduler_name': self.scheduler_name,
            'total_jobs': len(self.completed_jobs),
            'total_samples': sum(j.batch_size for j in self.completed_jobs),
        }
        
        stats.update(compute_stats(self.completed_jobs, 'all'))
        stats.update(compute_stats(interactive_jobs, 'interactive'))
        stats.update(compute_stats(batch_jobs, 'batch'))
        
        # Overall throughput
        if self.completed_jobs:
            total_samples = sum(j.batch_size for j in self.completed_jobs)
            total_time = max((j.end_time for j in self.completed_jobs if j.end_time is not None), default=0) - \
                        min((j.enqueue_time for j in self.completed_jobs if j.enqueue_time is not None), default=0)
            stats['overall_throughput_sps'] = (total_samples / total_time) if total_time > 0 else 0
            stats['overall_throughput_jobs_sec'] = len(self.completed_jobs) / total_time if total_time > 0 else 0
        
        return stats
    
    def finalize(self) -> None:
        """Close files and write summary."""
        self.csv_file.close()
        
        summary = self.compute_summary()
        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2)


