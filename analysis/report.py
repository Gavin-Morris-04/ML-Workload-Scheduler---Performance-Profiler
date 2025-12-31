"""Report generation utilities."""

import json
from pathlib import Path
from typing import Dict, Any


def load_summary(summary_path: str) -> Dict[str, Any]:
    """Load summary JSON."""
    with open(summary_path, 'r') as f:
        return json.load(f)


def print_summary(summary: Dict[str, Any]):
    """Print a formatted summary to console."""
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"Scheduler: {summary.get('scheduler_name', 'Unknown')}")
    print(f"Total Jobs: {summary.get('total_jobs', 0)}")
    print(f"Total Samples: {summary.get('total_samples', 0)}")
    
    print("\n--- Overall Metrics ---")
    print(f"Throughput (samples/sec): {summary.get('all_throughput_sps', 0):.2f}")
    print(f"Throughput (jobs/sec): {summary.get('overall_throughput_jobs_sec', 0):.2f}")
    print(f"Avg E2E Latency: {summary.get('all_avg_e2e_ms', 0):.2f} ms")
    print(f"P50: {summary.get('all_p50_e2e_ms', 0):.2f} ms")
    print(f"P95: {summary.get('all_p95_e2e_ms', 0):.2f} ms")
    print(f"P99: {summary.get('all_p99_e2e_ms', 0):.2f} ms")
    
    print("\n--- Interactive Jobs ---")
    print(f"Count: {summary.get('interactive_count', 0)}")
    print(f"Avg E2E: {summary.get('interactive_avg_e2e_ms', 0):.2f} ms")
    print(f"P99: {summary.get('interactive_p99_e2e_ms', 0):.2f} ms")
    print(f"SLO Violation Rate: {summary.get('interactive_slo_violation_rate', 0)*100:.2f}%")
    
    print("\n--- Batch Jobs ---")
    print(f"Count: {summary.get('batch_count', 0)}")
    print(f"Avg E2E: {summary.get('batch_avg_e2e_ms', 0):.2f} ms")
    print(f"P99: {summary.get('batch_p99_e2e_ms', 0):.2f} ms")
    print(f"SLO Violation Rate: {summary.get('batch_slo_violation_rate', 0)*100:.2f}%")
    
    print("\n" + "="*60 + "\n")



