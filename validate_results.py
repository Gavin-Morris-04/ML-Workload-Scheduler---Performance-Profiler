"""Validation script to check experiment results."""

import pandas as pd
import json
from pathlib import Path

def validate_trace(trace_path: str):
    """Validate trace CSV data."""
    df = pd.read_csv(trace_path)
    
    issues = []
    
    # Check for nulls
    if df['enqueue_time'].isna().sum() > 0:
        issues.append(f"Null enqueue_time: {df['enqueue_time'].isna().sum()}")
    if df['start_time'].isna().sum() > 0:
        issues.append(f"Null start_time: {df['start_time'].isna().sum()}")
    if df['end_time'].isna().sum() > 0:
        issues.append(f"Null end_time: {df['end_time'].isna().sum()}")
    
    # Check timing consistency
    if (df['end_time'] < df['start_time']).sum() > 0:
        issues.append(f"end_time < start_time: {(df['end_time'] < df['start_time']).sum()}")
    
    if (df['start_time'] < df['enqueue_time']).sum() > 0:
        issues.append(f"start_time < enqueue_time: {(df['start_time'] < df['enqueue_time']).sum()}")
    
    # Check negative values
    if (df['service_ms'] < 0).sum() > 0:
        issues.append(f"Negative service_ms: {(df['service_ms'] < 0).sum()}")
    if (df['e2e_ms'] < 0).sum() > 0:
        issues.append(f"Negative e2e_ms: {(df['e2e_ms'] < 0).sum()}")
    if (df['wait_ms'] < 0).sum() > 0:
        issues.append(f"Negative wait_ms: {(df['wait_ms'] < 0).sum()}")
    
    # Check throughput
    if (df['throughput_sps'] < 0).sum() > 0:
        issues.append(f"Negative throughput: {(df['throughput_sps'] < 0).sum()}")
    
    if issues:
        print("ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("[OK] All validations passed!")
        return True

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        trace_path = sys.argv[1]
    else:
        trace_path = "results/run_mixed_slo_microbatch/trace.csv"
    
    if not Path(trace_path).exists():
        print(f"Trace file not found: {trace_path}")
        sys.exit(1)
    
    validate_trace(trace_path)

