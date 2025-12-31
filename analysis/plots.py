"""Plotting utilities for results visualization."""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import json


def load_trace(trace_path: str) -> pd.DataFrame:
    """Load trace CSV."""
    return pd.read_csv(trace_path)


def plot_latency_cdf(df: pd.DataFrame, output_path: str, scheduler_name: str = ""):
    """Plot CDF of end-to-end latency."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Separate by class type
    for class_type in ['interactive', 'batch']:
        subset = df[df['class_type'] == class_type]
        if len(subset) > 0:
            sorted_latencies = np.sort(subset['e2e_ms'].dropna())
            y = np.arange(1, len(sorted_latencies) + 1) / len(sorted_latencies)
            ax.plot(sorted_latencies, y, label=f'{class_type.capitalize()}', linewidth=2)
    
    ax.set_xlabel('End-to-End Latency (ms)', fontsize=12)
    ax.set_ylabel('Cumulative Probability', fontsize=12)
    ax.set_title(f'Latency CDF - {scheduler_name}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_slo_violations(df: pd.DataFrame, output_path: str, scheduler_name: str = ""):
    """Plot SLO violation rates by class type."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    violation_rates = []
    labels = []
    
    for class_type in ['interactive', 'batch']:
        subset = df[df['class_type'] == class_type]
        if len(subset) > 0:
            violations = subset['slo_violation'].sum()
            total = len(subset)
            rate = (violations / total) * 100 if total > 0 else 0
            violation_rates.append(rate)
            labels.append(class_type.capitalize())
    
    colors = ['#e74c3c' if r > 5 else '#2ecc71' for r in violation_rates]
    bars = ax.bar(labels, violation_rates, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('SLO Violation Rate (%)', fontsize=12)
    ax.set_title(f'SLO Violations - {scheduler_name}', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(violation_rates) * 1.2 if violation_rates else 10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, rate in zip(bars, violation_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_latency_percentiles(df: pd.DataFrame, output_path: str, scheduler_name: str = ""):
    """Plot P50, P95, P99 latency bars."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    percentiles = [50, 95, 99]
    interactive_values = []
    batch_values = []
    
    for p in percentiles:
        interactive_subset = df[df['class_type'] == 'interactive']['e2e_ms'].dropna()
        batch_subset = df[df['class_type'] == 'batch']['e2e_ms'].dropna()
        
        interactive_values.append(np.percentile(interactive_subset, p) if len(interactive_subset) > 0 else 0)
        batch_values.append(np.percentile(batch_subset, p) if len(batch_subset) > 0 else 0)
    
    x = np.arange(len(percentiles))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, interactive_values, width, label='Interactive', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, batch_values, width, label='Batch', color='#9b59b6', alpha=0.8)
    
    ax.set_xlabel('Percentile', fontsize=12)
    ax.set_ylabel('Latency (ms)', fontsize=12)
    ax.set_title(f'Latency Percentiles - {scheduler_name}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'P{p}' for p in percentiles])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_throughput_comparison(summary_paths: List[str], scheduler_names: List[str], output_path: str):
    """Compare throughput across multiple schedulers."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    throughputs_sps = []
    throughputs_jobs = []
    
    for summary_path in summary_paths:
        with open(summary_path, 'r') as f:
            summary = json.load(f)
            throughputs_sps.append(summary.get('all_throughput_sps', 0))
            throughputs_jobs.append(summary.get('overall_throughput_jobs_sec', 0))
    
    # Samples per second
    bars1 = ax1.bar(scheduler_names, throughputs_sps, color='#16a085', alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Throughput (samples/sec)', fontsize=12)
    ax1.set_title('Throughput Comparison (Samples/sec)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars1, throughputs_sps):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.0f}', ha='center', va='bottom', fontsize=10)
    
    # Jobs per second
    bars2 = ax2.bar(scheduler_names, throughputs_jobs, color='#e67e22', alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Throughput (jobs/sec)', fontsize=12)
    ax2.set_title('Throughput Comparison (Jobs/sec)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars2, throughputs_jobs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_wait_time_distribution(df: pd.DataFrame, output_path: str, scheduler_name: str = ""):
    """Plot distribution of wait times."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for class_type in ['interactive', 'batch']:
        subset = df[df['class_type'] == class_type]
        if len(subset) > 0:
            wait_times = subset['wait_ms'].dropna()
            ax.hist(wait_times, bins=50, alpha=0.6, label=f'{class_type.capitalize()}', edgecolor='black')
    
    ax.set_xlabel('Wait Time (ms)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Wait Time Distribution - {scheduler_name}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def generate_all_plots(trace_path: str, summary_path: str, output_dir: str, scheduler_name: str = ""):
    """Generate all plots for a single run."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    df = load_trace(trace_path)
    
    plot_latency_cdf(df, str(output_path / "latency_cdf.png"), scheduler_name)
    plot_slo_violations(df, str(output_path / "slo_violations.png"), scheduler_name)
    plot_latency_percentiles(df, str(output_path / "latency_percentiles.png"), scheduler_name)
    plot_wait_time_distribution(df, str(output_path / "wait_time_dist.png"), scheduler_name)
    
    print(f"Generated plots in {output_path}")


