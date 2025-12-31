"""Main experiment runner for ML workload scheduling benchmarks."""

import argparse
import yaml
import random
import time
import uuid
from pathlib import Path
from typing import Dict, List
from datetime import datetime

# Runtime components
from runtime.job import Job
from runtime.job_queue import JobQueue
from runtime.executor import Executor
from runtime.resource_manager import ResourceManager
from runtime.lane_manager import LaneManager
from runtime.trace_logger import TraceLogger

# Workloads
from workloads.resnet18 import ResNet18Workload
from workloads.resnet50 import ResNet50Workload
from workloads.vit import ViTWorkload

# Schedulers
from scheduler.fifo import FIFOScheduler
from scheduler.priority_slo import PrioritySLOScheduler
from scheduler.batch_aware import BatchAwareScheduler
from scheduler.dynamic import DynamicScheduler

# Optimization
from optimization.microbatcher import MicroBatcher

# Analysis
from analysis.plots import generate_all_plots
from analysis.report import print_summary

# Predictor
from models.latency_predictor import LatencyPredictor


def load_config(config_path: str) -> Dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def generate_jobs(
    config: Dict,
    workloads: List[str],
    backends: List[str],
    precisions: List[str],
    batch_sizes: List[int]
) -> List[Job]:
    """Generate jobs with Poisson-like arrival pattern."""
    jobs = []
    current_time = 0.0
    arrival_rate = config.get('arrival_rate_jobs_per_sec', 10.0)
    duration = config.get('duration_seconds', 60.0)
    interactive_ratio = config.get('interactive_ratio', 0.3)
    
    # SLO definitions
    interactive_slo_ms = 100.0
    batch_slo_ms = 1000.0
    
    job_id_counter = 0
    
    while current_time < duration:
        # Generate inter-arrival time (exponential distribution)
        inter_arrival = random.expovariate(arrival_rate)
        current_time += inter_arrival
        
        if current_time >= duration:
            break
        
        # Determine job type
        is_interactive = random.random() < interactive_ratio
        class_type = "interactive" if is_interactive else "batch"
        slo_ms = interactive_slo_ms if is_interactive else batch_slo_ms
        priority = 1 if is_interactive else 5
        
        # Sample job parameters
        workload_name = random.choice(workloads)
        backend = random.choice(backends)
        precision = random.choice(precisions)
        batch_size = random.choice(batch_sizes)
        
        job = Job(
            job_id=f"job_{job_id_counter:06d}",
            class_type=class_type,
            slo_ms=slo_ms,
            priority=priority,
            workload_name=workload_name,
            backend=backend,
            precision=precision,
            batch_size=batch_size,
            created_time=current_time
        )
        
        jobs.append(job)
        job_id_counter += 1
    
    return jobs


def setup_workloads(config: Dict) -> Dict:
    """Set up and load all workloads."""
    workloads = {}
    device = "cpu"  # Can be extended to detect CUDA
    
    workload_classes = {
        "resnet18": ResNet18Workload,
        "resnet50": ResNet50Workload,
        "vit": ViTWorkload
    }
    
    enabled_workloads = config.get('workloads', ['resnet18', 'resnet50', 'vit'])
    backends = config.get('backends_enabled', ['pytorch'])
    precisions = config.get('precision_modes', ['fp32'])
    
    for workload_name in enabled_workloads:
        if workload_name not in workload_classes:
            continue
        
        for backend in backends:
            for precision in precisions:
                key = f"{workload_name}_{backend}_{precision}"
                workload = workload_classes[workload_name]()
                
                onnx_path = None
                if backend == "onnx":
                    onnx_path = f"models/onnx/{workload_name}.onnx"
                    if not Path(onnx_path).exists():
                        print(f"Warning: ONNX model not found at {onnx_path}, skipping")
                        continue
                
                try:
                    workload.load(device, precision, backend, onnx_path)
                    workloads[key] = workload
                except Exception as e:
                    print(f"Warning: Failed to load {key}: {e}")
    
    return workloads


def create_scheduler(config: Dict, resource_manager, lane_manager, predictor=None):
    """Create scheduler based on config."""
    scheduler_name = config.get('scheduler_name', 'fifo')
    
    if scheduler_name == 'fifo':
        return FIFOScheduler(resource_manager, lane_manager)
    elif scheduler_name == 'priority_slo':
        return PrioritySLOScheduler(resource_manager, lane_manager)
    elif scheduler_name == 'batch_aware':
        return BatchAwareScheduler(resource_manager, lane_manager)
    elif scheduler_name == 'dynamic':
        predictor_fn = None
        if predictor:
            predictor_fn = lambda job: predictor.predict_job(job)
        return DynamicScheduler(resource_manager, lane_manager, predictor_fn)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


def run_simulation(config: Dict):
    """Run the main simulation loop."""
    # Set random seed
    seed = config.get('seed', 42)
    random.seed(seed)
    
    # Create run ID
    run_id = config.get('run_name', f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    if not run_id.startswith('run_'):
        run_id = f"run_{run_id}"
    
    print(f"Starting experiment: {run_id}")
    print(f"Configuration: {config.get('scheduler_name', 'fifo')}")
    
    # Setup components
    resource_manager = None
    if config.get('resources', {}).get('enabled', False):
        total_tokens = config.get('resources', {}).get('total_tokens', 1000)
        cpu_fallback = config.get('resources', {}).get('cpu_fallback', False)
        resource_manager = ResourceManager(total_tokens, cpu_fallback)
    
    lane_manager = None
    if config.get('lanes', {}).get('enabled', False):
        num_lanes = config.get('lanes', {}).get('num_lanes', 2)
        lane_manager = LaneManager(num_lanes)
    
    # Setup predictor
    predictor = None
    if config.get('predictor', {}).get('enabled', False):
        predictor_path = config.get('predictor', {}).get('model_path', 'models/predictor.pkl')
        if Path(predictor_path).exists():
            predictor = LatencyPredictor.load(predictor_path)
            print(f"Loaded predictor from {predictor_path}")
        else:
            print(f"Warning: Predictor not found at {predictor_path}, continuing without it")
    
    # Create scheduler
    scheduler = create_scheduler(config, resource_manager, lane_manager, predictor)
    
    # Setup workloads
    workloads_dict = setup_workloads(config)
    if not workloads_dict:
        raise RuntimeError("No workloads loaded!")
    
    # Create executor and register workloads
    executor = Executor(resource_manager, lane_manager)
    for key, workload in workloads_dict.items():
        executor.register_workload(workload, key=key)
    
    # Setup microbatcher
    microbatcher = None
    if config.get('microbatch', {}).get('enabled', False) and scheduler.can_batch():
        window_ms = config.get('microbatch', {}).get('window_ms', 10.0)
        max_batch = config.get('microbatch', {}).get('max_batch', 128)
        microbatcher = MicroBatcher(
            enabled=True,
            window_ms=window_ms,
            max_batch_size=max_batch,
            resource_manager=resource_manager
        )
    
    # Create trace logger
    logger = TraceLogger(run_id, config.get('scheduler_name', 'fifo'))
    
    # Generate jobs
    workloads_list = config.get('workloads', ['resnet18', 'resnet50', 'vit'])
    backends = config.get('backends_enabled', ['pytorch'])
    precisions = config.get('precision_modes', ['fp32'])
    batch_sizes = config.get('batch_sizes', [1, 8, 32])
    
    jobs = generate_jobs(config, workloads_list, backends, precisions, batch_sizes)
    print(f"Generated {len(jobs)} jobs")
    
    # Create queue and enqueue jobs
    queue = JobQueue()
    for job in jobs:
        job.enqueue_time = job.created_time
        queue.push(job)
    
    # Simulation loop
    simulation_time = 0.0
    duration = config.get('duration_seconds', 60.0)
    
    print(f"Running simulation for {duration} seconds...")
    start_wall_time = time.time()
    
    jobs_processed = 0
    
    while simulation_time < duration and jobs_processed < len(jobs):
        # Get current simulation time (simplified - in real system this would be event-driven)
        if lane_manager and jobs_processed > 0:
            # Simulation time advances based on lane availability
            min_lane_time = min(lane.available_at for lane in lane_manager.lanes)
            simulation_time = max(simulation_time, min_lane_time)
        elif jobs_processed == 0:
            # Start at time 0
            simulation_time = 0.0
        else:
            # Simple time progression (fallback)
            simulation_time += 0.01
        
        # Select next job(s)
        if not queue:
            # No more jobs
            break
        
        seed_job = scheduler.select_next(queue, simulation_time)
        if seed_job is None:
            # No eligible jobs, advance time
            continue
        
        # Micro-batching
        if microbatcher:
            window_end = time.time() + (microbatcher.window_ms / 1000.0)
            bundle = microbatcher.gather_batch(seed_job, queue, simulation_time, window_end)
            
            if len(bundle) > 1:
                # Execute bundle
                executor.execute_bundle(bundle, simulation_time)
                for job in bundle:
                    logger.log_job(job)
                jobs_processed += len(bundle)
            else:
                # Single job
                executor.execute_job(seed_job, simulation_time)
                logger.log_job(seed_job)
                jobs_processed += 1
        else:
            # Single job execution
            executor.execute_job(seed_job, simulation_time)
            logger.log_job(seed_job)
            jobs_processed += 1
        
        # Update simulation time based on execution
        if seed_job.end_time is not None:
            simulation_time = max(simulation_time, seed_job.end_time)
        
        # Break if queue is empty
        if not queue:
            break
    
    # Process any remaining jobs (if time expired)
    remaining_jobs = queue.peek_all()
    if remaining_jobs:
        print(f"Warning: {len(remaining_jobs)} jobs remaining in queue (simulation time expired)")
    
    # Finalize logging
    logger.finalize()
    
    wall_time = time.time() - start_wall_time
    print(f"Simulation completed in {wall_time:.2f} seconds")
    
    # Generate plots
    trace_path = f"results/{run_id}/trace.csv"
    summary_path = f"results/{run_id}/summary.json"
    plots_dir = f"results/{run_id}/plots"
    
    if Path(trace_path).exists():
        generate_all_plots(trace_path, summary_path, plots_dir, config.get('scheduler_name', 'fifo'))
        
        # Print summary
        from analysis.report import load_summary
        summary = load_summary(summary_path)
        print_summary(summary)
    else:
        print(f"Warning: Trace file not found at {trace_path}")
    
    print(f"\nResults saved to results/{run_id}/")
    return run_id


def main():
    parser = argparse.ArgumentParser(description='Run ML workload scheduling experiment')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    args = parser.parse_args()
    
    config = load_config(args.config)
    run_id = run_simulation(config)
    print(f"\nExperiment complete: {run_id}")


if __name__ == "__main__":
    main()

