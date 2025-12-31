"""Job executor for running ML inference jobs."""

import time
from typing import List, Dict
from runtime.job import Job
from runtime.resource_manager import ResourceManager
from runtime.lane_manager import LaneManager
from workloads.base import Workload


class Executor:
    """Executes ML inference jobs."""
    
    def __init__(self, resource_manager: ResourceManager = None, lane_manager: LaneManager = None):
        self.resource_manager = resource_manager
        self.lane_manager = lane_manager
        self.workloads: Dict[str, Workload] = {}
    
    def register_workload(self, workload: Workload, key: str = None) -> None:
        """
        Register a workload for execution.
        
        Args:
            workload: The workload to register
            key: Optional key to use (defaults to workload.name)
        """
        if key is None:
            key = workload.name
        self.workloads[key] = workload
    
    def execute_job(self, job: Job, current_time: float) -> None:
        """
        Execute a single job.
        
        Args:
            job: The job to execute
            current_time: Current simulation time
        """
        # Get the workload using composite key
        workload_key = f"{job.workload_name}_{job.backend}_{job.precision}"
        workload = self.workloads.get(workload_key)
        if workload is None:
            raise ValueError(f"Workload {workload_key} not registered. Available: {list(self.workloads.keys())}")
        
        # Reserve resources if resource manager exists
        if self.resource_manager:
            if not self.resource_manager.reserve(job):
                raise RuntimeError(f"Cannot reserve resources for job {job.job_id}")
        
        # Set enqueue time if not set (should already be set when job was queued)
        if job.enqueue_time is None:
            # Fallback: use current time, but this shouldn't happen normally
            job.enqueue_time = current_time
        # Ensure enqueue_time is not later than current_time (sanity check)
        elif job.enqueue_time > current_time:
            # This shouldn't happen, but ensure we use the earlier time
            job.enqueue_time = min(job.enqueue_time, current_time)
        
        # Run inference
        try:
            _, inference_time = workload.infer(job.batch_size)
            
            # Schedule on lane if lane manager exists
            if self.lane_manager:
                self.lane_manager.schedule_job(job, inference_time, current_time)
            else:
                # Simple sequential execution
                # Ensure start_time >= enqueue_time
                start_time = current_time
                if job.enqueue_time is not None:
                    start_time = max(start_time, job.enqueue_time)
                job.start_time = start_time
                job.end_time = start_time + inference_time
                job.service_ms = inference_time * 1000.0
            
            job.compute_metrics()
        finally:
            # Release resources
            if self.resource_manager:
                self.resource_manager.release(job)
    
    def execute_bundle(self, jobs: List[Job], current_time: float) -> List[Job]:
        """
        Execute a bundle of jobs as a micro-batch.
        
        Args:
            jobs: List of jobs to execute together
            current_time: Current simulation time
        """
        if not jobs:
            return []
        
        # Use the first job's parameters for the combined inference
        # (In a real system, you'd actually combine the batches)
        seed_job = jobs[0]
        # combined_batch_size computed after resource filtering (above)
        if 'combined_batch_size' not in locals():
            combined_batch_size = sum(j.batch_size for j in jobs)
        
        workload_key = f"{seed_job.workload_name}_{seed_job.backend}_{seed_job.precision}"
        workload = self.workloads.get(workload_key)
        if workload is None:
            raise ValueError(f"Workload {workload_key} not registered. Available: {list(self.workloads.keys())}")
        
        # Reserve resources for all jobs
        if self.resource_manager:
            # Check total cost first
            total_cost = sum(self.resource_manager.compute_cost(job) for job in jobs)
            available = self.resource_manager.get_available_tokens()
            
            if available < total_cost:
                # Not enough resources for full bundle - execute only what fits
                # This can happen if resources changed between microbatcher check and execution
                reservable_jobs = []
                remaining_tokens = available
                
                for job in jobs:
                    job_cost = self.resource_manager.compute_cost(job)
                    if remaining_tokens >= job_cost:
                        reservable_jobs.append(job)
                        remaining_tokens -= job_cost
                    # Skip jobs that don't fit - they'll be picked up later
                
                if not reservable_jobs:
                    # None of the jobs can run - return empty list
                    # Don't raise exception, just return empty list
                    return []
                
                # Update jobs list to only include reservable ones
                jobs = reservable_jobs
                # Recompute combined batch size for the reservable jobs
                combined_batch_size = sum(j.batch_size for j in jobs)
            
            # Now reserve for all jobs in the (possibly reduced) bundle
            for job in jobs:
                if not self.resource_manager.reserve(job):
                    # This should not happen given the checks above
                    raise RuntimeError(f"Cannot reserve resources for job {job.job_id} (unexpected - checked {self.resource_manager.get_available_tokens()} available)")
        
        # Set enqueue times
        for job in jobs:
            if job.enqueue_time is None:
                job.enqueue_time = current_time
        
        # Run combined inference
        try:
            _, inference_time = workload.infer(combined_batch_size)
            
            # Schedule all jobs on lanes (they share the same service time)
            if self.lane_manager:
                # All jobs in bundle start and end at same time
                # Find the earliest possible start time for the bundle
                earliest_start = current_time
                for job in jobs:
                    if job.enqueue_time is not None:
                        earliest_start = max(earliest_start, job.enqueue_time)
                
                # Select lane and ensure start time respects both lane availability and enqueue times
                lane = self.lane_manager.select_lane(current_time)
                bundle_start_time = max(earliest_start, lane.available_at, current_time)
                
                for job in jobs:
                    job.lane_id = lane.lane_id
                    job.start_time = bundle_start_time
                    job.end_time = bundle_start_time + inference_time
                    job.service_ms = inference_time * 1000.0
                
                # Lane becomes available after bundle completes
                lane.available_at = bundle_start_time + inference_time
            else:
                # Sequential execution
                # Find earliest start time that respects all enqueue times
                bundle_start_time = current_time
                for job in jobs:
                    if job.enqueue_time is not None:
                        bundle_start_time = max(bundle_start_time, job.enqueue_time)
                
                end_time = bundle_start_time + inference_time
                for job in jobs:
                    job.start_time = bundle_start_time
                    job.end_time = end_time
                    job.service_ms = inference_time * 1000.0
            
            # Compute metrics for all jobs
            for job in jobs:
                job.compute_metrics()
            
            # Return list of successfully executed jobs
            return jobs
        finally:
            # Release resources for all jobs (in case of exception)
            if self.resource_manager:
                for job in jobs:
                    if job.job_id in self.resource_manager.reserved_jobs:
                        self.resource_manager.release(job)

