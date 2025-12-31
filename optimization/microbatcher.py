"""Micro-batching logic for combining compatible jobs."""

from typing import List, Optional
from runtime.job import Job
from runtime.job_queue import JobQueue
import time


class MicroBatcher:
    """Micro-batches compatible jobs together."""
    
    def __init__(
        self,
        enabled: bool = True,
        window_ms: float = 10.0,
        max_batch_size: int = 128,
        only_batch_class_type: str = "batch",
        resource_manager = None
    ):
        self.enabled = enabled
        self.window_ms = window_ms / 1000.0  # Convert to seconds
        self.max_batch_size = max_batch_size
        self.only_batch_class_type = only_batch_class_type  # Only batch jobs of this type
        self.resource_manager = resource_manager
    
    def can_batch(self, job1: Job, job2: Job) -> bool:
        """Check if two jobs can be batched together."""
        return (
            job1.workload_name == job2.workload_name and
            job1.backend == job2.backend and
            job1.precision == job2.precision and
            job1.class_type == job2.class_type
        )
    
    def gather_batch(
        self,
        seed_job: Job,
        queue: JobQueue,
        current_time: float,
        window_end_time: float
    ) -> List[Job]:
        """
        Gather compatible jobs from queue to form a micro-batch.
        
        Args:
            seed_job: The first job in the batch
            queue: Job queue to pull from
            current_time: Current simulation time
            window_end_time: Simulation time when batching window ends
        
        Returns:
            List of jobs to batch together (includes seed_job)
        """
        if not self.enabled:
            return [seed_job]
        
        bundle = [seed_job]
        current_batch_size = seed_job.batch_size
        
        # Only batch batch-type jobs (interactive should run immediately)
        if seed_job.class_type != self.only_batch_class_type:
            return bundle
        
        # Gather compatible jobs until window ends or max batch size reached
        available_jobs = queue.peek_all()
        
        # Track simulation time progression
        sim_time = current_time
        
        for job in available_jobs:
            if job.job_id == seed_job.job_id:
                continue
            
            # Check if we've exceeded the window (use simulation time, not wall clock)
            if sim_time >= window_end_time:
                break
            
            # Check if we've exceeded max batch size
            if current_batch_size + job.batch_size > self.max_batch_size:
                continue
            
            # Check compatibility
            if self.can_batch(seed_job, job):
                # Check resource availability if resource manager exists
                if self.resource_manager and not self.resource_manager.can_run(job):
                    continue
                
                bundle.append(job)
                current_batch_size += job.batch_size
                
                # Remove from queue
                queue.remove(job)
        
        return bundle

