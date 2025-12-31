"""Batch-aware scheduler that prefers jobs that can be batched together."""

from typing import List, Optional
from runtime.job import Job
from runtime.job_queue import JobQueue
from scheduler.common import filter_eligible_jobs, compute_batchability_score
from runtime.resource_manager import ResourceManager
from runtime.lane_manager import LaneManager


class BatchAwareScheduler:
    """Scheduler that prefers jobs that can be batched together."""
    
    def __init__(self, resource_manager: ResourceManager = None, lane_manager: LaneManager = None):
        self.resource_manager = resource_manager
        self.lane_manager = lane_manager
    
    def select_next(self, queue: JobQueue, current_time: float) -> Optional[Job]:
        """Select job with highest batchability potential."""
        def selector(jobs: List[Job]) -> Optional[int]:
            eligible = filter_eligible_jobs(jobs, self.resource_manager, self.lane_manager, current_time)
            if not eligible:
                return None
            
            # Score each job by how batchable it is with other jobs
            scores = []
            for job in eligible:
                batchability = 0.0
                # Check how many other jobs are batchable with this one
                for other_job in eligible:
                    if other_job.job_id != job.job_id:
                        batchability += compute_batchability_score(job, other_job)
                
                # Also prefer interactive jobs slightly
                priority_bonus = 1.0 if job.class_type == "interactive" else 0.5
                
                total_score = batchability + priority_bonus
                scores.append((total_score, job))
            
            # Select highest scoring job
            selected = max(scores, key=lambda x: x[0])[1]
            return jobs.index(selected)
        
        return queue.pop_next(selector)
    
    def can_batch(self) -> bool:
        """Batch-aware scheduler supports batching."""
        return True



