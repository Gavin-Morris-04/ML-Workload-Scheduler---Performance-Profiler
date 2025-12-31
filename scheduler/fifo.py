"""FIFO (First-In-First-Out) scheduler."""

from typing import List, Optional
from runtime.job import Job
from runtime.job_queue import JobQueue
from scheduler.common import filter_eligible_jobs
from runtime.resource_manager import ResourceManager
from runtime.lane_manager import LaneManager


class FIFOScheduler:
    """Simple FIFO scheduler - picks earliest enqueued job."""
    
    def __init__(self, resource_manager: ResourceManager = None, lane_manager: LaneManager = None):
        self.resource_manager = resource_manager
        self.lane_manager = lane_manager
    
    def select_next(self, queue: JobQueue, current_time: float) -> Optional[Job]:
        """Select the next job using FIFO policy."""
        def selector(jobs: List[Job]) -> Optional[int]:
            eligible = filter_eligible_jobs(jobs, self.resource_manager, self.lane_manager, current_time)
            if not eligible:
                return None
            
            # Find earliest enqueue time
            earliest_job = min(eligible, key=lambda j: j.enqueue_time if j.enqueue_time else float('inf'))
            return jobs.index(earliest_job)
        
        return queue.pop_next(selector)
    
    def can_batch(self) -> bool:
        """FIFO doesn't support batching."""
        return False


