"""Priority/SLO-aware scheduler (EDF - Earliest Deadline First)."""

from typing import List, Optional
from runtime.job import Job
from runtime.job_queue import JobQueue
from scheduler.common import filter_eligible_jobs, compare_priority, compare_deadline
from runtime.resource_manager import ResourceManager
from runtime.lane_manager import LaneManager


class PrioritySLOScheduler:
    """Scheduler that prioritizes interactive jobs and uses EDF within class."""
    
    def __init__(self, resource_manager: ResourceManager = None, lane_manager: LaneManager = None):
        self.resource_manager = resource_manager
        self.lane_manager = lane_manager
    
    def select_next(self, queue: JobQueue, current_time: float) -> Optional[Job]:
        """Select next job: interactive first, then earliest deadline."""
        def selector(jobs: List[Job]) -> Optional[int]:
            eligible = filter_eligible_jobs(jobs, self.resource_manager, self.lane_manager, current_time)
            if not eligible:
                return None
            
            # Separate interactive and batch
            interactive = [j for j in eligible if j.class_type == "interactive"]
            batch = [j for j in eligible if j.class_type == "batch"]
            
            # Prioritize interactive
            if interactive:
                # Among interactive, pick earliest deadline
                selected = min(interactive, key=lambda j: j.get_deadline())
            elif batch:
                # Among batch, pick earliest deadline
                selected = min(batch, key=lambda j: j.get_deadline())
            else:
                return None
            
            return jobs.index(selected)
        
        return queue.pop_next(selector)
    
    def can_batch(self) -> bool:
        """Priority/SLO scheduler doesn't support micro-batching."""
        return False


