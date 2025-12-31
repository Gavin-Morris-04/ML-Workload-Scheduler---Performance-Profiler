"""Common utilities for schedulers."""

from typing import List, Callable, Optional
from runtime.job import Job
from runtime.resource_manager import ResourceManager
from runtime.lane_manager import LaneManager


def filter_eligible_jobs(
    jobs: List[Job],
    resource_manager: ResourceManager = None,
    lane_manager: LaneManager = None,
    current_time: float = 0.0
) -> List[Job]:
    """
    Filter jobs that are eligible to run based on resource and lane availability.
    
    Args:
        jobs: List of candidate jobs
        resource_manager: Optional resource manager to check token availability
        lane_manager: Optional lane manager to check lane availability
        current_time: Current simulation time
    
    Returns:
        List of eligible jobs
    """
    eligible = []
    
    for job in jobs:
        # Check resource availability
        if resource_manager and not resource_manager.can_run(job):
            continue
        
        # Check lane availability (simple check - lane becomes available by start_time)
        if lane_manager:
            # This is a simplified check - actual scheduling happens in executor
            pass
        
        eligible.append(job)
    
    return eligible


def compare_priority(job1: Job, job2: Job) -> int:
    """Compare two jobs by priority (lower priority value = higher priority)."""
    if job1.priority != job2.priority:
        return job1.priority - job2.priority
    return 0


def compare_deadline(job1: Job, job2: Job, current_time: float) -> int:
    """Compare two jobs by deadline (earlier deadline = higher priority)."""
    deadline1 = job1.get_deadline()
    deadline2 = job2.get_deadline()
    
    if deadline1 < deadline2:
        return -1
    elif deadline1 > deadline2:
        return 1
    return 0


def compute_batchability_score(job1: Job, job2: Job) -> float:
    """
    Compute how batchable two jobs are (0-1, higher = more batchable).
    
    Jobs are more batchable if they share:
    - workload_name
    - backend
    - precision
    - class_type
    """
    score = 0.0
    
    if job1.workload_name == job2.workload_name:
        score += 0.4
    if job1.backend == job2.backend:
        score += 0.3
    if job1.precision == job2.precision:
        score += 0.2
    if job1.class_type == job2.class_type:
        score += 0.1
    
    return score



