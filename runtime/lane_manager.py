"""Lane manager for concurrency simulation."""

from typing import List
from runtime.job import Job


class Lane:
    """Represents a single execution lane."""
    
    def __init__(self, lane_id: int):
        self.lane_id = lane_id
        self.available_at: float = 0.0  # Unix timestamp when lane becomes available


class LaneManager:
    """Manages multiple execution lanes for concurrency simulation."""
    
    def __init__(self, num_lanes: int = 2):
        self.num_lanes = num_lanes
        self.lanes: List[Lane] = [Lane(i) for i in range(num_lanes)]
    
    def select_lane(self, current_time: float) -> Lane:
        """Select the lane that becomes available first."""
        # Find lane that becomes available earliest
        available_lane = min(self.lanes, key=lambda l: l.available_at)
        return available_lane
    
    def schedule_job(self, job: Job, service_time: float, current_time: float) -> None:
        """
        Schedule a job on a lane.
        
        Args:
            job: The job to schedule
            service_time: Service time in seconds
            current_time: Current simulation time
        """
        lane = self.select_lane(current_time)
        job.lane_id = lane.lane_id
        
        # Job starts when lane becomes available (or immediately if lane is free)
        start_time = max(current_time, lane.available_at)
        job.start_time = start_time
        job.end_time = start_time + service_time
        
        # Lane becomes available after job completes
        lane.available_at = job.end_time
    
    def get_utilization(self, current_time: float, total_time: float) -> float:
        """Compute lane utilization (0-1)."""
        if total_time == 0:
            return 0.0
        
        total_busy_time = sum(max(0, lane.available_at - 0) for lane in self.lanes)
        max_possible_time = self.num_lanes * total_time
        return min(1.0, total_busy_time / max_possible_time) if max_possible_time > 0 else 0.0


