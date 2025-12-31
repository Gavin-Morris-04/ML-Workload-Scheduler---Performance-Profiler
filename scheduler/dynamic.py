"""Dynamic scheduler with latency prediction and multi-objective optimization."""

from typing import List, Optional, Callable
from runtime.job import Job
from runtime.job_queue import JobQueue
from scheduler.common import filter_eligible_jobs
from runtime.resource_manager import ResourceManager
from runtime.lane_manager import LaneManager


class DynamicScheduler:
    """
    Dynamic scheduler that uses predictions and multi-objective scoring.
    
    Minimizes P99 latency, reduces SLO violations, maintains throughput.
    """
    
    def __init__(
        self,
        resource_manager: ResourceManager = None,
        lane_manager: LaneManager = None,
        predictor: Callable[[Job], float] = None
    ):
        self.resource_manager = resource_manager
        self.lane_manager = lane_manager
        self.predictor = predictor  # Function that takes a job and returns predicted service_ms
        
        # Scoring weights (can be tuned)
        self.w1 = 10.0  # Interactive bonus
        self.w2 = 1.0   # Urgency (1/time_to_deadline)
        self.w3 = 5.0   # Batchability bonus
        self.w4 = 2.0   # Inverse prediction (prefer faster jobs)
        self.w5 = 0.1   # Resource cost penalty
    
    def compute_score(self, job: Job, current_time: float, eligible_jobs: List[Job]) -> float:
        """Compute scheduling score for a job."""
        # Interactive bonus
        interactive_bonus = self.w1 if job.class_type == "interactive" else 0.0
        
        # Urgency (time to deadline)
        time_to_deadline = job.time_to_deadline(current_time)
        if time_to_deadline > 0:
            urgency = self.w2 / (time_to_deadline * 1000.0 + 1.0)  # Convert to ms, add 1 to avoid div by 0
        else:
            urgency = self.w2 * 10.0  # Past deadline - very urgent
        
        # Batchability bonus
        batchability = 0.0
        for other_job in eligible_jobs:
            if other_job.job_id != job.job_id:
                if (job.workload_name == other_job.workload_name and
                    job.backend == other_job.backend and
                    job.precision == other_job.precision):
                    batchability += 1.0
        
        batchability_bonus = self.w3 * min(batchability / 5.0, 1.0)  # Cap at 1.0
        
        # Inverse prediction (prefer faster jobs)
        if self.predictor:
            pred_ms = self.predictor(job) / 1000.0  # Convert to seconds
            if pred_ms > 0:
                inverse_pred = self.w4 / pred_ms
            else:
                inverse_pred = 0.0
        else:
            # Fallback: estimate based on workload (simple heuristic)
            workload_costs = {"resnet18": 0.05, "resnet50": 0.1, "vit": 0.2}
            base_cost = workload_costs.get(job.workload_name, 0.1)
            estimated_time = base_cost * job.batch_size
            inverse_pred = self.w4 / (estimated_time + 0.01)
        
        # Resource cost penalty
        if self.resource_manager:
            cost = self.resource_manager.compute_cost(job)
            cost_penalty = self.w5 * (cost / 1000.0)  # Normalize
        else:
            cost_penalty = 0.0
        
        total_score = interactive_bonus + urgency + batchability_bonus + inverse_pred - cost_penalty
        return total_score
    
    def select_next(self, queue: JobQueue, current_time: float) -> Optional[Job]:
        """Select next job using dynamic scoring."""
        def selector(jobs: List[Job]) -> Optional[int]:
            eligible = filter_eligible_jobs(jobs, self.resource_manager, self.lane_manager, current_time)
            if not eligible:
                return None
            
            # Score all eligible jobs
            scores = [(self.compute_score(job, current_time, eligible), job) for job in eligible]
            
            # Select highest scoring job
            selected = max(scores, key=lambda x: x[0])[1]
            return jobs.index(selected)
        
        return queue.pop_next(selector)
    
    def can_batch(self) -> bool:
        """Dynamic scheduler supports batching."""
        return True


