"""Resource manager for memory-aware scheduling."""

from typing import Dict
from runtime.job import Job


class ResourceManager:
    """Manages resource tokens for memory-aware scheduling."""
    
    def __init__(self, total_tokens: int = 1000, cpu_fallback: bool = False):
        self.total_tokens = total_tokens
        self.cpu_fallback = cpu_fallback
        self.available_tokens = total_tokens
        self.reserved_jobs: Dict[str, int] = {}  # job_id -> tokens
        
        # Base model costs (in tokens)
        self.model_costs = {
            "resnet18": 80,
            "resnet50": 160,
            "vit": 260
        }
        
        # Activation cost multiplier per batch size
        self.activation_multiplier = 4
    
    def compute_cost(self, job: Job) -> int:
        """
        Compute resource cost for a job.
        
        Cost = base_model_cost + (batch_size * activation_multiplier)
        If fp16: multiply by 0.7
        """
        base_cost = self.model_costs.get(job.workload_name, 100)
        activation_cost = job.batch_size * self.activation_multiplier
        total_cost = base_cost + activation_cost
        
        if job.precision == "fp16":
            total_cost = int(total_cost * 0.7)
        
        return total_cost
    
    def can_run(self, job: Job) -> bool:
        """Check if there are enough tokens available for the job."""
        cost = self.compute_cost(job)
        return self.available_tokens >= cost
    
    def reserve(self, job: Job) -> bool:
        """
        Reserve tokens for a job.
        
        Returns:
            True if reservation successful, False otherwise
        """
        cost = self.compute_cost(job)
        if self.available_tokens >= cost:
            self.available_tokens -= cost
            self.reserved_jobs[job.job_id] = cost
            job.resource_cost_tokens = cost
            return True
        return False
    
    def release(self, job: Job) -> None:
        """Release tokens reserved for a job."""
        if job.job_id in self.reserved_jobs:
            tokens = self.reserved_jobs.pop(job.job_id)
            self.available_tokens += tokens
    
    def get_available_tokens(self) -> int:
        """Get current available token count."""
        return self.available_tokens



