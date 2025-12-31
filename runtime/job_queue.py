"""Job queue implementation with flexible scheduling selectors."""

from typing import List, Callable, Optional
from runtime.job import Job


class JobQueue:
    """Queue that supports custom selection functions for schedulers."""
    
    def __init__(self):
        self._jobs: List[Job] = []
    
    def push(self, job: Job) -> None:
        """Add a job to the queue."""
        self._jobs.append(job)
    
    def pop_next(self, selector_fn: Callable[[List[Job]], Optional[int]]) -> Optional[Job]:
        """
        Pop the next job using a selector function.
        
        Args:
            selector_fn: Function that takes a list of jobs and returns
                        the index of the job to pop, or None if none eligible.
        
        Returns:
            The selected job, or None if no eligible job found.
        """
        if not self._jobs:
            return None
        
        idx = selector_fn(self._jobs)
        if idx is None or idx < 0 or idx >= len(self._jobs):
            return None
        
        return self._jobs.pop(idx)
    
    def peek_all(self) -> List[Job]:
        """Return a copy of all jobs in the queue (read-only)."""
        return list(self._jobs)
    
    def remove(self, job: Job) -> bool:
        """Remove a specific job from the queue."""
        try:
            self._jobs.remove(job)
            return True
        except ValueError:
            return False
    
    def __len__(self) -> int:
        return len(self._jobs)
    
    def __bool__(self) -> bool:
        return len(self._jobs) > 0
    
    def clear(self) -> None:
        """Clear all jobs from the queue."""
        self._jobs.clear()



