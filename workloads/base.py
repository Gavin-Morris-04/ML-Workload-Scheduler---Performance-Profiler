"""Base workload interface for ML models."""

from abc import ABC, abstractmethod
from typing import Tuple
import torch


class Workload(ABC):
    """Abstract base class for ML workloads."""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.device = None
        self.precision = None
        self.onnx_session = None
    
    @abstractmethod
    def load(self, device: str, precision: str, backend: str = "pytorch", onnx_path: str = None):
        """
        Load the model for inference.
        
        Args:
            device: "cpu" or "cuda"
            precision: "fp32" or "fp16"
            backend: "pytorch" or "onnx"
            onnx_path: Path to ONNX model file (if using ONNX backend)
        """
        pass
    
    @abstractmethod
    def infer(self, batch_size: int) -> Tuple[torch.Tensor, float]:
        """
        Run inference on a batch.
        
        Args:
            batch_size: Number of samples in the batch
        
        Returns:
            Tuple of (output_tensor, inference_time_seconds)
        """
        pass
    
    @abstractmethod
    def export_onnx(self, output_path: str, batch_size: int = 1):
        """
        Export model to ONNX format.
        
        Args:
            output_path: Where to save the ONNX file
            batch_size: Batch size for the exported model
        """
        pass
    
    def get_input_shape(self) -> Tuple[int, int, int, int]:
        """Return input shape as (batch, channels, height, width)."""
        return (1, 3, 224, 224)
    
    def create_dummy_input(self, batch_size: int) -> torch.Tensor:
        """Create dummy input tensor for inference."""
        return torch.randn(batch_size, 3, 224, 224).to(self.device)



