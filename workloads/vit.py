"""Vision Transformer (ViT) workload implementation."""

import torch
import torch.nn as nn
from torchvision import models
from typing import Tuple
import onnxruntime as ort
import numpy as np
from workloads.base import Workload


class ViTWorkload(Workload):
    """Vision Transformer B/16 model workload (slow)."""
    
    def __init__(self):
        super().__init__("vit")
    
    def load(self, device: str, precision: str, backend: str = "pytorch", onnx_path: str = None):
        self.device = device
        self.precision = precision
        
        if backend == "onnx":
            if onnx_path is None:
                raise ValueError("ONNX path required for ONNX backend")
            self.onnx_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
            self.backend = "onnx"
        else:
            # Use ViT-B/16 from torchvision
            try:
                self.model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
            except AttributeError:
                self.model = models.vit_b_16(pretrained=True)
            self.model.eval()
            self.model.to(device)
            
            if precision == "fp16" and device == "cuda":
                self.model = self.model.half()
            elif precision == "fp16" and device == "cpu":
                self.precision = "fp32"
            
            self.backend = "pytorch"
    
    def infer(self, batch_size: int) -> Tuple[torch.Tensor, float]:
        """Run inference and return output + time."""
        input_tensor = self.create_dummy_input(batch_size)
        
        if self.precision == "fp16" and self.device == "cuda":
            input_tensor = input_tensor.half()
        
        if self.backend == "onnx":
            input_np = input_tensor.cpu().numpy().astype(np.float32)
            input_name = self.onnx_session.get_inputs()[0].name
            
            start_event = torch.cuda.Event(enable_timing=True) if self.device == "cuda" else None
            end_event = torch.cuda.Event(enable_timing=True) if self.device == "cuda" else None
            
            if start_event:
                torch.cuda.synchronize()
                start_event.record()
            
            import time
            start_time = time.time()
            outputs = self.onnx_session.run(None, {input_name: input_np})
            
            if end_event:
                end_event.record()
                torch.cuda.synchronize()
                inference_time = start_event.elapsed_time(end_event) / 1000.0
            else:
                inference_time = time.time() - start_time
            
            output_tensor = torch.from_numpy(outputs[0])
        else:
            with torch.no_grad():
                if self.device == "cuda":
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize()
                    start_event.record()
                    
                    output_tensor = self.model(input_tensor)
                    
                    end_event.record()
                    torch.cuda.synchronize()
                    inference_time = start_event.elapsed_time(end_event) / 1000.0
                else:
                    import time
                    start_time = time.time()
                    output_tensor = self.model(input_tensor)
                    inference_time = time.time() - start_time
        
        return output_tensor, inference_time
    
    def export_onnx(self, output_path: str, batch_size: int = 1):
        """Export model to ONNX."""
        if self.model is None:
            try:
                model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
            except AttributeError:
                model = models.vit_b_16(pretrained=True)
            model.eval()
        else:
            model = self.model
            model.eval()
        
        dummy_input = torch.randn(batch_size, 3, 224, 224)
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
                opset_version=18,
                verbose=False
            )

