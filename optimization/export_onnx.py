"""ONNX model export utilities."""

import os
import sys
from pathlib import Path

# Fix Windows encoding issues
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add parent directory to path for imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from workloads.resnet18 import ResNet18Workload
from workloads.resnet50 import ResNet50Workload
from workloads.vit import ViTWorkload


def export_all_models(output_dir: str = "models/onnx"):
    """Export all workloads to ONNX format."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    workloads = [
        (ResNet18Workload(), "resnet18"),
        (ResNet50Workload(), "resnet50"),
        (ViTWorkload(), "vit")
    ]
    
    for workload, name in workloads:
        onnx_path = output_path / f"{name}.onnx"
        print(f"Exporting {name} to {onnx_path}...")
        
        try:
            workload.export_onnx(str(onnx_path), batch_size=1)
            print(f"[OK] Exported {name}")
        except Exception as e:
            print(f"[FAILED] Failed to export {name}: {e}")
    
    print("ONNX export complete!")


if __name__ == "__main__":
    export_all_models()

