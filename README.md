# ML Workload Scheduler & Performance Profiler

A production-grade inference runtime system for scheduling mixed ML workloads under constraints. This system implements tail latency optimization, mixed SLOs (Service Level Objectives), micro-batching, memory-aware scheduling, PyTorch/ONNX backends, concurrency simulation, and a learned latency predictor.

## Overview

This system is designed to simulate and benchmark ML inference scheduling strategies under realistic constraints:

- **Mixed Workloads**: Interactive (100ms SLO) and batch (1000ms SLO) jobs
- **Multiple Models**: ResNet-18, ResNet-50, Vision Transformer (ViT)
- **Backend Support**: PyTorch and ONNX Runtime
- **Scheduling Algorithms**: FIFO, Priority/SLO-aware, Batch-aware, Dynamic
- **Optimizations**: Micro-batching, memory-aware scheduling, concurrency simulation
- **ML-Driven**: Learned latency predictor for data-driven scheduling

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Experiment Runner                         │
│                   (run_experiment.py)                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
   ┌────▼────┐   ┌────▼────┐   ┌────▼────┐
   │ Job     │   │Scheduler│   │Executor │
   │ Queue   │──▶│ (4 algos)│──▶│         │
   └─────────┘   └────┬────┘   └────┬────┘
                      │              │
         ┌────────────┼──────────────┼────────────┐
         │            │              │            │
    ┌────▼────┐  ┌───▼───┐    ┌────▼────┐  ┌───▼────┐
    │Micro    │  │Resource│    │Workloads│  │Lane    │
    │Batcher  │  │Manager │    │(3 models)│  │Manager │
    └─────────┘  └────────┘    └──────────┘  └────────┘
```

### Components

1. **Runtime** (`runtime/`): Core job execution infrastructure
   - `job.py`: Job dataclass with metadata and metrics
   - `job_queue.py`: Flexible queue with custom selectors
   - `executor.py`: Job execution engine
   - `resource_manager.py`: Token-based memory management
   - `lane_manager.py`: Multi-lane concurrency simulation
   - `trace_logger.py`: CSV trace logging

2. **Workloads** (`workloads/`): ML model implementations
   - `resnet18.py`: Fast model (~80 tokens)
   - `resnet50.py`: Medium model (~160 tokens)
   - `vit.py`: Slow model (~260 tokens)

3. **Schedulers** (`scheduler/`): Scheduling algorithms
   - `fifo.py`: First-In-First-Out baseline
   - `priority_slo.py`: Interactive-first with EDF (Earliest Deadline First)
   - `batch_aware.py`: Batchability-aware selection
   - `dynamic.py`: Multi-objective with latency prediction

4. **Optimization** (`optimization/`):
   - `microbatcher.py`: Combine compatible jobs into micro-batches
   - `export_onnx.py`: Export PyTorch models to ONNX

5. **Profiling & Analysis** (`profiler/`, `analysis/`):
   - `profiler.py`: Inference timing utilities
   - `plots.py`: Generate latency CDFs, SLO violations, percentiles
   - `report.py`: Summary statistics

6. **ML Components** (`models/`):
   - `latency_predictor.py`: RandomForest/GradientBoosting predictor for service time

## Installation

### 1. Create Virtual Environment

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Mac/Linux
python -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

**CPU-only (safe default):**
```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install onnx onnxruntime pandas numpy matplotlib pyyaml scikit-learn tqdm
```

**With CUDA (if you have NVIDIA GPU):**
```bash
# Visit https://pytorch.org/ for the correct CUDA version command
# Example for CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install onnx onnxruntime pandas numpy matplotlib pyyaml scikit-learn tqdm
```

## Quick Start

### Basic Run (FIFO Scheduler)

```bash
python run_experiment.py --config configs/fifo_baseline.yaml
```

This will:
1. Generate jobs with Poisson arrival pattern
2. Schedule them using FIFO
3. Execute inference
4. Generate `results/run_*/trace.csv`, `summary.json`, and plots

### Advanced Run (Dynamic Scheduler with Micro-batching)

```bash
python run_experiment.py --config configs/mixed_slo_microbatch.yaml
```

### With Learned Predictor

1. **Train the predictor** (first time):
```python
from models.latency_predictor import collect_profiling_data, LatencyPredictor
from workloads.resnet18 import ResNet18Workload
from workloads.resnet50 import ResNet50Workload
from workloads.vit import ViTWorkload

# Collect profiling data
collect_profiling_data(
    workloads={
        'resnet18': ResNet18Workload,
        'resnet50': ResNet50Workload,
        'vit': ViTWorkload
    },
    batch_sizes=[1, 8, 32, 128],
    backends=['pytorch'],
    precisions=['fp32'],
    output_path='results/profiles.csv'
)

# Train predictor
predictor = LatencyPredictor(model_type='random_forest')
predictor.train('results/profiles.csv')
predictor.save('models/predictor.pkl')
```

2. **Run with predictor**:
```bash
python run_experiment.py --config configs/dynamic_predictor.yaml
```

## Configuration

Configuration files are YAML and specify:

- **Experiment parameters**: duration, arrival rate, interactive ratio
- **Scheduler**: `fifo`, `priority_slo`, `batch_aware`, or `dynamic`
- **Workloads**: Which models to use
- **Backends**: `pytorch` or `onnx`
- **Precision**: `fp32` or `fp16` (fp16 requires CUDA)
- **Micro-batching**: Window size, max batch size
- **Resources**: Token-based memory constraints
- **Lanes**: Number of concurrent execution lanes
- **Predictor**: Enable learned latency predictor

See `configs/` for examples.

## ONNX Export

To use ONNX backends, first export models:

```bash
python optimization/export_onnx.py
```

This creates `models/onnx/resnet18.onnx`, `resnet50.onnx`, `vit.onnx`.

Then enable ONNX in your config:
```yaml
backends_enabled:
  - pytorch
  - onnx
```

## Output

Each run creates a directory `results/<run_id>/` containing:

- **`trace.csv`**: Per-job metrics (latency, wait time, SLO violations)
- **`summary.json`**: Aggregate statistics (P50/P95/P99, throughput, violation rates)
- **`plots/`**: 
  - `latency_cdf.png`: Cumulative distribution of end-to-end latency
  - `slo_violations.png`: SLO violation rates by job class
  - `latency_percentiles.png`: P50/P95/P99 bars
  - `wait_time_dist.png`: Wait time histograms

## Metrics

The system tracks:

- **Latency**: End-to-end (e2e), wait time, service time
- **Percentiles**: P50, P95, P99
- **Throughput**: Jobs/sec and samples/sec
- **SLO Violations**: Rate for interactive and batch separately
- **Fairness**: Ratio of batch wait to interactive wait

## Schedulers Explained

### FIFO
Baseline: Processes jobs in arrival order.

### Priority/SLO
- Interactive jobs always prioritized
- Within class, uses EDF (Earliest Deadline First)

### Batch-Aware
- Prefers jobs that can be batched together
- Scores jobs by batchability with other queued jobs

### Dynamic
Multi-objective optimizer that considers:
- Interactive priority bonus
- Time-to-deadline urgency
- Batchability with other jobs
- Predicted service time (if predictor enabled)
- Resource cost penalty

## Resource Token Model

Memory is simulated using tokens:
- **Base model costs**: ResNet-18 (80), ResNet-50 (160), ViT (260)
- **Activation cost**: `batch_size × 4` tokens
- **FP16 reduction**: 30% discount (0.7× multiplier)

Total capacity defaults to 1000 tokens. Jobs can only run if sufficient tokens are available.

## Micro-batching

Compatible jobs (same workload, backend, precision, class) can be combined into micro-batches:
- **Window**: 10ms default (gather compatible jobs within window)
- **Max batch**: 128 samples
- **Policy**: Only batch-type jobs; interactive runs immediately

Each job in a micro-batch shares the same service time but has individual e2e latency based on enqueue time.

## Concurrency Simulation (Lanes)

Simulates 2+ execution lanes (streams):
- Jobs scheduled on the lane that becomes available first
- Start time = `max(current_time, lane.available_at)`
- Enables overlapping execution simulation

## Mapping to AMD GPUs / ROCm

This system abstracts runtime optimization concepts applicable to AMD GPU stacks:

### Backend Abstraction
- PyTorch → ROCm PyTorch backend
- ONNX → ONNX Runtime with ROCm execution provider
- Easy to add: TensorRT, OpenVINO, DirectML

### Memory Bandwidth + Batching Tradeoffs
- Token model simulates VRAM constraints
- Micro-batching maximizes memory bandwidth utilization
- FP16 optimization maps to AMD mixed-precision (FP16/BF16)

### Runtime Optimization
- Learned predictor → offline profiling + online adaptation
- Dynamic scheduling → kernel fusion opportunities
- Lane simulation → ROCm HIP streams / command queues

### Real-World Extensions
- Replace token model with actual VRAM tracking
- Use ROCm profiler (rocprof) for real latency measurements
- Integrate with ROCm MIGraphX for optimized inference graphs

## Development

### Project Structure
```
ml-workload-scheduler/
├── runtime/           # Core execution infrastructure
├── workloads/         # ML model implementations
├── scheduler/         # Scheduling algorithms
├── optimization/      # Micro-batching, ONNX export
├── profiler/          # Performance profiling
├── analysis/          # Plotting and reporting
├── models/            # Latency predictor
├── configs/           # YAML configuration files
├── results/           # Output traces and plots
└── run_experiment.py  # Main entry point
```

### Adding a New Scheduler

1. Create `scheduler/my_scheduler.py`:
```python
from scheduler.common import filter_eligible_jobs

class MyScheduler:
    def select_next(self, queue, current_time):
        def selector(jobs):
            eligible = filter_eligible_jobs(jobs, ...)
            # Your selection logic
            return index
        return queue.pop_next(selector)
```

2. Add to `run_experiment.py` in `create_scheduler()`.

### Adding a New Workload

1. Inherit from `Workload` in `workloads/base.py`
2. Implement `load()`, `infer()`, `export_onnx()`
3. Register in `run_experiment.py` `setup_workloads()`

## Performance Tips

- **CPU-only**: Runs fine but slower; consider reducing `arrival_rate_jobs_per_sec`
- **GPU**: Enable CUDA, use FP16, increase arrival rate
- **ONNX**: Often faster than PyTorch for inference; export models first
- **Predictor**: Train once, reuse across experiments

## Citation

If you use this system in research, please cite:

```bibtex
@software{ml_workload_scheduler,
  title={ML Workload Scheduler \& Performance Profiler},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/ml-workload-scheduler}
}
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- TorchVision models (ResNet, ViT)
- ONNX Runtime team
- AMD ROCm for inspiration on GPU runtime optimization

---

**Built for AMD ML Systems interviews** - demonstrates understanding of inference runtime optimization, scheduling theory, and production ML systems.



