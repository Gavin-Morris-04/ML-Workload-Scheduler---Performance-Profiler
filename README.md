# ML Workload Scheduler & Performance Profiler

A production-grade inference runtime system for scheduling mixed ML workloads under constraints. This system implements tail latency optimization, mixed SLOs (Service Level Objectives), micro-batching, memory-aware scheduling, PyTorch/ONNX backends, concurrency simulation, and a learned latency predictor.

**Perfect for:**
- Understanding ML inference scheduling challenges
- Benchmarking scheduling algorithms
- Learning about production ML systems design
- Portfolio projects for ML/AI engineering roles

## 🚀 Quick Start (30 seconds)

```bash
# 1. Clone and enter directory
git clone https://github.com/Gavin-Morris-04/ML-Workload-Scheduler---Performance-Profiler.git
cd ML-Workload-Scheduler---Performance-Profiler

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows PowerShell
# OR: source .venv/bin/activate  # Mac/Linux

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Run your first experiment
python run_experiment.py --config configs/fifo_baseline.yaml
```

That's it! Results will be in `results/run_fifo_baseline/`

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

### Prerequisites

- **Python 3.8 or higher** (Python 3.10+ recommended)
- **Git** (to clone the repository)
- **pip** (Python package manager)

### Step 1: Clone the Repository

```bash
git clone https://github.com/Gavin-Morris-04/ML-Workload-Scheduler---Performance-Profiler.git
cd ML-Workload-Scheduler---Performance-Profiler
```

### Step 2: Create Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**Mac/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

**Option A: CPU-only (Recommended for first-time users)**

```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Install PyTorch CPU version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install all other dependencies
pip install -r requirements.txt
```

**Option B: With CUDA (If you have NVIDIA GPU)**

First, check your CUDA version:
```bash
nvidia-smi
```

Then install PyTorch with the matching CUDA version. Visit https://pytorch.org/ for the exact command for your CUDA version.

**Example for CUDA 11.8:**
```bash
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

**Option C: Install from requirements.txt (Easiest)**

This will install CPU versions by default:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Note:** If you install from `requirements.txt` without specifying a PyTorch index, you'll get CPU versions. For GPU support, install PyTorch separately first (see Option B), then run `pip install -r requirements.txt` to get the remaining dependencies.

### Step 4: Verify Installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python verify_setup.py
```

If the verification script runs without errors, you're ready to go!

## Quick Start

### Run Your First Experiment (No Configuration Needed)

The simplest way to get started - run with default settings:

```bash
python run_experiment.py --config configs/fifo_baseline.yaml
```

This will:
1. Generate ~600 jobs with mixed interactive/batch workloads
2. Schedule them using FIFO algorithm
3. Execute inference on ResNet-18, ResNet-50, and ViT models
4. Generate results in `results/run_fifo_baseline/`:
   - `trace.csv` - Per-job metrics
   - `summary.json` - Aggregate statistics
   - `plots/` - Visualization charts

**Expected runtime:** 30-60 seconds (depending on your CPU/GPU)

### Run Different Experiments

**Basic FIFO (Simple baseline):**
```bash
python run_experiment.py --config configs/fifo_baseline.yaml
```

**Full-featured (Dynamic scheduler + micro-batching + resources):**
```bash
python run_experiment.py --config configs/mixed_slo_microbatch.yaml
```

**With learned latency predictor:**
```bash
# First, train the predictor (one-time setup, takes ~5-10 minutes)
python -c "
from models.latency_predictor import collect_profiling_data, LatencyPredictor
from workloads.resnet18 import ResNet18Workload
from workloads.resnet50 import ResNet50Workload
from workloads.vit import ViTWorkload

print('Collecting profiling data...')
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

print('Training predictor...')
predictor = LatencyPredictor(model_type='random_forest')
predictor.train('results/profiles.csv')
predictor.save('models/predictor.pkl')
print('Predictor saved to models/predictor.pkl')
"

# Then run with predictor
python run_experiment.py --config configs/dynamic_predictor.yaml
```

### Export ONNX Models (Optional)

If you want to use ONNX Runtime backend instead of PyTorch:

```bash
python optimization/export_onnx.py
```

This creates `models/onnx/*.onnx` files. Then edit your config YAML to include `onnx` in `backends_enabled`.

## Configuration

All experiments are configured via YAML files in the `configs/` directory. Each config file controls:

### Key Configuration Parameters

- **`run_name`**: Name for this experiment run
- **`seed`**: Random seed for reproducibility (default: 42)
- **`duration_seconds`**: How long to run the simulation (default: 60.0)
- **`arrival_rate_jobs_per_sec`**: Job arrival rate (default: 10.0)
- **`interactive_ratio`**: Fraction of interactive jobs (default: 0.3)
- **`scheduler_name`**: Which scheduler to use
  - `fifo` - First-in-first-out (baseline)
  - `priority_slo` - Priority-based with SLO awareness
  - `batch_aware` - Prefers batchable jobs
  - `dynamic` - Multi-objective with ML prediction
- **`workloads`**: Models to use (`resnet18`, `resnet50`, `vit`)
- **`backends_enabled`**: `pytorch` and/or `onnx`
- **`precision_modes`**: `fp32` and/or `fp16` (fp16 requires CUDA)
- **`batch_sizes`**: List of batch sizes to test (e.g., `[1, 8, 32]`)
- **`microbatch.enabled`**: Enable micro-batching (true/false)
- **`resources.enabled`**: Enable memory-aware scheduling (true/false)
- **`lanes.enabled`**: Enable multi-lane concurrency simulation (true/false)
- **`predictor.enabled`**: Use learned latency predictor (true/false)

### Example: Modify Config for Your Needs

Edit `configs/fifo_baseline.yaml`:

```yaml
duration_seconds: 120.0        # Run for 2 minutes instead of 1
arrival_rate_jobs_per_sec: 5.0 # Slower arrival rate
interactive_ratio: 0.5         # 50% interactive jobs
batch_sizes: [1, 4, 16]        # Smaller batch sizes
```

Then run:
```bash
python run_experiment.py --config configs/fifo_baseline.yaml
```

See `configs/` directory for ready-to-use examples.

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

## Understanding the Output

Each experiment run creates a directory `results/<run_id>/` containing:

### 1. `trace.csv` - Per-Job Data
Every completed job has one row with:
- Job ID, type (interactive/batch), workload name
- Backend, precision, batch size
- Timing: enqueue_time, start_time, end_time
- Metrics: wait_ms, service_ms, e2e_ms, throughput_sps
- SLO: slo_ms, slo_violation (0 or 1)
- Resources: lane_id, resource_cost_tokens

**Open in Excel/Google Sheets or analyze with pandas:**
```python
import pandas as pd
df = pd.read_csv('results/run_fifo_baseline/trace.csv')
print(df.describe())
```

### 2. `summary.json` - Aggregate Statistics
Contains:
- Overall: total_jobs, total_samples, throughput
- Latency percentiles: P50, P95, P99 (for all, interactive, batch)
- SLO violations: counts and rates
- Wait times and service times

**View summary:**
```bash
cat results/run_fifo_baseline/summary.json
```

Or use the analysis script:
```bash
python -c "from analysis.report import load_summary, print_summary; print_summary(load_summary('results/run_fifo_baseline/summary.json'))"
```

### 3. `plots/` - Visualizations
- **`latency_cdf.png`**: Cumulative distribution of end-to-end latency (interactive vs batch)
- **`slo_violations.png`**: Bar chart of SLO violation rates
- **`latency_percentiles.png`**: P50/P95/P99 latency comparison
- **`wait_time_dist.png`**: Histogram of queue wait times

All plots are automatically generated and saved as PNG files.

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

## Project Structure

```
ML-Workload-Scheduler---Performance-Profiler/
├── runtime/              # Core execution infrastructure
│   ├── job.py           # Job definition and metrics
│   ├── job_queue.py     # Flexible job queue
│   ├── executor.py      # Job execution engine
│   ├── resource_manager.py  # Memory-aware scheduling
│   ├── lane_manager.py  # Multi-lane concurrency
│   └── trace_logger.py  # CSV/JSON logging
├── workloads/           # ML model implementations
│   ├── base.py         # Abstract workload interface
│   ├── resnet18.py     # ResNet-18 (fast)
│   ├── resnet50.py     # ResNet-50 (medium)
│   └── vit.py          # Vision Transformer (slow)
├── scheduler/           # Scheduling algorithms
│   ├── fifo.py         # First-in-first-out
│   ├── priority_slo.py # Priority + EDF
│   ├── batch_aware.py  # Batchability-aware
│   ├── dynamic.py      # Multi-objective ML-driven
│   └── common.py       # Shared utilities
├── optimization/        # Performance optimizations
│   ├── microbatcher.py # Micro-batching logic
│   └── export_onnx.py  # ONNX model export
├── profiler/            # Performance measurement
│   ├── profiler.py     # Inference timing
│   └── metrics.py      # Metrics computation
├── analysis/            # Results analysis
│   ├── plots.py        # Visualization generation
│   └── report.py       # Summary reports
├── models/              # ML components
│   └── latency_predictor.py  # Learned latency model
├── configs/             # Experiment configurations
│   ├── fifo_baseline.yaml
│   ├── mixed_slo_microbatch.yaml
│   └── dynamic_predictor.yaml
├── results/             # Output directory (git-ignored)
│   └── run_*/          # Per-run results
├── run_experiment.py    # Main entry point
├── validate_results.py  # Data validation script
├── verify_setup.py      # Installation verification
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

### Development

To add new features or modify the system:

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

## Performance Tips & Troubleshooting

### Performance Optimization

- **CPU-only systems**: 
  - Reduce `arrival_rate_jobs_per_sec` to 5-7 (default is 10)
  - Use smaller `batch_sizes: [1, 4, 8]` instead of `[1, 8, 32, 128]`
  - Reduce `duration_seconds` for faster testing (e.g., 30.0)

- **GPU systems**:
  - Install CUDA-enabled PyTorch (see installation section)
  - Enable `fp16` precision in config
  - Can handle higher arrival rates (10-20 jobs/sec)
  
- **ONNX Runtime**:
  - Often 20-30% faster than PyTorch for inference
  - Export models first: `python optimization/export_onnx.py`
  - Add `onnx` to `backends_enabled` in config

- **Predictor**:
  - Train once (takes 5-10 minutes), then reuse
  - Speeds up dynamic scheduler decision-making

### Common Issues

**Problem: "ModuleNotFoundError: No module named 'workloads'"**
```bash
# Make sure you're in the project directory
cd ML-Workload-Scheduler---Performance-Profiler
# And virtual environment is activated
.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate    # Mac/Linux
```

**Problem: "No module named 'onnxscript'"**
```bash
pip install onnxscript
```

**Problem: "CUDA out of memory" or "Cannot reserve resources"**
- Reduce `arrival_rate_jobs_per_sec` in config
- Use smaller batch sizes
- Reduce `total_tokens` in resources section

**Problem: Results show negative latencies**
- This was a bug that's been fixed. Make sure you have the latest code.
- If you see this, validate with: `python validate_results.py results/<run_id>/trace.csv`

**Problem: Experiments run very slowly**
- On CPU: This is normal for ResNet-50 and ViT models
- Reduce duration or arrival rate for faster testing
- Use only `resnet18` workload for quick tests

### Validation

Validate your results to ensure data integrity:
```bash
python validate_results.py results/run_fifo_baseline/trace.csv
```

This checks for:
- Missing/null timestamps
- Negative latencies
- Inconsistent timing data
- Invalid metrics

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



