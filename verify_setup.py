"""Quick verification script to check that all components are importable."""

import sys

def check_imports():
    """Check that all main modules can be imported."""
    modules = [
        'runtime.job',
        'runtime.job_queue',
        'runtime.executor',
        'runtime.resource_manager',
        'runtime.lane_manager',
        'runtime.trace_logger',
        'workloads.base',
        'workloads.resnet18',
        'workloads.resnet50',
        'workloads.vit',
        'scheduler.fifo',
        'scheduler.priority_slo',
        'scheduler.batch_aware',
        'scheduler.dynamic',
        'optimization.microbatcher',
        'profiler.profiler',
        'analysis.plots',
        'models.latency_predictor'
    ]
    
    failed = []
    for module in modules:
        try:
            __import__(module)
            print(f"✓ {module}")
        except ImportError as e:
            print(f"✗ {module}: {e}")
            failed.append(module)
    
    if failed:
        print(f"\n{len(failed)} modules failed to import")
        return False
    else:
        print(f"\nAll {len(modules)} modules imported successfully!")
        return True

if __name__ == "__main__":
    success = check_imports()
    sys.exit(0 if success else 1)



