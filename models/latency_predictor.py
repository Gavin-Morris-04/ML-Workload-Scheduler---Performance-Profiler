"""Learned latency predictor for scheduling optimization."""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from typing import Optional, Dict
import json


class LatencyPredictor:
    """Learned latency predictor using regression."""
    
    def __init__(self, model_type: str = "random_forest"):
        self.model_type = model_type
        self.model = None
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        self.feature_names = None
        self.is_fitted = False
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for training/prediction."""
        # Categorical features
        categorical = ['workload_name', 'backend', 'precision']
        X_cat = self.encoder.fit_transform(df[categorical]) if not self.is_fitted else \
                self.encoder.transform(df[categorical])
        
        # Numerical features
        numerical = ['batch_size']
        X_num = df[numerical].values
        
        # Combine
        X = np.hstack([X_num, X_cat])
        return X
    
    def train(self, data_path: str = "results/profiles.csv") -> None:
        """Train the predictor on profiling data."""
        if not Path(data_path).exists():
            raise FileNotFoundError(f"Profile data not found at {data_path}. Run profiling first.")
        
        df = pd.read_csv(data_path)
        
        # Prepare features
        if not self.is_fitted:
            X = self.prepare_features(df)
            self.is_fitted = True
        else:
            X = self.prepare_features(df)
        
        y = df['service_ms'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        if self.model_type == "random_forest":
            self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        elif self.model_type == "gradient_boosting":
            self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        print(f"Train R²: {train_score:.4f}, Test R²: {test_score:.4f}")
    
    def predict(self, workload_name: str, backend: str, precision: str, batch_size: int) -> float:
        """Predict service time in milliseconds."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        
        # Create feature vector
        df = pd.DataFrame([{
            'workload_name': workload_name,
            'backend': backend,
            'precision': precision,
            'batch_size': batch_size
        }])
        
        X = self.prepare_features(df)
        prediction = self.model.predict(X)[0]
        
        return max(0.0, prediction)  # Ensure non-negative
    
    def predict_job(self, job) -> float:
        """Predict service time for a job object."""
        return self.predict(job.workload_name, job.backend, job.precision, job.batch_size)
    
    def save(self, path: str) -> None:
        """Save the trained model."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'encoder': self.encoder,
                'is_fitted': self.is_fitted,
                'model_type': self.model_type
            }, f)
    
    @classmethod
    def load(cls, path: str) -> 'LatencyPredictor':
        """Load a trained model."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        predictor = cls(model_type=data['model_type'])
        predictor.model = data['model']
        predictor.encoder = data['encoder']
        predictor.is_fitted = data['is_fitted']
        
        return predictor


def collect_profiling_data(
    workloads: Dict,
    batch_sizes: list = [1, 8, 32, 128],
    backends: list = ["pytorch", "onnx"],
    precisions: list = ["fp32", "fp16"],
    output_path: str = "results/profiles.csv"
):
    """Collect profiling data for all workload configurations."""
    import sys
    from pathlib import Path
    
    # Add parent directory to path for imports
    script_dir = Path(__file__).parent.parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    
    from profiler.profiler import Profiler
    import pandas as pd
    
    profiler = Profiler(warmup_iterations=3, timed_iterations=10)
    results = []
    
    for workload_name, workload_class in workloads.items():
        print(f"Profiling {workload_name}...")
        
        for backend in backends:
            for precision in precisions:
                for batch_size in batch_sizes:
                    try:
                        # Load workload
                        w = workload_class()
                        device = "cpu"  # Or "cuda" if available
                        
                        onnx_path = None
                        if backend == "onnx":
                            onnx_path = f"models/onnx/{workload_name}.onnx"
                            if not Path(onnx_path).exists():
                                print(f"  Skipping {workload_name} {backend} - ONNX not found")
                                continue
                        
                        w.load(device, precision, backend, onnx_path)
                        
                        # Profile
                        service_ms, throughput = profiler.profile(w, batch_size)
                        
                        results.append({
                            'workload_name': workload_name,
                            'backend': backend,
                            'precision': precision,
                            'batch_size': batch_size,
                            'service_ms': service_ms,
                            'throughput_sps': throughput
                        })
                        
                        print(f"  {backend}/{precision}/bs{batch_size}: {service_ms:.2f}ms")
                    except Exception as e:
                        print(f"  Error profiling {workload_name} {backend} {precision} bs{batch_size}: {e}")
    
    # Save results
    df = pd.DataFrame(results)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved profiling data to {output_path}")
    
    return df

