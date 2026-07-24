"""qradiomics.classification — tabular binary classification utilities.

Public-package model registry and benchmark runner. All models, including
the extended/AutoML backends (XGB, LGBM, FLAML, TPOT), are registered
directly in ``qradiomics.classification.registry`` — none of them live in
``qradiomics_private``. XGB/LGBM/FLAML/TPOT are optional imports: each
builder raises a helpful ``ImportError`` (with an install hint) if its
backing package is not installed.
"""
from qradiomics.classification.registry import MODEL_REGISTRY, build_model
from qradiomics.classification.benchmark import BenchmarkResult, cross_val_benchmark

__all__ = ["MODEL_REGISTRY", "build_model", "BenchmarkResult", "cross_val_benchmark"]
