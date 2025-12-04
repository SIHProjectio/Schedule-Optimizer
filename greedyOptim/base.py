"""
Base module for trainset scheduling optimization.

This module provides backward compatibility with the legacy API.
For new code, use the modular imports from greedyOptim directly:

    from greedyOptim import optimize_trainset_schedule, OptimizationConfig
    from greedyOptim import GeneticAlgorithmOptimizer, CMAESOptimizer
"""
from .scheduler import optimize_trainset_schedule, compare_optimization_methods
from .models import OptimizationResult, OptimizationConfig

__all__ = [
    'optimize_trainset_schedule',
    'compare_optimization_methods',
    'OptimizationResult',
    'OptimizationConfig'
]
