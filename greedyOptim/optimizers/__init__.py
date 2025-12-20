"""
Optimizers module for greedyOptim package.
Contains all optimization algorithms.
"""

from .base_optimizer import BaseOptimizer

from .genetic_algorithm import GeneticAlgorithmOptimizer

from .advanced_optimizers import (
    CMAESOptimizer,
    ParticleSwarmOptimizer,
    SimulatedAnnealingOptimizer
)

from .hybrid_optimizers import (
    MultiObjectiveOptimizer,
    AdaptiveOptimizer,
    EnsembleOptimizer,
    HyperParameterOptimizer,
    optimize_with_hybrid_methods
)

# Optional OR-Tools integration
try:
    from .ortools_optimizers import (
        optimize_with_ortools,
        check_ortools_availability,
        CPSATOptimizer,
        MIPOptimizer
    )
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False
    optimize_with_ortools = None
    check_ortools_availability = None
    CPSATOptimizer = None
    MIPOptimizer = None

__all__ = [
    'BaseOptimizer',
    'GeneticAlgorithmOptimizer',
    'CMAESOptimizer', 'ParticleSwarmOptimizer', 'SimulatedAnnealingOptimizer',
    'MultiObjectiveOptimizer', 'AdaptiveOptimizer', 'EnsembleOptimizer',
    'HyperParameterOptimizer', 'optimize_with_hybrid_methods',
    'ORTOOLS_AVAILABLE'
]

if ORTOOLS_AVAILABLE:
    __all__.extend([
        'optimize_with_ortools', 'check_ortools_availability',
        'CPSATOptimizer', 'MIPOptimizer'
    ])
