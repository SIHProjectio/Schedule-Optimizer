"""
Trainset Scheduling Optimization Package

This package provides multi-objective optimization algorithms for metro trainset scheduling.
It includes various optimization methods like Genetic Algorithm, CMA-ES, PSO, and Simulated Annealing.

Main classes:
- TrainsetSchedulingOptimizer: Main interface for optimization
- OptimizationConfig: Configuration parameters
- OptimizationResult: Result container

Usage:
    from greedyOptim import optimize_trainset_schedule, OptimizationConfig
    
    config = OptimizationConfig(required_service_trains=20, min_standby=3)
    result = optimize_trainset_schedule(data, method='ga', config=config)
"""

from .models import (
    OptimizationResult, OptimizationConfig, TrainsetConstraints,
    ScheduleResult, ScheduleTrainset, ServiceBlock, FleetSummary,
    OptimizationMetrics, ScheduleAlert, TrainStatus, MaintenanceType, AlertSeverity
)
from .evaluator import TrainsetSchedulingEvaluator
from .genetic_algorithm import GeneticAlgorithmOptimizer
from .advanced_optimizers import CMAESOptimizer, ParticleSwarmOptimizer, SimulatedAnnealingOptimizer
from .hybrid_optimizers import (
    MultiObjectiveOptimizer, AdaptiveOptimizer, EnsembleOptimizer,
    HyperParameterOptimizer, optimize_with_hybrid_methods
)
from .scheduler import (
    TrainsetSchedulingOptimizer,
    optimize_trainset_schedule,
    compare_optimization_methods
)
from .error_handling import (
    safe_optimize, DataValidator, OptimizationError,
    DataValidationError, ConstraintViolationError, ConfigurationError
)
from .schedule_generator import ScheduleGenerator, generate_schedule_from_result

# Optional OR-Tools integration
try:
    from .ortools_optimizers import (
        optimize_with_ortools, check_ortools_availability,
        CPSATOptimizer, MIPOptimizer
    )
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False
    optimize_with_ortools = None
    check_ortools_availability = None
    CPSATOptimizer = None
    MIPOptimizer = None

__version__ = "1.0.0"
__author__ = "Metro Optimization Team"

__all__ = [
    'OptimizationResult',
    'OptimizationConfig', 
    'TrainsetConstraints',
    'ScheduleResult',
    'ScheduleTrainset',
    'ServiceBlock',
    'FleetSummary',
    'OptimizationMetrics',
    'ScheduleAlert',
    'TrainStatus',
    'MaintenanceType',
    'AlertSeverity',
    'TrainsetSchedulingEvaluator',
    'GeneticAlgorithmOptimizer',
    'CMAESOptimizer',
    'ParticleSwarmOptimizer',
    'SimulatedAnnealingOptimizer',
    'MultiObjectiveOptimizer',
    'AdaptiveOptimizer', 
    'EnsembleOptimizer',
    'HyperParameterOptimizer',
    'TrainsetSchedulingOptimizer',
    'optimize_trainset_schedule',
    'compare_optimization_methods',
    'optimize_with_hybrid_methods',
    'safe_optimize',
    'DataValidator',
    'OptimizationError',
    'DataValidationError',
    'ConstraintViolationError',
    'ConfigurationError',
    'ScheduleGenerator',
    'generate_schedule_from_result'
]

# Add OR-Tools to exports if available
if ORTOOLS_AVAILABLE:
    __all__.extend([
        'optimize_with_ortools',
        'check_ortools_availability', 
        'CPSATOptimizer',
        'MIPOptimizer'
    ])
