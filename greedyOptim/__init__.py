"""
Trainset Scheduling Optimization Package

This package provides multi-objective optimization algorithms for metro trainset scheduling.
It includes various optimization methods like Genetic Algorithm, CMA-ES, PSO, and Simulated Annealing.

Main classes:
- TrainsetSchedulingOptimizer: Main interface for optimization
- OptimizationConfig: Configuration parameters
- OptimizationResult: Result container
- BaseOptimizer: Base class for all optimizers

Usage:
    from greedyOptim import optimize_trainset_schedule, OptimizationConfig
    
    config = OptimizationConfig(required_service_trains=20, min_standby=3)
    result = optimize_trainset_schedule(data, method='ga', config=config)
"""

# Utilities (shared across modules)
from .utils import (
    normalize_certificate_status,
    normalize_component_status,
    normalize_operational_status,
    decode_solution,
    create_block_assignment,
    extract_solution_groups,
    build_block_assignments_dict
)

# Core data models
from .models import (
    OptimizationResult, OptimizationConfig, TrainsetConstraints,
    ScheduleResult, ScheduleTrainset, ServiceBlock, FleetSummary,
    OptimizationMetrics, ScheduleAlert, TrainStatus, MaintenanceType, AlertSeverity,
    StationStop, Trip
)

# Base optimizer class
from .base_optimizer import BaseOptimizer

# Evaluator
from .evaluator import TrainsetSchedulingEvaluator

# Optimizers
from .genetic_algorithm import GeneticAlgorithmOptimizer
from .advanced_optimizers import CMAESOptimizer, ParticleSwarmOptimizer, SimulatedAnnealingOptimizer
from .hybrid_optimizers import (
    MultiObjectiveOptimizer, AdaptiveOptimizer, EnsembleOptimizer,
    HyperParameterOptimizer, optimize_with_hybrid_methods
)

# Main interface
from .scheduler import (
    TrainsetSchedulingOptimizer,
    optimize_trainset_schedule,
    compare_optimization_methods
)

# Error handling
from .error_handling import (
    safe_optimize, DataValidator, OptimizationError,
    DataValidationError, ConstraintViolationError, ConfigurationError
)

# Schedule generation
from .schedule_generator import ScheduleGenerator, generate_schedule_from_result

# Station/route data
from .station_loader import (
    StationDataLoader, get_station_loader, get_route_distance, get_terminals,
    Station, RouteInfo
)
from .service_blocks import ServiceBlockGenerator

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

__version__ = "2.0.0"
__author__ = "Metro Optimization Team"

__all__ = [
    # Utilities
    'normalize_certificate_status',
    'normalize_component_status',
    'normalize_operational_status',
    'decode_solution',
    'create_block_assignment',
    'extract_solution_groups',
    'build_block_assignments_dict',
    # Models
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
    'StationStop',
    'Trip',
    # Base classes
    'BaseOptimizer',
    'TrainsetSchedulingEvaluator',
    # Optimizers
    'GeneticAlgorithmOptimizer',
    'CMAESOptimizer',
    'ParticleSwarmOptimizer',
    'SimulatedAnnealingOptimizer',
    'MultiObjectiveOptimizer',
    'AdaptiveOptimizer', 
    'EnsembleOptimizer',
    'HyperParameterOptimizer',
    # Main interface
    'TrainsetSchedulingOptimizer',
    'optimize_trainset_schedule',
    'compare_optimization_methods',
    'optimize_with_hybrid_methods',
    # Error handling
    'safe_optimize',
    'DataValidator',
    'OptimizationError',
    'DataValidationError',
    'ConstraintViolationError',
    'ConfigurationError',
    # Schedule generation
    'ScheduleGenerator',
    'generate_schedule_from_result',
    # Station/route
    'StationDataLoader',
    'get_station_loader',
    'get_route_distance',
    'get_terminals',
    'Station',
    'RouteInfo',
    'ServiceBlockGenerator',
    # Flags
    'ORTOOLS_AVAILABLE'
]

# Add OR-Tools to exports if available
if ORTOOLS_AVAILABLE:
    __all__.extend([
        'optimize_with_ortools',
        'check_ortools_availability', 
        'CPSATOptimizer',
        'MIPOptimizer'
    ])
