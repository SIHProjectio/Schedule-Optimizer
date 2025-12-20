"""
greedyOptim - Train Scheduling Optimization Package

This package provides optimization algorithms for trainset scheduling,
with support for various metaheuristic and exact methods.

Submodules:
    - core: Data models, utilities, error handling
    - optimizers: Optimization algorithms (GA, CMA-ES, PSO, SA, hybrid, OR-Tools)
    - scheduling: Scheduler, evaluator, schedule generation
    - routing: Station data loading and route utilities
"""

# Re-export core components for backward compatibility
from greedyOptim.core import (
    # Models
    OptimizationResult, OptimizationConfig, TrainsetConstraints,
    ScheduleResult, ScheduleTrainset, ServiceBlock, FleetSummary,
    OptimizationMetrics, ScheduleAlert, TrainStatus, MaintenanceType, AlertSeverity,
    StationStop, Trip,
    # Utils
    normalize_certificate_status, normalize_component_status, normalize_operational_status,
    decode_solution, create_block_assignment, extract_solution_groups,
    build_block_assignments_dict, repair_block_assignment, mutate_block_assignment,
    CERTIFICATE_STATUS_MAP, COMPONENT_STATUS_MAP, OPERATIONAL_STATUS_MAP,
    # Error handling
    OptimizationError, DataValidationError, ConstraintViolationError, ConfigurationError,
    DataValidator, ErrorHandler
)

# Re-export scheduling components
from greedyOptim.scheduling import (
    TrainsetSchedulingEvaluator,
    TrainsetSchedulingOptimizer,
    optimize_trainset_schedule,
    compare_optimization_methods,
    ScheduleGenerator,
    generate_schedule_from_result,
    ServiceBlockGenerator,
    create_service_blocks_for_schedule
)

# Re-export optimizers
from greedyOptim.optimizers import (
    BaseOptimizer,
    GeneticAlgorithmOptimizer,
    CMAESOptimizer, ParticleSwarmOptimizer, SimulatedAnnealingOptimizer,
    MultiObjectiveOptimizer, AdaptiveOptimizer, EnsembleOptimizer,
    HyperParameterOptimizer, optimize_with_hybrid_methods,
    ORTOOLS_AVAILABLE
)

# Re-export routing
from greedyOptim.routing import (
    StationDataLoader,
    get_station_loader,
    get_route_distance,
    get_terminals,
    Station,
    RouteInfo
)

# Conditional OR-Tools exports
if ORTOOLS_AVAILABLE:
    from greedyOptim.optimizers import (
        optimize_with_ortools,
        check_ortools_availability,
        CPSATOptimizer,
        MIPOptimizer
    )

__all__ = [
    # Core models
    'OptimizationResult', 'OptimizationConfig', 'TrainsetConstraints',
    'ScheduleResult', 'ScheduleTrainset', 'ServiceBlock', 'FleetSummary',
    'OptimizationMetrics', 'ScheduleAlert', 'TrainStatus', 'MaintenanceType', 'AlertSeverity',
    'StationStop', 'Trip',
    # Core utils
    'normalize_certificate_status', 'normalize_component_status', 'normalize_operational_status',
    'decode_solution', 'create_block_assignment', 'extract_solution_groups',
    'build_block_assignments_dict', 'repair_block_assignment', 'mutate_block_assignment',
    # Error handling
    'OptimizationError', 'DataValidationError', 'ConstraintViolationError', 'ConfigurationError',
    'DataValidator', 'ErrorHandler',
    # Scheduling
    'TrainsetSchedulingEvaluator', 'TrainsetSchedulingOptimizer',
    'optimize_trainset_schedule', 'compare_optimization_methods',
    'ScheduleGenerator', 'generate_schedule_from_result',
    'ServiceBlockGenerator', 'create_service_blocks_for_schedule',
    # Optimizers
    'BaseOptimizer', 'GeneticAlgorithmOptimizer',
    'CMAESOptimizer', 'ParticleSwarmOptimizer', 'SimulatedAnnealingOptimizer',
    'MultiObjectiveOptimizer', 'AdaptiveOptimizer', 'EnsembleOptimizer',
    'HyperParameterOptimizer', 'optimize_with_hybrid_methods',
    'ORTOOLS_AVAILABLE',
    # Routing
    'StationDataLoader', 'get_station_loader', 'get_route_distance', 'get_terminals',
    'Station', 'RouteInfo'
]
