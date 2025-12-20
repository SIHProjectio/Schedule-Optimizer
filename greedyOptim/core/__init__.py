"""
Core module for greedyOptim package.
Contains data models, utilities, and error handling.
"""

from .models import (
    OptimizationResult, OptimizationConfig, TrainsetConstraints,
    ScheduleResult, ScheduleTrainset, ServiceBlock, FleetSummary,
    OptimizationMetrics, ScheduleAlert, TrainStatus, MaintenanceType, AlertSeverity,
    StationStop, Trip
)

from .utils import (
    normalize_certificate_status,
    normalize_component_status,
    normalize_operational_status,
    decode_solution,
    create_block_assignment,
    extract_solution_groups,
    build_block_assignments_dict,
    repair_block_assignment,
    mutate_block_assignment,
    CERTIFICATE_STATUS_MAP,
    COMPONENT_STATUS_MAP,
    OPERATIONAL_STATUS_MAP
)

from .error_handling import (
    OptimizationError,
    DataValidationError,
    ConstraintViolationError,
    ConfigurationError,
    DataValidator,
    ErrorHandler,
    safe_optimize
)

__all__ = [
    # Models
    'OptimizationResult', 'OptimizationConfig', 'TrainsetConstraints',
    'ScheduleResult', 'ScheduleTrainset', 'ServiceBlock', 'FleetSummary',
    'OptimizationMetrics', 'ScheduleAlert', 'TrainStatus', 'MaintenanceType', 'AlertSeverity',
    'StationStop', 'Trip',
    # Utils
    'normalize_certificate_status', 'normalize_component_status', 'normalize_operational_status',
    'decode_solution', 'create_block_assignment', 'extract_solution_groups',
    'build_block_assignments_dict', 'repair_block_assignment', 'mutate_block_assignment',
    'CERTIFICATE_STATUS_MAP', 'COMPONENT_STATUS_MAP', 'OPERATIONAL_STATUS_MAP',
    # Error handling
    'OptimizationError', 'DataValidationError', 'ConstraintViolationError', 'ConfigurationError',
    'DataValidator', 'ErrorHandler', 'safe_optimize'
]
