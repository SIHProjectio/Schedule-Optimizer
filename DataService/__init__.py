"""
DataService - Metro Train Scheduling Data Generation and API

Structured submodules:
- core: Data models, enums, utilities, and configuration
- generators: Synthetic data generation
- optimizers: Schedule optimization algorithms
- api: FastAPI routes and endpoints
"""

# Core models and utilities
from .core.models import (
    # Enums
    TrainStatus, CertificateStatus, MaintenanceType, Severity,
    # Models
    DaySchedule, Trainset, ServiceBlock, ScheduleRequest,
    Route, Station, TrainHealthStatus, OperationalHours,
    FitnessCertificates, JobCards, Branding,
    FleetSummary, OptimizationMetrics, Alert, DecisionRationale,
)
from .core.utils import (
    time_to_minutes, minutes_to_time, time_range_overlap,
    is_peak_hour, format_iso_datetime, parse_iso_date,
)
from .core.config_loader import ConfigLoader, get_config_loader, reload_config

# Generators
from .generators.metro_generator import MetroDataGenerator
from .generators.enhanced_generator import EnhancedMetroDataGenerator
from .generators.synthetic_base import MetroSyntheticDataGenerator

# Optimizers
from .optimizers.schedule_optimizer import MetroScheduleOptimizer

# API
from .api import app

__all__ = [
    # Core - Enums
    'TrainStatus',
    'CertificateStatus',
    'MaintenanceType',
    'Severity',
    
    # Core - Models
    'DaySchedule',
    'Trainset',
    'TrainStatus',
    'ServiceBlock',
    'ScheduleRequest',
    'Route',
    'Station',
    'TrainHealthStatus',
    'FitnessCertificates',
    'JobCards',
    'Branding',
    'OperationalHours',
    'FleetSummary',
    'OptimizationMetrics',
    'Alert',
    'DecisionRationale',
    
    # Core - Utils
    'time_to_minutes',
    'minutes_to_time',
    'time_range_overlap',
    'is_peak_hour',
    'format_iso_datetime',
    'parse_iso_date',
    
    # Config
    'ConfigLoader',
    'get_config_loader',
    'reload_config',
    
    # Generators
    'MetroDataGenerator',
    'EnhancedMetroDataGenerator',
    'MetroSyntheticDataGenerator',
    
    # Optimizers
    'MetroScheduleOptimizer',
    
    # API
    'app',
]

__version__ = '2.0.0'  # Updated for restructured architecture
