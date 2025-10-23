"""
DataService - Metro Train Scheduling Data Generation and API
"""

from .metro_models import (
    DaySchedule, Trainset, TrainStatus, ServiceBlock,
    ScheduleRequest, Route, Station, TrainHealthStatus,
    FitnessCertificates, JobCards, Branding
)
from .metro_data_generator import MetroDataGenerator
from .schedule_optimizer import MetroScheduleOptimizer

__all__ = [
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
    'MetroDataGenerator',
    'MetroScheduleOptimizer',
]

__version__ = '1.0.0'
