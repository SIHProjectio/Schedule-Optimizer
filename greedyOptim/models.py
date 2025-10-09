"""
Data models and dataclasses for the optimization system.
"""
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class OptimizationResult:
    """Result of a trainset scheduling optimization run."""
    selected_trainsets: List[str]
    standby_trainsets: List[str]
    maintenance_trainsets: List[str]
    objectives: Dict[str, float]
    fitness_score: float
    explanation: Dict[str, str]


@dataclass
class OptimizationConfig:
    """Configuration parameters for optimization algorithms."""
    required_service_trains: int = 20
    min_standby: int = 2
    population_size: int = 100
    generations: int = 200
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elite_size: int = 5


@dataclass
class TrainsetConstraints:
    """Hard and soft constraints for a trainset."""
    has_valid_certificates: bool
    has_critical_jobs: bool
    component_warnings: List[str]
    maintenance_due: bool
    mileage: int
    last_service_days: int