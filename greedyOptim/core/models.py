"""
Data models and dataclasses for the optimization system.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class TrainStatus(str, Enum):
    """Train operational status for schedule output."""
    REVENUE_SERVICE = "REVENUE_SERVICE"
    STANDBY = "STANDBY"
    MAINTENANCE = "MAINTENANCE"
    OUT_OF_SERVICE = "OUT_OF_SERVICE"


class MaintenanceType(str, Enum):
    """Types of maintenance."""
    SCHEDULED_INSPECTION = "SCHEDULED_INSPECTION"
    PREVENTIVE = "PREVENTIVE"
    CORRECTIVE = "CORRECTIVE"


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class StationStop:
    """A single station stop within a trip."""
    station_code: str
    station_name: str
    arrival_time: Optional[str]  # HH:MM format, None for origin
    departure_time: Optional[str]  # HH:MM format, None for destination
    distance_from_origin_km: float
    platform: Optional[int] = None
    
    def to_dict(self) -> Dict:
        result = {
            'station_code': self.station_code,
            'station_name': self.station_name,
            'distance_from_origin_km': self.distance_from_origin_km
        }
        if self.arrival_time:
            result['arrival_time'] = self.arrival_time
        if self.departure_time:
            result['departure_time'] = self.departure_time
        if self.platform:
            result['platform'] = self.platform
        return result


@dataclass
class Trip:
    """A single trip from origin to destination with all stops."""
    trip_id: str
    trip_number: int  # 1, 2, 3... within the block
    direction: str  # "UP" (towards Pettah) or "DOWN" (towards Aluva)
    origin: str
    destination: str
    departure_time: str
    arrival_time: str
    stops: List[StationStop] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'trip_id': self.trip_id,
            'trip_number': self.trip_number,
            'direction': self.direction,
            'origin': self.origin,
            'destination': self.destination,
            'departure_time': self.departure_time,
            'arrival_time': self.arrival_time,
            'stops': [s.to_dict() for s in self.stops]
        }


@dataclass
class ServiceBlock:
    """A service block represents a continuous operating period for a train."""
    block_id: str
    departure_time: str  # HH:MM format
    origin: str
    destination: str
    trip_count: int
    estimated_km: int
    # Enhanced fields
    journey_time_minutes: Optional[float] = None
    trips: List[Trip] = field(default_factory=list)  # Detailed trip breakdown
    period: Optional[str] = None  # morning_peak, midday, evening_peak, late_evening
    is_peak: bool = False
    
    def to_dict(self) -> Dict:
        result = {
            'block_id': self.block_id,
            'departure_time': self.departure_time,
            'origin': self.origin,
            'destination': self.destination,
            'trip_count': self.trip_count,
            'estimated_km': self.estimated_km
        }
        if self.journey_time_minutes:
            result['journey_time_minutes'] = self.journey_time_minutes
        if self.period:
            result['period'] = self.period
        result['is_peak'] = self.is_peak
        if self.trips:
            result['trips'] = [t.to_dict() for t in self.trips]
        return result


@dataclass
class ScheduleTrainset:
    """Complete trainset information in the schedule."""
    trainset_id: str
    status: TrainStatus
    readiness_score: float
    daily_km_allocation: int
    cumulative_km: int
    
    # For REVENUE_SERVICE
    assigned_duty: Optional[str] = None
    priority_rank: Optional[int] = None
    service_blocks: List[ServiceBlock] = field(default_factory=list)
    stabling_bay: Optional[str] = None
    
    # For STANDBY
    standby_reason: Optional[str] = None
    
    # For MAINTENANCE
    maintenance_type: Optional[MaintenanceType] = None
    ibl_bay: Optional[str] = None
    estimated_completion: Optional[str] = None
    
    # Alerts
    alerts: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        result = {
            'trainset_id': self.trainset_id,
            'status': self.status.value,
            'readiness_score': self.readiness_score,
            'daily_km_allocation': self.daily_km_allocation,
            'cumulative_km': self.cumulative_km,
            'alerts': self.alerts
        }
        if self.assigned_duty:
            result['assigned_duty'] = self.assigned_duty
        if self.priority_rank is not None:
            result['priority_rank'] = self.priority_rank
        if self.service_blocks:
            result['service_blocks'] = [b.to_dict() for b in self.service_blocks]
        if self.stabling_bay:
            result['stabling_bay'] = self.stabling_bay
        if self.standby_reason:
            result['standby_reason'] = self.standby_reason
        if self.maintenance_type:
            result['maintenance_type'] = self.maintenance_type.value
        if self.ibl_bay:
            result['ibl_bay'] = self.ibl_bay
        if self.estimated_completion:
            result['estimated_completion'] = self.estimated_completion
        return result


@dataclass
class FleetSummary:
    """Summary statistics for the fleet."""
    total_trainsets: int
    revenue_service: int
    standby: int
    maintenance: int
    availability_percent: float
    
    def to_dict(self) -> Dict:
        return {
            'total_trainsets': self.total_trainsets,
            'revenue_service': self.revenue_service,
            'standby': self.standby,
            'maintenance': self.maintenance,
            'availability_percent': self.availability_percent
        }


@dataclass
class ScheduleAlert:
    """Alert or warning in the schedule."""
    trainset_id: str
    severity: AlertSeverity
    alert_type: str
    message: str
    
    def to_dict(self) -> Dict:
        return {
            'trainset_id': self.trainset_id,
            'severity': self.severity.value,
            'alert_type': self.alert_type,
            'message': self.message
        }


@dataclass
class OptimizationMetrics:
    """Metrics about the optimization."""
    fitness_score: float
    method: str
    mileage_variance_coefficient: float
    total_planned_km: int
    optimization_runtime_ms: int
    
    def to_dict(self) -> Dict:
        return {
            'fitness_score': self.fitness_score,
            'method': self.method,
            'mileage_variance_coefficient': self.mileage_variance_coefficient,
            'total_planned_km': self.total_planned_km,
            'optimization_runtime_ms': self.optimization_runtime_ms
        }


@dataclass
class ScheduleResult:
    """Complete schedule result with all trainsets and service blocks."""
    schedule_id: str
    generated_at: str
    valid_from: str
    valid_until: str
    depot: str
    
    trainsets: List[ScheduleTrainset]
    fleet_summary: FleetSummary
    optimization_metrics: OptimizationMetrics
    alerts: List[ScheduleAlert] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization and benchmarks."""
        return {
            'schedule_id': self.schedule_id,
            'generated_at': self.generated_at,
            'valid_from': self.valid_from,
            'valid_until': self.valid_until,
            'depot': self.depot,
            'trainsets': [ts.to_dict() for ts in self.trainsets],
            'fleet_summary': self.fleet_summary.to_dict(),
            'optimization_metrics': self.optimization_metrics.to_dict(),
            'alerts': [a.to_dict() for a in self.alerts]
        }


@dataclass
class OptimizationResult:
    """Result of a trainset scheduling optimization run."""
    selected_trainsets: List[str]
    standby_trainsets: List[str]
    maintenance_trainsets: List[str]
    objectives: Dict[str, float]
    fitness_score: float
    explanation: Dict[str, str]
    # Block assignments: maps trainset_id -> list of block_ids
    service_block_assignments: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class OptimizationConfig:
    """Configuration parameters for optimization algorithms."""
    required_service_trains: int = 20
    min_standby: int = 2
    population_size: int = 100
    generations: int = 200
    iterations: int = 15  # For SA, CMA-ES, PSO (configurable)
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elite_size: int = 5
    optimize_block_assignment: bool = True  # Enable block assignment optimization


@dataclass
class TrainsetConstraints:
    """Hard and soft constraints for a trainset."""
    has_valid_certificates: bool
    has_critical_jobs: bool
    component_warnings: List[str]
    maintenance_due: bool
    mileage: int
    last_service_days: int