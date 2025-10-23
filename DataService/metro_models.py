"""
Data models for Metro Train Scheduling System
Comprehensive models matching the KMRL (Kochi Metro Rail Limited) structure
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Literal
from datetime import datetime, time
from enum import Enum


class TrainStatus(str, Enum):
    """Train operational status"""
    REVENUE_SERVICE = "REVENUE_SERVICE"
    STANDBY = "STANDBY"
    MAINTENANCE = "MAINTENANCE"
    CLEANING = "CLEANING"
    OUT_OF_SERVICE = "OUT_OF_SERVICE"


class CertificateStatus(str, Enum):
    """Fitness certificate status"""
    VALID = "VALID"
    EXPIRING_SOON = "EXPIRING_SOON"
    EXPIRED = "EXPIRED"


class MaintenanceType(str, Enum):
    """Types of maintenance operations"""
    SCHEDULED_INSPECTION = "SCHEDULED_INSPECTION"
    PREVENTIVE = "PREVENTIVE"
    CORRECTIVE = "CORRECTIVE"
    BREAKDOWN = "BREAKDOWN"


class Severity(str, Enum):
    """Alert severity levels"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class FitnessCertificate(BaseModel):
    """Individual fitness certificate"""
    valid_until: str  # ISO format date
    status: CertificateStatus


class FitnessCertificates(BaseModel):
    """All fitness certificates for a trainset"""
    rolling_stock: FitnessCertificate
    signalling: FitnessCertificate
    telecom: FitnessCertificate


class JobCards(BaseModel):
    """Job cards and maintenance tasks"""
    open: int
    blocking: List[str] = Field(default_factory=list)


class Branding(BaseModel):
    """Advertising/branding information"""
    advertiser: str
    contract_hours_remaining: int
    exposure_priority: Literal["NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL"]


class ServiceBlock(BaseModel):
    """A service block represents a continuous operating period"""
    block_id: str
    departure_time: str  # HH:MM format
    origin: str
    destination: str
    trip_count: int  # Number of round trips in this block
    estimated_km: int


class Trainset(BaseModel):
    """Complete trainset information"""
    trainset_id: str
    status: TrainStatus
    priority_rank: Optional[int] = None
    assigned_duty: Optional[str] = None
    
    # Service blocks for revenue service trains
    service_blocks: List[ServiceBlock] = Field(default_factory=list)
    
    # Maintenance information
    maintenance_type: Optional[MaintenanceType] = None
    ibl_bay: Optional[str] = None  # Inspection/Berthing Location
    estimated_completion: Optional[str] = None
    
    # Cleaning information
    cleaning_bay: Optional[str] = None
    cleaning_type: Optional[str] = None
    scheduled_service_start: Optional[str] = None
    
    # Operational metrics
    daily_km_allocation: int
    cumulative_km: int
    stabling_bay: Optional[str] = None
    
    # Compliance and health
    fitness_certificates: FitnessCertificates
    job_cards: JobCards
    
    # Branding
    branding: Optional[Branding] = None
    
    # Computed scores
    readiness_score: float = Field(ge=0.0, le=1.0)
    constraints_met: bool
    
    # Alerts
    alerts: List[str] = Field(default_factory=list)
    standby_reason: Optional[str] = None


class FleetSummary(BaseModel):
    """Summary statistics for the entire fleet"""
    total_trainsets: int
    revenue_service: int
    standby: int
    maintenance: int
    cleaning: int
    availability_percent: float


class OptimizationMetrics(BaseModel):
    """Metrics about the optimization result"""
    mileage_variance_coefficient: float
    avg_readiness_score: float
    branding_sla_compliance: float
    shunting_movements_required: int
    total_planned_km: int
    fitness_expiry_violations: int
    optimization_runtime_ms: int = 0


class Alert(BaseModel):
    """Alert or conflict in the schedule"""
    trainset_id: str
    severity: Severity
    type: str
    message: str


class DecisionRationale(BaseModel):
    """Explanation of optimization decisions"""
    algorithm_version: str
    objective_weights: Dict[str, float]
    constraint_violations: int
    optimization_runtime_ms: int


class DaySchedule(BaseModel):
    """Complete daily schedule for all trains"""
    schedule_id: str
    generated_at: str  # ISO format with timezone
    valid_from: str  # ISO format
    valid_until: str  # ISO format
    depot: str
    
    trainsets: List[Trainset]
    fleet_summary: FleetSummary
    optimization_metrics: OptimizationMetrics
    conflicts_and_alerts: List[Alert]
    decision_rationale: DecisionRationale


class Station(BaseModel):
    """Metro station information"""
    station_id: str
    name: str
    sequence: int  # Position in the line (1-25)
    distance_from_origin_km: float
    avg_dwell_time_seconds: int = 30  # Average stopping time


class Route(BaseModel):
    """Single metro line route"""
    route_id: str
    name: str
    stations: List[Station]
    total_distance_km: float
    avg_speed_kmh: float = 35
    turnaround_time_minutes: int = 10  # Time needed at terminals


class OperationalHours(BaseModel):
    """Service hours configuration"""
    start_time: time = time(5, 0)  # 5:00 AM
    end_time: time = time(23, 0)  # 11:00 PM
    peak_hours: List[tuple] = Field(
        default_factory=lambda: [
            (time(7, 0), time(10, 0)),   # Morning peak
            (time(17, 0), time(20, 0))   # Evening peak
        ]
    )
    peak_frequency_minutes: int = 5  # Train every 5 minutes during peak
    off_peak_frequency_minutes: int = 10  # Train every 10 minutes off-peak


class TrainHealthStatus(BaseModel):
    """Health status for optimization"""
    trainset_id: str
    is_fully_healthy: bool
    available_hours: Optional[List[tuple]] = None  # (start_hour, end_hour) if partial
    unavailable_reason: Optional[str] = None
    cumulative_mileage: int
    days_since_maintenance: int
    component_health: Dict[str, float]  # Component: health_score (0-1)


class ScheduleRequest(BaseModel):
    """Request for schedule generation"""
    date: str  # YYYY-MM-DD
    num_trains: int = Field(default=25, ge=15, le=40)
    num_stations: int = Field(default=25, ge=10, le=50)
    route_name: str = "Aluva-Pettah Line"
    depot_name: str = "Muttom_Depot"
    
    # Optional: override train health
    train_health_overrides: Optional[List[TrainHealthStatus]] = None
    
    # Optimization parameters
    min_service_trains: int = 20
    min_standby_trains: int = 2
    max_daily_km_per_train: int = 300
    balance_mileage: bool = True
    prioritize_branding: bool = True
