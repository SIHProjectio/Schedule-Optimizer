"""
Schedule Generator Module
Converts optimization results into complete schedules with service blocks.
"""
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from .models import (
    OptimizationResult, OptimizationConfig,
    ScheduleResult, ScheduleTrainset, ServiceBlock, FleetSummary,
    OptimizationMetrics, ScheduleAlert, TrainStatus, MaintenanceType, AlertSeverity,
    StationStop, Trip
)
from .service_blocks import ServiceBlockGenerator
from .station_loader import get_station_loader, StationDataLoader
from .utils import normalize_certificate_status, normalize_component_status


# Depot configuration
DEPOT_BAYS = [f"BAY-{str(i).zfill(2)}" for i in range(1, 16)]
IBL_BAYS = [f"IBL-{str(i).zfill(2)}" for i in range(1, 6)]

# Standby reasons
STANDBY_REASONS = ["MILEAGE_BALANCING", "EMERGENCY_BACKUP", "PEAK_HOUR_RESERVE", "CREW_UNAVAILABLE"]


class ScheduleGenerator:
    """Generates complete schedules from optimization results."""
    
    def __init__(self, data: Dict, config: Optional[OptimizationConfig] = None):
        """Initialize schedule generator.
        
        Args:
            data: Input data with trainset_status, fitness_certificates, component_health
            config: Optimization configuration
        """
        self.data = data
        self.config = config or OptimizationConfig()
        self.service_block_generator = ServiceBlockGenerator()
        
        # Initialize station loader for detailed trip generation
        try:
            self.station_loader = get_station_loader()
        except Exception:
            self.station_loader = None
        
        # Build lookups
        self._build_lookups()
    
    def _build_lookups(self):
        """Build lookup dictionaries for quick access."""
        self.status_map = {ts['trainset_id']: ts for ts in self.data['trainset_status']}
        
        # Fitness certificates
        self.cert_map = {}
        for cert in self.data.get('fitness_certificates', []):
            ts_id = cert['trainset_id']
            if ts_id not in self.cert_map:
                self.cert_map[ts_id] = []
            self.cert_map[ts_id].append(cert)
        
        # Component health
        self.health_map = {}
        for health in self.data.get('component_health', []):
            ts_id = health['trainset_id']
            if ts_id not in self.health_map:
                self.health_map[ts_id] = []
            self.health_map[ts_id].append(health)
    
    def generate_schedule(
        self,
        optimization_result: OptimizationResult,
        method: str = "ga",
        runtime_ms: int = 0,
        date: Optional[str] = None,
        depot: str = "Muttom_Depot"
    ) -> ScheduleResult:
        """Generate complete schedule from optimization result.
        
        Args:
            optimization_result: Result from optimizer with trainset allocations
            method: Optimization method used
            runtime_ms: Optimization runtime in milliseconds
            date: Schedule date (default: today)
            depot: Depot name
            
        Returns:
            Complete ScheduleResult with service blocks
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        now = datetime.now()
        
        # Generate schedule ID
        schedule_id = f"SCH-{date.replace('-', '')}-{random.randint(100, 999)}"
        
        # Generate trainset schedules
        trainsets = []
        alerts = []
        total_km = 0
        mileages = []
        
        # Check if we have optimized block assignments
        has_optimized_blocks = (
            optimization_result.service_block_assignments and 
            len(optimization_result.service_block_assignments) > 0
        )
        
        # Build block lookup for optimized assignments
        block_lookup = {}
        if has_optimized_blocks:
            all_blocks = self.service_block_generator.get_all_service_blocks()
            block_lookup = {b['block_id']: b for b in all_blocks}
        
        # Service trains
        num_service = len(optimization_result.selected_trainsets)
        for idx, ts_id in enumerate(optimization_result.selected_trainsets):
            # Get optimized blocks for this trainset if available
            assigned_block_ids = None
            if has_optimized_blocks:
                assigned_block_ids = optimization_result.service_block_assignments.get(ts_id, [])
            
            trainset, ts_alerts, km = self._generate_service_trainset(
                ts_id, idx, num_service, 
                assigned_block_ids=assigned_block_ids,
                block_lookup=block_lookup
            )
            trainsets.append(trainset)
            alerts.extend(ts_alerts)
            total_km += km
            mileages.append(km)
        
        # Standby trains
        for ts_id in optimization_result.standby_trainsets:
            trainset, ts_alerts = self._generate_standby_trainset(ts_id)
            trainsets.append(trainset)
            alerts.extend(ts_alerts)
        
        # Maintenance trains
        for ts_id in optimization_result.maintenance_trainsets:
            trainset, ts_alerts = self._generate_maintenance_trainset(ts_id)
            trainsets.append(trainset)
            alerts.extend(ts_alerts)
        
        # Calculate fleet summary
        fleet_summary = FleetSummary(
            total_trainsets=len(trainsets),
            revenue_service=len(optimization_result.selected_trainsets),
            standby=len(optimization_result.standby_trainsets),
            maintenance=len(optimization_result.maintenance_trainsets),
            availability_percent=round(
                (len(optimization_result.selected_trainsets) + len(optimization_result.standby_trainsets))
                / max(1, len(trainsets)) * 100, 1
            )
        )
        
        # Calculate mileage variance
        if len(mileages) > 1:
            import statistics
            mean_km = statistics.mean(mileages)
            std_km = statistics.stdev(mileages)
            variance_coeff = std_km / mean_km if mean_km > 0 else 0
        else:
            variance_coeff = 0
        
        # Optimization metrics
        opt_metrics = OptimizationMetrics(
            fitness_score=optimization_result.fitness_score,
            method=method,
            mileage_variance_coefficient=round(variance_coeff, 3),
            total_planned_km=total_km,
            optimization_runtime_ms=runtime_ms
        )
        
        return ScheduleResult(
            schedule_id=schedule_id,
            generated_at=now.isoformat(),
            valid_from=f"{date}T05:00:00+05:30",
            valid_until=f"{date}T23:00:00+05:30",
            depot=depot,
            trainsets=trainsets,
            fleet_summary=fleet_summary,
            optimization_metrics=opt_metrics,
            alerts=alerts
        )
    
    def _generate_trips_for_block(
        self,
        block_data: Dict,
        block_id: str
    ) -> List[Trip]:
        """Generate detailed trips with station stops for a service block.
        
        Args:
            block_data: Block data dictionary
            block_id: Block ID
            
        Returns:
            List of Trip objects with station stops
        """
        if not self.station_loader:
            return []
        
        trips = []
        origin = block_data['origin']
        destination = block_data['destination']
        trip_count = block_data.get('trip_count', 1)
        
        # Parse initial departure time
        dep_time_str = block_data['departure_time']
        current_time = datetime.strptime(dep_time_str, '%H:%M')
        
        # Get terminals for direction determination
        terminals = self.station_loader.terminals
        
        for trip_num in range(1, trip_count + 1):
            # Determine direction
            if origin == terminals[0]:  # Aluva
                direction = "DOWN"  # Towards Pettah
            else:
                direction = "UP"  # Towards Aluva
            
            # Get station sequence with times
            try:
                station_sequence = self.station_loader.get_station_sequence_for_trip(
                    origin, destination,
                    include_times=True,
                    departure_time=current_time.strftime('%H:%M')
                )
            except Exception:
                station_sequence = []
            
            # Convert to StationStop objects
            stops = []
            for station in station_sequence:
                stop = StationStop(
                    station_code=station.get('code', ''),
                    station_name=station['name'],
                    arrival_time=station.get('arrival_time'),
                    departure_time=station.get('departure_time'),
                    distance_from_origin_km=station.get('distance_from_origin_km', 0),
                    platform=random.choice([1, 2])  # Random platform assignment
                )
                stops.append(stop)
            
            # Calculate arrival time at destination
            if stops:
                arrival_time = stops[-1].arrival_time or current_time.strftime('%H:%M')
            else:
                # Fallback calculation
                journey_minutes = block_data.get('journey_time_minutes', 53)
                arrival = current_time + timedelta(minutes=journey_minutes)
                arrival_time = arrival.strftime('%H:%M')
            
            trip = Trip(
                trip_id=f"{block_id}-T{trip_num:02d}",
                trip_number=trip_num,
                direction=direction,
                origin=origin,
                destination=destination,
                departure_time=current_time.strftime('%H:%M'),
                arrival_time=arrival_time,
                stops=stops
            )
            trips.append(trip)
            
            # Calculate next trip start time (arrival + turnaround)
            journey_minutes = block_data.get('journey_time_minutes', 53)
            turnaround_minutes = 3  # Turnaround at terminal
            total_minutes = journey_minutes + turnaround_minutes
            current_time = current_time + timedelta(minutes=total_minutes)
            
            # Swap origin and destination for return trip
            origin, destination = destination, origin
        
        return trips
    
    def _generate_service_trainset(
        self,
        trainset_id: str,
        index: int,
        num_service: int,
        assigned_block_ids: list = None,
        block_lookup: dict = None
    ) -> tuple:
        """Generate schedule for a service trainset.
        
        Args:
            trainset_id: ID of the trainset
            index: Index in service trainsets list
            num_service: Total number of service trainsets
            assigned_block_ids: List of block IDs from optimizer (if using optimized blocks)
            block_lookup: Dictionary mapping block_id to block data
        
        Returns:
            Tuple of (ScheduleTrainset, alerts, daily_km)
        """
        ts_data = self.status_map.get(trainset_id, {})
        cumulative_km = ts_data.get('total_mileage_km', 0)
        
        # Use optimized blocks if available, otherwise fall back to index-based generation
        if assigned_block_ids and block_lookup:
            # Use optimizer-assigned blocks
            blocks_data = []
            for block_id in assigned_block_ids:
                if block_id in block_lookup:
                    blocks_data.append(block_lookup[block_id])
            # Sort by departure time
            blocks_data.sort(key=lambda b: b['departure_time'])
        else:
            # Fall back to legacy index-based block generation
            blocks_data = self.service_block_generator.generate_service_blocks(index, num_service)
        
        service_blocks = []
        for b in blocks_data:
            block_id = b['block_id']
            
            # Generate detailed trips with station stops
            trips = self._generate_trips_for_block(b, block_id)
            
            service_block = ServiceBlock(
                block_id=block_id,
                departure_time=b['departure_time'],
                origin=b['origin'],
                destination=b['destination'],
                trip_count=b['trip_count'],
                estimated_km=b['estimated_km'],
                journey_time_minutes=b.get('journey_time_minutes'),
                period=b.get('period'),
                is_peak=b.get('is_peak', False),
                trips=trips
            )
            service_blocks.append(service_block)
        
        daily_km = sum(b.estimated_km for b in service_blocks)
        
        # Calculate readiness score
        readiness = self._calculate_readiness(trainset_id)
        
        # Generate alerts
        alerts = self._check_alerts(trainset_id)
        alert_messages = [a.message for a in alerts]
        
        trainset = ScheduleTrainset(
            trainset_id=trainset_id,
            status=TrainStatus.REVENUE_SERVICE,
            readiness_score=readiness,
            daily_km_allocation=daily_km,
            cumulative_km=cumulative_km,
            assigned_duty=f"DUTY-{chr(65 + index // 10)}{(index % 10) + 1}",
            priority_rank=index + 1,
            service_blocks=service_blocks,
            stabling_bay=random.choice(DEPOT_BAYS),
            alerts=alert_messages
        )
        
        return trainset, alerts, daily_km
    
    def _generate_standby_trainset(self, trainset_id: str) -> tuple:
        """Generate schedule for a standby trainset.
        
        Returns:
            Tuple of (ScheduleTrainset, alerts)
        """
        ts_data = self.status_map.get(trainset_id, {})
        cumulative_km = ts_data.get('total_mileage_km', 0)
        readiness = self._calculate_readiness(trainset_id)
        
        alerts = self._check_alerts(trainset_id)
        alert_messages = [a.message for a in alerts]
        
        trainset = ScheduleTrainset(
            trainset_id=trainset_id,
            status=TrainStatus.STANDBY,
            readiness_score=readiness,
            daily_km_allocation=0,
            cumulative_km=cumulative_km,
            stabling_bay=random.choice(DEPOT_BAYS),
            standby_reason=random.choice(STANDBY_REASONS),
            alerts=alert_messages
        )
        
        return trainset, alerts
    
    def _generate_maintenance_trainset(self, trainset_id: str) -> tuple:
        """Generate schedule for a maintenance trainset.
        
        Returns:
            Tuple of (ScheduleTrainset, alerts)
        """
        ts_data = self.status_map.get(trainset_id, {})
        cumulative_km = ts_data.get('total_mileage_km', 0)
        readiness = self._calculate_readiness(trainset_id)
        
        # Estimate completion time (4-12 hours from now)
        completion = datetime.now() + timedelta(hours=random.randint(4, 12))
        
        alerts = self._check_alerts(trainset_id)
        alert_messages = [a.message for a in alerts]
        
        trainset = ScheduleTrainset(
            trainset_id=trainset_id,
            status=TrainStatus.MAINTENANCE,
            readiness_score=readiness,
            daily_km_allocation=0,
            cumulative_km=cumulative_km,
            maintenance_type=MaintenanceType.SCHEDULED_INSPECTION,
            ibl_bay=random.choice(IBL_BAYS),
            estimated_completion=completion.isoformat(),
            alerts=alert_messages
        )
        
        return trainset, alerts
    
    def _calculate_readiness(self, trainset_id: str) -> float:
        """Calculate readiness score for a trainset."""
        score = 1.0
        
        # Check certificates
        certs = self.cert_map.get(trainset_id, [])
        for cert in certs:
            status = normalize_certificate_status(cert.get('status', 'Valid'))
            if status == 'Expired':
                score -= 0.3
            elif status == 'Expiring-Soon':
                score -= 0.1
            elif status == 'Suspended':
                score -= 0.2
        
        # Check component health
        components = self.health_map.get(trainset_id, [])
        for comp in components:
            status = normalize_component_status(comp.get('status', 'Good'))
            if status == 'Critical':
                score -= 0.15
            elif status == 'Warning':
                score -= 0.05
        
        return max(0.0, min(1.0, round(score, 2)))
    
    def _check_alerts(self, trainset_id: str) -> List[ScheduleAlert]:
        """Check for alerts on a trainset."""
        alerts = []
        
        # Check certificates
        certs = self.cert_map.get(trainset_id, [])
        for cert in certs:
            status = normalize_certificate_status(cert.get('status', 'Valid'))
            dept = cert.get('department', 'Unknown')
            
            if status == 'Expired':
                alerts.append(ScheduleAlert(
                    trainset_id=trainset_id,
                    severity=AlertSeverity.HIGH,
                    alert_type="CERTIFICATE_EXPIRED",
                    message=f"{dept} certificate expired"
                ))
            elif status == 'Expiring-Soon':
                alerts.append(ScheduleAlert(
                    trainset_id=trainset_id,
                    severity=AlertSeverity.MEDIUM,
                    alert_type="CERTIFICATE_EXPIRING",
                    message=f"{dept} certificate expiring soon"
                ))
        
        # Check component health
        components = self.health_map.get(trainset_id, [])
        for comp in components:
            status = normalize_component_status(comp.get('status', 'Good'))
            comp_name = comp.get('component', 'Unknown')
            wear = comp.get('wear_level', 0)
            
            if status == 'Critical':
                alerts.append(ScheduleAlert(
                    trainset_id=trainset_id,
                    severity=AlertSeverity.HIGH,
                    alert_type="COMPONENT_CRITICAL",
                    message=f"{comp_name} in critical condition (wear: {wear}%)"
                ))
            elif status == 'Warning' and wear > 80:
                alerts.append(ScheduleAlert(
                    trainset_id=trainset_id,
                    severity=AlertSeverity.MEDIUM,
                    alert_type="COMPONENT_WARNING",
                    message=f"{comp_name} requires attention (wear: {wear}%)"
                ))
        
        return alerts


def generate_schedule_from_result(
    data: Dict,
    optimization_result: OptimizationResult,
    method: str = "ga",
    runtime_ms: int = 0,
    config: Optional[OptimizationConfig] = None,
    date: Optional[str] = None,
    depot: str = "Muttom_Depot"
) -> ScheduleResult:
    """Convenience function to generate schedule from optimization result.
    
    Args:
        data: Input data dictionary
        optimization_result: Result from optimizer
        method: Optimization method used
        runtime_ms: Runtime in milliseconds
        config: Optimization configuration
        date: Schedule date
        depot: Depot name
        
    Returns:
        Complete ScheduleResult
    """
    generator = ScheduleGenerator(data, config)
    return generator.generate_schedule(
        optimization_result,
        method=method,
        runtime_ms=runtime_ms,
        date=date,
        depot=depot
    )
