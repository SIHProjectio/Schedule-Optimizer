"""
Metro Train Schedule Optimizer
Generates optimal daily schedules from 5:00 AM to 11:00 PM
Considers train health, maintenance, branding, and mileage balancing
"""
import random
from datetime import datetime, time, timedelta
from typing import List, Dict, Tuple, Optional

from ..core.models import (
    DaySchedule, Trainset, TrainStatus, ServiceBlock, FleetSummary,
    OptimizationMetrics, Alert, Severity, DecisionRationale,
    TrainHealthStatus, Route, OperationalHours, FitnessCertificates,
    JobCards, Branding, CertificateStatus, MaintenanceType
)
from ..generators.metro_generator import MetroDataGenerator
from ..core.config_loader import get_config_loader


class MetroScheduleOptimizer:
    """Optimize daily metro train schedules"""
    
    def __init__(
        self,
        date: str,
        num_trains: int,
        route: Route,
        train_health: List[TrainHealthStatus],
        depot_name: str = "Muttom_Depot",
        include_job_cards: bool = False,
        config_dir: Optional[str] = None
    ):
        self.date = date
        self.num_trains = num_trains
        self.route = route
        self.train_health = {t.trainset_id: t for t in train_health}
        self.depot_name = depot_name
        self.generator = MetroDataGenerator(num_trains, config_dir=config_dir)
        self.include_job_cards = include_job_cards
        self.config = get_config_loader(config_dir)
        
        # Operating parameters from config
        operational_config = self.config.get_operational_hours()
        self.op_hours = OperationalHours.from_config(operational_config)
        self.one_way_time_minutes = int(
            (route.total_distance_km / route.avg_speed_kmh) * 60
        )
        self.round_trip_time_minutes = (
            self.one_way_time_minutes * 2 + route.turnaround_time_minutes * 2
        )
        
        # Pre-generate train data
        self.train_data = self._initialize_train_data()
    
    def _initialize_train_data(self) -> Dict[str, Dict]:
        """Initialize all train-specific data"""
        data = {}
        mileages = self.generator.get_realistic_mileage_distribution(self.num_trains)
        
        for i, train_id in enumerate(self.generator.trainset_ids):
            health = self.train_health[train_id]
            fitness_certs = self.generator.generate_fitness_certificates(train_id)
            job_cards = self.generator.generate_job_cards(train_id) if self.include_job_cards else JobCards(open=0, blocking=[])
            branding = self.generator.generate_branding()
            
            readiness = self.generator.calculate_readiness_score(
                fitness_certs, job_cards, health.component_health
            )
            
            data[train_id] = {
                "health": health,
                "fitness_certs": fitness_certs,
                "job_cards": job_cards,
                "branding": branding,
                "readiness_score": readiness,
                "cumulative_km": mileages[i],
                "stabling_bay": random.choice(self.generator.DEPOT_BAYS)
            }
        
        return data
    
    def _calculate_service_hours(self) -> int:
        """Calculate total service hours in a day"""
        start = datetime.combine(datetime.today(), self.op_hours.start_time)
        end = datetime.combine(datetime.today(), self.op_hours.end_time)
        return int((end - start).total_seconds() / 3600)
    
    def _is_train_available(
        self,
        train_id: str,
        start_hour: int,
        end_hour: int
    ) -> bool:
        """Check if train is available for given time window"""
        health = self.train_data[train_id]["health"]
        
        if health.is_fully_healthy:
            return True
        
        if not health.available_hours:
            return False
        
        # Check if requested window overlaps with available hours
        for avail_start, avail_end in health.available_hours:
            req_start = time(start_hour, 0)
            req_end = time(end_hour, 0)
            
            if req_start >= avail_start and req_end <= avail_end:
                return True
        
        return False
    
    def _rank_trains_for_service(self) -> List[Tuple[str, float]]:
        """Rank trains by suitability for revenue service"""
        rankings = []
        
        for train_id, data in self.train_data.items():
            score = 0.0
            
            # Base readiness score (40% weight)
            score += data["readiness_score"] * 0.4
            
            # Certificate validity (20% weight)
            certs = data["fitness_certs"]
            if certs.rolling_stock.status == CertificateStatus.VALID:
                score += 0.15
            if certs.signalling.status == CertificateStatus.VALID:
                score += 0.05
                
            # No blocking job cards (15% weight)
            if len(data["job_cards"].blocking) == 0:
                score += 0.15
            
            # Branding priority (15% weight)
            branding = data["branding"]
            if branding.exposure_priority == "CRITICAL":
                score += 0.15
            elif branding.exposure_priority == "HIGH":
                score += 0.10
            elif branding.exposure_priority == "MEDIUM":
                score += 0.05
            
            # Mileage balancing (10% weight) - prefer lower mileage
            max_mileage = 200000
            mileage_factor = 1.0 - (data["cumulative_km"] / max_mileage)
            score += mileage_factor * 0.10
            
            rankings.append((train_id, score))
        
        return sorted(rankings, key=lambda x: x[1], reverse=True)
    
    def _generate_service_blocks(
        self,
        train_id: str,
        duty_name: str,
        num_blocks: int = 2
    ) -> Tuple[List[ServiceBlock], int]:
        """Generate service blocks for a train"""
        blocks = []
        total_km = 0
        
        # Distribute service across the day
        service_hours = self._calculate_service_hours()
        block_duration_hours = service_hours // num_blocks
        
        current_hour = self.op_hours.start_time.hour
        
        for i in range(num_blocks):
            block_start_hour = current_hour + (i * block_duration_hours)
            if block_start_hour >= self.op_hours.end_time.hour:
                break
            
            # Calculate trips for this block
            block_minutes = block_duration_hours * 60
            trips = max(1, block_minutes // self.round_trip_time_minutes)
            
            # Alternate origin/destination
            if i % 2 == 0:
                origin = self.route.stations[0].name
                destination = self.route.stations[-1].name
            else:
                origin = self.route.stations[-1].name
                destination = self.route.stations[0].name
            
            block_km = int(trips * self.route.total_distance_km * 2)  # Round trips
            total_km += block_km
            
            block = ServiceBlock(
                block_id=f"BLK-{random.randint(1, 999):03d}",
                departure_time=f"{block_start_hour:02d}:{random.randint(0, 45):02d}",
                origin=origin,
                destination=destination,
                trip_count=trips,
                estimated_km=block_km
            )
            blocks.append(block)
        
        return blocks, total_km
    
    def _assign_train_status(
        self,
        train_id: str,
        rank: int,
        required_service: int,
        min_standby: int
    ) -> Tuple[TrainStatus, Optional[str], List[ServiceBlock], int]:
        """Assign status and duty to a train"""
        data = self.train_data[train_id]
        health = data["health"]
        
        # Check if train is unavailable
        if not health.is_fully_healthy and not health.available_hours:
            # Determine maintenance or out of service
            if data["job_cards"].open > 0 or len(data["job_cards"].blocking) > 0:
                return TrainStatus.MAINTENANCE, None, [], 0
            else:
                return TrainStatus.MAINTENANCE, None, [], 0
        
        # Check for blocking maintenance
        if len(data["job_cards"].blocking) > 0:
            return TrainStatus.MAINTENANCE, None, [], 0
        
        # Check for expired certificates
        certs = data["fitness_certs"]
        if certs.rolling_stock.status == CertificateStatus.EXPIRED:
            return TrainStatus.MAINTENANCE, None, [], 0
        
        # Assign to revenue service
        if rank <= required_service:
            # Check availability for full day
            if self._is_train_available(
                train_id,
                self.op_hours.start_time.hour,
                self.op_hours.end_time.hour
            ):
                duty = f"DUTY-{chr(65 + (rank-1) // 10)}{(rank-1) % 10 + 1}"
                blocks, km = self._generate_service_blocks(train_id, duty)
                return TrainStatus.REVENUE_SERVICE, duty, blocks, km
        
        # Assign to standby
        if rank <= required_service + min_standby:
            return TrainStatus.STANDBY, None, [], 0
        
        # Random assignment of remaining trains
        roll = random.random()
        if roll < 0.05:
            return TrainStatus.CLEANING, None, [], 0
        elif roll < 0.15:
            return TrainStatus.STANDBY, None, [], 0
        else:
            return TrainStatus.MAINTENANCE, None, [], 0
    
    def optimize_schedule(
        self,
        min_service_trains: int = 20,
        min_standby: int = 2,
        max_daily_km: int = 300
    ) -> DaySchedule:
        """Generate optimized daily schedule"""
        start_time = datetime.now()
        
        # Rank trains
        rankings = self._rank_trains_for_service()
        
        # Build trainset list
        trainsets = []
        status_counts = {
            TrainStatus.REVENUE_SERVICE: 0,
            TrainStatus.STANDBY: 0,
            TrainStatus.MAINTENANCE: 0,
            TrainStatus.CLEANING: 0
        }
        total_km = 0
        readiness_scores = []
        
        for rank, (train_id, score) in enumerate(rankings, 1):
            data = self.train_data[train_id]
            
            # Assign status and blocks
            status, duty, blocks, daily_km = self._assign_train_status(
                train_id, rank, min_service_trains, min_standby
            )
            
            status_counts[status] += 1
            total_km += daily_km
            readiness_scores.append(data["readiness_score"])
            
            # Build trainset object
            trainset = Trainset(
                trainset_id=train_id,
                status=status,
                priority_rank=rank if status == TrainStatus.REVENUE_SERVICE else None,
                assigned_duty=duty,
                service_blocks=blocks,
                daily_km_allocation=daily_km,
                cumulative_km=data["cumulative_km"],
                stabling_bay=data["stabling_bay"] if status != TrainStatus.MAINTENANCE else None,
                fitness_certificates=data["fitness_certs"],
                job_cards=data["job_cards"],
                branding=data["branding"],
                readiness_score=data["readiness_score"],
                constraints_met=data["readiness_score"] >= 0.7
            )
            
            # Add status-specific fields
            if status == TrainStatus.MAINTENANCE:
                trainset.maintenance_type = MaintenanceType.SCHEDULED_INSPECTION
                trainset.ibl_bay = random.choice(self.generator.IBL_BAYS)
                completion_time = datetime.now() + timedelta(hours=random.randint(4, 12))
                trainset.estimated_completion = completion_time.isoformat()
            elif status == TrainStatus.CLEANING:
                trainset.cleaning_bay = random.choice(self.generator.WASH_BAYS)
                trainset.cleaning_type = random.choice(["DEEP_INTERIOR", "EXTERIOR", "FULL"])
                completion_time = datetime.now() + timedelta(hours=random.randint(2, 4))
                trainset.estimated_completion = completion_time.isoformat()
                trainset.scheduled_service_start = f"{random.randint(12, 18):02d}:30"
            elif status == TrainStatus.STANDBY:
                trainset.standby_reason = random.choice([
                    "MILEAGE_BALANCING", "EMERGENCY_BACKUP", "PEAK_HOUR_RESERVE"
                ])
            
            # Generate alerts
            alerts = []
            if data["fitness_certs"].telecom.status == CertificateStatus.EXPIRING_SOON:
                alerts.append("TELECOM_CERT_EXPIRES_SOON")
            if len(data["job_cards"].blocking) > 0:
                alerts.append(f"{len(data['job_cards'].blocking)}_BLOCKING_JOB_CARDS")
            trainset.alerts = alerts
            
            trainsets.append(trainset)
        
        # Build fleet summary
        fleet_summary = FleetSummary(
            total_trainsets=self.num_trains,
            revenue_service=status_counts[TrainStatus.REVENUE_SERVICE],
            standby=status_counts[TrainStatus.STANDBY],
            maintenance=status_counts[TrainStatus.MAINTENANCE],
            cleaning=status_counts[TrainStatus.CLEANING],
            availability_percent=round(
                (status_counts[TrainStatus.REVENUE_SERVICE] + status_counts[TrainStatus.STANDBY])
                / self.num_trains * 100, 1
            )
        )
        
        # Calculate optimization metrics
        mileages = [data["cumulative_km"] for data in self.train_data.values()]
        variance = (max(mileages) - min(mileages)) / (sum(mileages) / len(mileages))
        
        optimization_metrics = OptimizationMetrics(
            mileage_variance_coefficient=round(variance, 3),
            avg_readiness_score=round(sum(readiness_scores) / len(readiness_scores), 2),
            branding_sla_compliance=1.0,  # Placeholder
            shunting_movements_required=random.randint(5, 15),
            total_planned_km=total_km,
            fitness_expiry_violations=0
        )
        
        # Generate alerts
        conflicts = []
        for trainset in trainsets:
            data = self.train_data[trainset.trainset_id]
            
            if data["fitness_certs"].telecom.status == CertificateStatus.EXPIRING_SOON:
                conflicts.append(Alert(
                    trainset_id=trainset.trainset_id,
                    severity=Severity.MEDIUM,
                    type="CERTIFICATE_EXPIRING",
                    message="Telecom certificate expires soon"
                ))
            
            if len(data["job_cards"].blocking) > 0:
                conflicts.append(Alert(
                    trainset_id=trainset.trainset_id,
                    severity=Severity.HIGH,
                    type="BLOCKING_MAINTENANCE",
                    message=f"{len(data['job_cards'].blocking)} open job cards preventing service"
                ))
        
        # Decision rationale
        end_time = datetime.now()
        runtime_ms = int((end_time - start_time).total_seconds() * 1000)
        
        rationale = DecisionRationale(
            algorithm_version="v2.5.0",
            objective_weights={
                "service_readiness": 0.35,
                "mileage_balancing": 0.25,
                "branding_priority": 0.20,
                "operational_cost": 0.20
            },
            constraint_violations=0,
            optimization_runtime_ms=runtime_ms
        )
        
        # Build complete schedule
        schedule_id = f"KMRL-{self.date}-{random.choice(['DAWN', 'ALPHA', 'PRIME'])}"
        now = datetime.now()
        
        schedule = DaySchedule(
            schedule_id=schedule_id,
            generated_at=now.isoformat(),
            valid_from=f"{self.date}T{self.op_hours.start_time.isoformat()}+05:30",
            valid_until=f"{self.date}T{self.op_hours.end_time.isoformat()}+05:30",
            depot=self.depot_name,
            trainsets=trainsets,
            fleet_summary=fleet_summary,
            optimization_metrics=optimization_metrics,
            conflicts_and_alerts=conflicts,
            decision_rationale=rationale
        )
        
        return schedule
