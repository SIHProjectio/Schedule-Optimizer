"""
Enhanced Metro Synthetic Data Generator
Generates realistic metro train scheduling data with time-based constraints
"""
import random
import uuid
from datetime import datetime, timedelta, time
from typing import List, Dict, Tuple
from .metro_models import (
    TrainHealthStatus, Station, Route, FitnessCertificates,
    FitnessCertificate, CertificateStatus, JobCards, Branding
)


class MetroDataGenerator:
    """Generate synthetic data for metro train scheduling"""
    
    STATIONS_ALUVA_PETTAH = [
        "Aluva", "Pulinchodu", "Companypadi", "Ambattukavu", "Muttom",
        "Kalamassery", "Cochin University", "Pathadipalam", "Edapally",
        "Changampuzha Park", "Palarivattom", "J.L.N Stadium", "Kaloor",
        "Town Hall", "M.G. Road", "Maharaja's College", "Ernakulam South",
        "Kadavanthra", "Elamkulam", "Vyttila", "Thaikoodam", "Petta",
        "Vadakkekotta", "SN Junction", "Pettah"
    ]
    
    DEPOT_BAYS = [f"BAY-{str(i).zfill(2)}" for i in range(1, 16)]
    IBL_BAYS = [f"IBL-{str(i).zfill(2)}" for i in range(1, 6)]
    WASH_BAYS = [f"WASH-BAY-{str(i).zfill(2)}" for i in range(1, 4)]
    
    ADVERTISERS = [
        "COCACOLA-2024", "FLIPKART-FESTIVE", "AMAZON-PRIME",
        "RELIANCE-JIO", "TATA-MOTORS", "SAMSUNG-GALAXY",
        "NONE"
    ]
    
    UNAVAILABLE_REASONS = [
        "SCHEDULED_MAINTENANCE", "BRAKE_SYSTEM_REPAIR",
        "HVAC_REPLACEMENT", "BOGIE_OVERHAUL", "ELECTRICAL_FAULT",
        "ACCIDENT_DAMAGE", "PANTOGRAPH_REPAIR", "DOOR_SYSTEM_FAULT"
    ]
    
    def __init__(self, num_trains: int = 25, num_stations: int = 25):
        self.num_trains = num_trains
        self.num_stations = min(num_stations, len(self.STATIONS_ALUVA_PETTAH))
        self.trainset_ids = [f"TS-{str(i+1).zfill(3)}" for i in range(num_trains)]
        
    def generate_route(self, route_name: str = "Aluva-Pettah Line") -> Route:
        """Generate metro route with stations"""
        stations = []
        total_distance = 25.612  # Actual KMRL distance
        
        for i in range(self.num_stations):
            distance = (total_distance / (self.num_stations - 1)) * i
            station = Station(
                station_id=f"STN-{str(i+1).zfill(3)}",
                name=self.STATIONS_ALUVA_PETTAH[i],
                sequence=i + 1,
                distance_from_origin_km=round(distance, 2),
                avg_dwell_time_seconds=random.randint(20, 45)
            )
            stations.append(station)
        
        return Route(
            route_id="KMRL-LINE-01",
            name=route_name,
            stations=stations,
            total_distance_km=total_distance,
            avg_speed_kmh=random.randint(32, 38),
            turnaround_time_minutes=random.randint(8, 12)
        )
    
    def generate_train_health_statuses(self) -> List[TrainHealthStatus]:
        """Generate health status for all trains"""
        statuses = []
        
        for i, ts_id in enumerate(self.trainset_ids):
            # Determine train health category
            health_roll = random.random()
            
            if health_roll < 0.65:  # 65% fully healthy
                is_healthy = True
                available_hours = None
                reason = None
            elif health_roll < 0.85:  # 20% partially healthy
                is_healthy = False
                # Random availability window
                start_hour = random.randint(5, 12)
                end_hour = random.randint(start_hour + 4, 23)
                available_hours = [(time(start_hour, 0), time(end_hour, 0))]
                reason = f"Limited availability: {random.choice(['Minor repairs', 'Partial maintenance', 'Certificate renewal pending'])}"
            else:  # 15% unavailable
                is_healthy = False
                available_hours = []
                reason = random.choice(self.UNAVAILABLE_REASONS)
            
            status = TrainHealthStatus(
                trainset_id=ts_id,
                is_fully_healthy=is_healthy,
                available_hours=available_hours,
                unavailable_reason=reason,
                cumulative_mileage=random.randint(50000, 200000),
                days_since_maintenance=random.randint(1, 45),
                component_health={
                    "brakes": random.uniform(0.7, 1.0),
                    "hvac": random.uniform(0.65, 1.0),
                    "doors": random.uniform(0.7, 1.0),
                    "bogies": random.uniform(0.75, 1.0),
                    "pantograph": random.uniform(0.7, 1.0),
                    "battery": random.uniform(0.65, 1.0),
                    "motor": random.uniform(0.75, 1.0)
                }
            )
            statuses.append(status)
        
        return statuses
    
    def generate_fitness_certificates(self, train_id: str) -> FitnessCertificates:
        """Generate fitness certificates for a train"""
        now = datetime.now()
        
        def random_cert_status() -> Tuple[str, CertificateStatus]:
            roll = random.random()
            if roll < 0.75:  # 75% valid
                days_valid = random.randint(10, 60)
                return (now + timedelta(days=days_valid)).isoformat(), CertificateStatus.VALID
            elif roll < 0.90:  # 15% expiring soon
                days_valid = random.randint(1, 9)
                return (now + timedelta(days=days_valid)).isoformat(), CertificateStatus.EXPIRING_SOON
            else:  # 10% expired
                days_expired = random.randint(1, 5)
                return (now - timedelta(days=days_expired)).isoformat(), CertificateStatus.EXPIRED
        
        rs_date, rs_status = random_cert_status()
        sig_date, sig_status = random_cert_status()
        tel_date, tel_status = random_cert_status()
        
        return FitnessCertificates(
            rolling_stock=FitnessCertificate(valid_until=rs_date, status=rs_status),
            signalling=FitnessCertificate(valid_until=sig_date, status=sig_status),
            telecom=FitnessCertificate(valid_until=tel_date, status=tel_status)
        )
    
    def generate_job_cards(self, train_id: str) -> JobCards:
        """Generate job cards for a train"""
        num_open = random.choices([0, 1, 2, 3, 4, 5], weights=[50, 25, 15, 7, 2, 1])[0]
        
        blocking = []
        if num_open > 0:
            num_blocking = random.choices([0, 1, 2, 3], weights=[70, 20, 8, 2])[0]
            if num_blocking > 0:
                components = ["BRAKE", "HVAC", "DOOR", "BOGIE", "PANTOGRAPH", "ELECTRICAL"]
                selected = random.sample(components, min(num_blocking, len(components)))
                blocking = [f"JC-{random.randint(40000, 49999)}-{comp}" for comp in selected]
        
        return JobCards(open=num_open, blocking=blocking)
    
    def generate_branding(self) -> Branding:
        """Generate branding information"""
        advertiser = random.choice(self.ADVERTISERS)
        
        if advertiser == "NONE":
            return Branding(
                advertiser="NONE",
                contract_hours_remaining=0,
                exposure_priority="NONE"
            )
        
        return Branding(
            advertiser=advertiser,
            contract_hours_remaining=random.randint(50, 500),
            exposure_priority=random.choice(["LOW", "MEDIUM", "HIGH", "CRITICAL"])
        )
    
    def calculate_readiness_score(
        self,
        fitness_certs: FitnessCertificates,
        job_cards: JobCards,
        component_health: Dict[str, float]
    ) -> float:
        """Calculate overall readiness score for a train"""
        score = 1.0
        
        # Certificate penalties
        if fitness_certs.rolling_stock.status == CertificateStatus.EXPIRED:
            score -= 0.4
        elif fitness_certs.rolling_stock.status == CertificateStatus.EXPIRING_SOON:
            score -= 0.1
            
        if fitness_certs.signalling.status == CertificateStatus.EXPIRED:
            score -= 0.3
        elif fitness_certs.signalling.status == CertificateStatus.EXPIRING_SOON:
            score -= 0.05
            
        if fitness_certs.telecom.status == CertificateStatus.EXPIRED:
            score -= 0.2
        elif fitness_certs.telecom.status == CertificateStatus.EXPIRING_SOON:
            score -= 0.05
        
        # Job card penalties
        if job_cards.open > 0:
            score -= min(0.15, job_cards.open * 0.03)
        if len(job_cards.blocking) > 0:
            score -= min(0.25, len(job_cards.blocking) * 0.1)
        
        # Component health impact
        avg_health = sum(component_health.values()) / len(component_health)
        health_factor = (avg_health - 0.5) * 0.2  # -0.1 to +0.1
        score += health_factor
        
        return max(0.0, min(1.0, score))
    
    def generate_depot_layout(self) -> Dict[str, List[str]]:
        """Generate depot bay layout"""
        return {
            "stabling_bays": self.DEPOT_BAYS.copy(),
            "ibl_bays": self.IBL_BAYS.copy(),
            "wash_bays": self.WASH_BAYS.copy()
        }
    
    def get_realistic_mileage_distribution(self, num_trains: int) -> List[int]:
        """Generate realistic cumulative mileage distribution"""
        # Create a distribution with some variance
        base_mileage = 120000
        mileages = []
        
        for i in range(num_trains):
            # Add variance based on age and usage patterns
            variance = random.randint(-40000, 50000)
            mileage = base_mileage + variance
            mileages.append(max(50000, min(200000, mileage)))
        
        return mileages
