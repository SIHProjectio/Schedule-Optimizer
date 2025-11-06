"""
Enhanced synthetic data generator for metro trainset scheduling.
Provides more realistic and optimization-friendly data generation.
"""
import json
import random
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import uuid
from enum import Enum


class TrainsetStatus(Enum):
    """Trainset operational status"""
    AVAILABLE = "Available"
    IN_SERVICE = "In-Service"
    MAINTENANCE = "Maintenance"
    STANDBY = "Standby"
    OUT_OF_ORDER = "Out-of-Order"


class CertificateStatus(Enum):
    """Certificate validity status"""
    VALID = "Valid"
    EXPIRED = "Expired"
    EXPIRING_SOON = "Expiring-Soon"
    SUSPENDED = "Suspended"


class Priority(Enum):
    """Priority levels"""
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class EnhancedMetroDataGenerator:
    """Enhanced synthetic data generator with realistic constraints and dependencies."""
    
    def __init__(self, num_trainsets: int = 25, seed: Optional[int] = None):
        if seed:
            random.seed(seed)
            np.random.seed(seed)
            
        self.num_trainsets = num_trainsets
        self.trainset_ids = [f"TS-{str(i+1).zfill(3)}" for i in range(num_trainsets)]
        self.departments = ["Rolling Stock", "Signalling", "Telecom", "Safety", "HVAC"]
        self.brands = ["Brand-A", "Brand-B", "Brand-C", "Brand-D", "Brand-E"]
        
        # Realistic component lifespans and thresholds
        self.components = {
            "Bogie": {"wear_threshold": 85, "unit": "% wear", "service_life_km": 800000},
            "Brake_Pad": {"wear_threshold": 70, "unit": "% remaining", "service_life_km": 150000},
            "HVAC": {"wear_threshold": 80, "unit": "% efficiency", "service_life_km": 500000},
            "Door_System": {"wear_threshold": 90, "unit": "cycles", "service_life_km": 600000},
            "Pantograph": {"wear_threshold": 75, "unit": "% condition", "service_life_km": 400000},
            "Battery": {"wear_threshold": 80, "unit": "% capacity", "service_life_km": 300000},
            "Traction_Motor": {"wear_threshold": 85, "unit": "% efficiency", "service_life_km": 1000000},
            "Compressor": {"wear_threshold": 75, "unit": "% performance", "service_life_km": 600000}
        }
        
        # Generate base trainset characteristics
        self._generate_trainset_profiles()
    
    def _generate_trainset_profiles(self):
        """Generate realistic profiles for each trainset."""
        self.trainset_profiles = {}
        
        for ts_id in self.trainset_ids:
            # Age and mileage correlation
            age_years = random.uniform(1, 15)
            annual_mileage = random.randint(80000, 120000)
            total_mileage = int(age_years * annual_mileage + random.randint(-20000, 20000))
            
            # Reliability decreases with age and high mileage
            base_reliability = max(0.7, 0.98 - (age_years * 0.015) - (total_mileage / 2000000))
            
            profile = {
                "age_years": age_years,
                "total_mileage_km": total_mileage,
                "base_reliability": base_reliability,
                "manufacturer": random.choice(["Manufacturer-A", "Manufacturer-B", "Manufacturer-C"]),
                "last_major_overhaul": datetime.now() - timedelta(days=random.randint(180, 1800)),
                "preferred_routes": random.sample(["Route-1", "Route-2", "Route-3", "Route-4"], 
                                                random.randint(1, 3))
            }
            self.trainset_profiles[ts_id] = profile
    
    def generate_enhanced_trainset_status(self) -> List[Dict]:
        """Generate realistic trainset status with correlations."""
        statuses = []
        
        # Ensure we have minimum required trainsets available
        available_count = 0
        target_available = max(22, int(self.num_trainsets * 0.85))  # 85% availability target
        
        for i, ts_id in enumerate(self.trainset_ids):
            profile = self.trainset_profiles[ts_id]
            
            # Determine status based on profile
            if available_count < target_available and i < len(self.trainset_ids) - 3:
                # Force some trainsets to be available
                if random.random() < 0.9:
                    operational_status = TrainsetStatus.AVAILABLE.value
                    current_location = random.choice(["Depot-A", "Depot-B", "Standby-Bay"])
                    available_count += 1
                else:
                    operational_status = random.choice([
                        TrainsetStatus.MAINTENANCE.value,
                        TrainsetStatus.STANDBY.value
                    ])
                    current_location = "IBL" if operational_status == TrainsetStatus.MAINTENANCE.value else "Depot-A"
            else:
                # Natural distribution for remaining trainsets
                weights = [0.7, 0.1, 0.15, 0.04, 0.01]  # Available, In-Service, Maintenance, Standby, OOO
                operational_status = random.choices(
                    [s.value for s in TrainsetStatus], 
                    weights=weights
                )[0]
                
                if operational_status == TrainsetStatus.AVAILABLE.value:
                    available_count += 1
                    current_location = random.choice(["Depot-A", "Depot-B", "Standby-Bay"])
                elif operational_status == TrainsetStatus.IN_SERVICE.value:
                    current_location = "In-Service"
                elif operational_status == TrainsetStatus.MAINTENANCE.value:
                    current_location = "IBL"
                else:
                    current_location = "Depot-A"
            
            # Calculate service intervals based on mileage
            days_since_service = min(
                random.randint(1, 45),
                int((profile["total_mileage_km"] % 10000) / 200)  # More mileage = more recent service needed
            )
            
            status = {
                "trainset_id": ts_id,
                "current_location": current_location,
                "operational_status": operational_status,
                "last_service_date": (datetime.now() - timedelta(days=days_since_service)).isoformat(),
                "total_mileage_km": profile["total_mileage_km"],
                "daily_mileage_km": random.randint(180, 420),
                "operational_hours": int(profile["total_mileage_km"] / 35),  # Kochi Metro avg operating speed: 35 km/h
                "age_years": round(profile["age_years"], 1),
                "base_reliability_score": round(profile["base_reliability"], 3),
                "manufacturer": profile["manufacturer"],
                "last_updated": datetime.now().isoformat(),
                "energy_efficiency_rating": random.choice(["A", "A", "B", "B", "C"]),  # Most are efficient
                "capacity_passengers": random.choice([320, 360, 400])  # Standard capacities
            }
            statuses.append(status)
        
        return statuses
    
    def generate_realistic_fitness_certificates(self) -> List[Dict]:
        """Generate fitness certificates with realistic expiry patterns."""
        certificates = []
        
        for ts_id in self.trainset_ids:
            profile = self.trainset_profiles[ts_id]
            
            for dept in self.departments:
                # Certificate validity periods vary by department
                validity_periods = {
                    "Rolling Stock": 365,  # 1 year
                    "Signalling": 180,     # 6 months
                    "Telecom": 90,         # 3 months
                    "Safety": 365,         # 1 year
                    "HVAC": 180           # 6 months
                }
                
                validity_days = validity_periods.get(dept, 180)
                
                # Issue date based on maintenance cycles
                issue_days_ago = random.randint(1, validity_days - 10)
                issue_date = datetime.now() - timedelta(days=issue_days_ago)
                expiry_date = issue_date + timedelta(days=validity_days)
                
                # Status determination
                days_to_expiry = (expiry_date - datetime.now()).days
                if days_to_expiry < 0:
                    status = CertificateStatus.EXPIRED.value
                elif days_to_expiry <= 30:
                    status = CertificateStatus.EXPIRING_SOON.value
                elif profile["base_reliability"] < 0.8 and random.random() < 0.1:
                    status = CertificateStatus.SUSPENDED.value
                else:
                    status = CertificateStatus.VALID.value
                
                cert = {
                    "certificate_id": str(uuid.uuid4()),
                    "trainset_id": ts_id,
                    "department": dept,
                    "issue_date": issue_date.isoformat(),
                    "expiry_date": expiry_date.isoformat(),
                    "status": status,
                    "inspector_id": f"INS-{random.randint(100, 999)}",
                    "compliance_score": random.randint(
                        75 if status == CertificateStatus.VALID.value else 60, 
                        100 if status == CertificateStatus.VALID.value else 85
                    ),
                    "validity_period_days": validity_days,
                    "renewal_required": days_to_expiry <= 30,
                    "remarks": self._generate_certificate_remarks(status, dept)
                }
                certificates.append(cert)
        
        return certificates
    
    def _generate_certificate_remarks(self, status: str, department: str) -> str:
        """Generate realistic certificate remarks."""
        if status == CertificateStatus.VALID.value:
            return random.choice([
                "All systems operational",
                "Minor maintenance recommended",
                "Performance within acceptable limits",
                "No issues identified"
            ])
        elif status == CertificateStatus.EXPIRING_SOON.value:
            return f"{department} certification renewal due soon"
        elif status == CertificateStatus.EXPIRED.value:
            return f"{department} certification expired - renewal required"
        else:
            return f"{department} certification suspended - investigation required"
    
    def generate_correlated_job_cards(self) -> List[Dict]:
        """Generate job cards correlated with trainset conditions."""
        job_cards = []
        job_types = ["Preventive", "Corrective", "Breakdown", "Inspection", "Upgrade"]
        
        for ts_id in self.trainset_ids:
            profile = self.trainset_profiles[ts_id]
            
            # More jobs for older/higher mileage trainsets
            job_probability = 0.2 + (profile["age_years"] / 50) + (profile["total_mileage_km"] / 2000000)
            num_jobs = np.random.poisson(job_probability * 3)  # Poisson distribution
            
            for _ in range(num_jobs):
                # Job priority based on trainset condition
                if profile["base_reliability"] < 0.75:
                    priority = random.choice([Priority.CRITICAL.value, Priority.HIGH.value])
                elif profile["base_reliability"] < 0.85:
                    priority = random.choice([Priority.HIGH.value, Priority.MEDIUM.value])
                else:
                    priority = random.choice([Priority.MEDIUM.value, Priority.LOW.value])
                
                # Status based on priority
                if priority == Priority.CRITICAL.value:
                    status = "Open"
                    estimated_hours = random.randint(8, 48)
                else:
                    status = random.choice(["Open", "Open", "In-Progress", "Closed", "Pending-Parts"])
                    estimated_hours = random.randint(2, 24)
                
                job = {
                    "job_card_id": f"JC-{random.randint(10000, 99999)}",
                    "trainset_id": ts_id,
                    "work_order_number": f"WO-{random.randint(100000, 999999)}",
                    "job_type": random.choice(job_types),
                    "priority": priority,
                    "status": status,
                    "created_date": (datetime.now() - timedelta(days=random.randint(1, 30))).isoformat(),
                    "estimated_completion": (datetime.now() + timedelta(hours=estimated_hours)).isoformat(),
                    "assigned_technician": f"TECH-{random.randint(100, 999)}",
                    "component": random.choice(list(self.components.keys())),
                    "description": self._generate_job_description(),
                    "estimated_hours": estimated_hours,
                    "cost_estimate": random.randint(5000, 50000) * (1 if priority == Priority.LOW.value else 2)
                }
                job_cards.append(job)
        
        return job_cards
    
    def _generate_job_description(self) -> str:
        """Generate realistic job descriptions."""
        return random.choice([
            "Routine maintenance required",
            "Component inspection needed",
            "Performance optimization",
            "Safety system check",
            "Preventive maintenance",
            "Wear part replacement",
            "System calibration",
            "Diagnostic testing required"
        ])
    
    def generate_realistic_component_health(self) -> List[Dict]:
        """Generate component health data correlated with mileage and age."""
        health_data = []
        
        for ts_id in self.trainset_ids:
            profile = self.trainset_profiles[ts_id]
            
            for comp_name, comp_info in self.components.items():
                # Calculate wear based on mileage and service life
                wear_ratio = profile["total_mileage_km"] / comp_info["service_life_km"]
                base_wear = min(95, wear_ratio * 100)
                
                # Add random variation
                wear_level = max(0, min(100, base_wear + random.randint(-15, 10)))
                
                # Health score inversely related to wear
                health_score = max(60, 100 - wear_level + random.randint(-5, 5))
                
                # Status based on wear level and threshold
                if wear_level > comp_info["wear_threshold"]:
                    status = "Warning" if wear_level < 90 else "Critical"
                elif wear_level > comp_info["wear_threshold"] * 0.8:
                    status = "Fair"
                else:
                    status = "Good"
                
                # Next maintenance based on wear rate
                km_to_maintenance = max(1000, 
                    int((comp_info["service_life_km"] * (comp_info["wear_threshold"] / 100) - 
                         profile["total_mileage_km"]) * 0.1))
                
                health = {
                    "trainset_id": ts_id,
                    "component": comp_name,
                    "health_score": health_score,
                    "wear_level": round(wear_level, 1),
                    "threshold": comp_info["wear_threshold"],
                    "unit": comp_info["unit"],
                    "status": status,
                    "next_maintenance_km": km_to_maintenance,
                    "service_life_km": comp_info["service_life_km"],
                    "current_mileage_km": profile["total_mileage_km"],
                    "last_maintenance_date": (profile["last_major_overhaul"] + 
                                            timedelta(days=random.randint(0, 180))).isoformat(),
                    "predicted_failure_date": (datetime.now() + 
                                              timedelta(days=random.randint(30, 365))).isoformat(),
                    "maintenance_urgency": "High" if status in ["Warning", "Critical"] else "Normal",
                    "timestamp": datetime.now().isoformat()
                }
                health_data.append(health)
        
        return health_data
    
    def generate_optimized_branding_contracts(self) -> List[Dict]:
        """Generate branding contracts with optimization constraints."""
        contracts = []
        
        # Select trainsets for branding (not all will have contracts)
        branded_trainsets = random.sample(self.trainset_ids, 
                                        random.randint(int(self.num_trainsets * 0.4), 
                                                      int(self.num_trainsets * 0.7)))
        
        for ts_id in branded_trainsets:
            profile = self.trainset_profiles[ts_id]
            brand = random.choice(self.brands)
            
            # Contract value based on trainset reliability and routes
            base_value = random.randint(800000, 1500000)
            reliability_multiplier = profile["base_reliability"]
            route_multiplier = len(profile["preferred_routes"]) * 0.1 + 0.9
            
            contract_value = int(base_value * reliability_multiplier * route_multiplier)
            
            # Exposure requirements
            daily_target = random.randint(8, 14)
            contracted_hours = daily_target * 30  # Monthly
            
            # Current performance (some underperforming for optimization challenge)
            performance_factor = random.uniform(0.7, 1.1)
            actual_hours = int(contracted_hours * performance_factor)
            
            contract = {
                "trainset_id": ts_id,
                "brand": brand,
                "contract_id": f"ADV-{random.randint(1000, 9999)}",
                "start_date": (datetime.now() - timedelta(days=random.randint(30, 180))).isoformat(),
                "end_date": (datetime.now() + timedelta(days=random.randint(90, 365))).isoformat(),
                "contracted_exposure_hours": contracted_hours,
                "actual_exposure_hours": actual_hours,
                "daily_target_hours": daily_target,
                "contract_value": contract_value,
                "penalty_per_hour_shortfall": random.randint(800, 2500),
                "bonus_per_excess_hour": random.randint(400, 1200),
                "performance_ratio": round(actual_hours / contracted_hours, 3),
                "status": "Compliant" if actual_hours >= contracted_hours * 0.95 else "At-Risk",
                "priority_level": random.choice(["High", "Medium", "Low"]),
                "route_restrictions": profile["preferred_routes"],
                "minimum_daily_hours": max(4, daily_target - 2),
                "maximum_daily_hours": daily_target + 4
            }
            contracts.append(contract)
        
        return contracts
    
    def generate_complete_enhanced_dataset(self) -> Dict:
        """Generate complete enhanced dataset with all improvements."""
        print("Generating enhanced synthetic data...")
        
        dataset = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "num_trainsets": self.num_trainsets,
                "system": "Kochi Metro Rail - Enhanced",
                "data_version": "2.0",
                "generator": "EnhancedMetroDataGenerator",
                "features": [
                    "Age-correlated reliability",
                    "Mileage-based component wear",
                    "Realistic certificate expiry",
                    "Correlated job priorities",
                    "Optimized branding constraints"
                ]
            },
            "trainset_profiles": self.trainset_profiles,
            "trainset_status": self.generate_enhanced_trainset_status(),
            "fitness_certificates": self.generate_realistic_fitness_certificates(),
            "job_cards": self.generate_correlated_job_cards(),
            "component_health": self.generate_realistic_component_health(),
            "branding_contracts": self.generate_optimized_branding_contracts(),
            # Keep the existing generators for other data
            "iot_sensors": self._generate_iot_sensors(),
            "maintenance_schedule": self._generate_maintenance_schedule(),
            "performance_metrics": self._generate_performance_metrics(),
            "cleaning_slots": self._generate_cleaning_slots(),
            "manual_overrides": self._generate_manual_overrides(),
            "external_factors": self._generate_external_factors()
        }
        
        return dataset
    
    def _generate_iot_sensors(self) -> List[Dict]:
        """Generate IoT sensor data (simplified version of original)."""
        sensor_data = []
        for ts_id in self.trainset_ids:
            profile = self.trainset_profiles[ts_id]
            
            # Sensor readings affected by trainset age/condition
            reliability_factor = profile["base_reliability"]
            
            sensors = {
                "trainset_id": ts_id,
                "timestamp": datetime.now().isoformat(),
                "vibration": {
                    "bogie_1": round(random.uniform(0.5, 3.5) / reliability_factor, 2),
                    "bogie_2": round(random.uniform(0.5, 3.5) / reliability_factor, 2),
                    "unit": "mm/s"
                },
                "temperature": {
                    "motor_1": round(random.uniform(45, 85) + (1 - reliability_factor) * 10, 1),
                    "motor_2": round(random.uniform(45, 85) + (1 - reliability_factor) * 10, 1),
                    "unit": "°C"
                },
                "overall_condition": "Good" if reliability_factor > 0.85 else "Fair" if reliability_factor > 0.75 else "Poor"
            }
            sensor_data.append(sensors)
        return sensor_data
    
    def _generate_maintenance_schedule(self) -> List[Dict]:
        """Generate maintenance schedules based on trainset profiles."""
        schedules = []
        maintenance_types = ["A-Check", "B-Check", "C-Check", "D-Check", "Overhaul"]
        
        for ts_id in self.trainset_ids:
            profile = self.trainset_profiles[ts_id]
            
            # Maintenance frequency based on age and mileage
            if profile["total_mileage_km"] > 1500000 or profile["age_years"] > 10:
                maint_type = random.choice(["C-Check", "D-Check", "Overhaul"])
                urgency = "Mandatory"
            elif profile["base_reliability"] < 0.8:
                maint_type = random.choice(["B-Check", "C-Check"])
                urgency = "Scheduled"
            else:
                maint_type = random.choice(["A-Check", "B-Check"])
                urgency = "Optional"
            
            schedule = {
                "trainset_id": ts_id,
                "maintenance_type": maint_type,
                "scheduled_date": (datetime.now() + timedelta(days=random.randint(1, 90))).isoformat(),
                "estimated_duration_hours": {
                    "A-Check": random.randint(4, 8),
                    "B-Check": random.randint(12, 24),
                    "C-Check": random.randint(48, 72),
                    "D-Check": random.randint(120, 200),
                    "Overhaul": random.randint(300, 500)
                }[maint_type],
                "priority": urgency,
                "km_since_last_maintenance": profile["total_mileage_km"] % 50000,
                "status": "Overdue" if urgency == "Mandatory" and random.random() < 0.3 else "Scheduled"
            }
            schedules.append(schedule)
        
        return schedules
    
    def _generate_performance_metrics(self) -> List[Dict]:
        """Generate performance metrics (simplified)."""
        metrics = []
        for ts_id in self.trainset_ids:
            profile = self.trainset_profiles[ts_id]
            
            # Performance correlated with reliability
            for days_ago in range(7):  # Last week only for enhanced version
                date = datetime.now() - timedelta(days=days_ago)
                
                availability = profile["base_reliability"] > random.uniform(0.7, 0.95)
                punctuality = min(100, profile["base_reliability"] * 100 + random.uniform(-5, 5))
                
                metric = {
                    "trainset_id": ts_id,
                    "date": date.date().isoformat(),
                    "service_availability": availability,
                    "punctuality_percent": round(punctuality, 2),
                    "km_traveled": random.randint(150, 450) if availability else 0,
                    "reliability_score": round(profile["base_reliability"], 3)
                }
                metrics.append(metric)
        
        return metrics
    
    def _generate_cleaning_slots(self) -> List[Dict]:
        """Generate cleaning bay data (simplified)."""
        bays = ["Cleaning-Bay-1", "Cleaning-Bay-2", "Cleaning-Bay-3"]
        shifts = ["Morning", "Afternoon", "Night"]
        
        slots = []
        for bay in bays:
            for shift in shifts:
                slot = {
                    "bay_name": bay,
                    "date": datetime.now().date().isoformat(),
                    "shift": shift,
                    "capacity": random.randint(3, 5),
                    "occupied": random.randint(1, 4),
                    "available": random.randint(0, 2)
                }
                slots.append(slot)
        return slots
    
    def _generate_manual_overrides(self) -> List[Dict]:
        """Generate manual overrides (simplified)."""
        overrides = []
        for _ in range(random.randint(2, 5)):
            override = {
                "override_id": str(uuid.uuid4()),
                "trainset_id": random.choice(self.trainset_ids),
                "timestamp": datetime.now().isoformat(),
                "supervisor_id": f"SUP-{random.randint(100, 999)}",
                "action": random.choice(["Force-Induction", "Hold-Back", "Priority-Change"]),
                "reason": random.choice([
                    "Emergency service requirement",
                    "VIP movement",  
                    "Component inspection needed"
                ])
            }
            overrides.append(override)
        return overrides
    
    def _generate_external_factors(self) -> Dict:
        """Generate external factors (simplified)."""
        return {
            "date": datetime.now().date().isoformat(),
            "weather": {
                "condition": random.choice(["Clear", "Cloudy", "Rainy"]),
                "temperature": round(random.uniform(20, 35), 1)
            },
            "ridership_forecast": {
                "expected_passengers": random.randint(80000, 150000),
                "load_factor": round(random.uniform(0.6, 0.9), 2)
            }
        }
    
    def save_to_json(self, filename: str = "metro_enhanced_data.json") -> Dict:
        """Save enhanced data to JSON file."""
        data = self.generate_complete_enhanced_dataset()
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n✅ Enhanced synthetic data saved to {filename}")
        self._print_data_summary(data)
        return data
    
    def _print_data_summary(self, data: Dict):
        """Print summary of generated data."""
        print(f"\n📊 Enhanced Dataset Summary:")
        print(f"{'='*50}")
        print(f"Trainsets: {len(data['trainset_status'])}")
        print(f"Available for service: {sum(1 for ts in data['trainset_status'] if ts['operational_status'] == 'Available')}")
        print(f"Fitness Certificates: {len(data['fitness_certificates'])}")
        print(f"- Valid: {sum(1 for cert in data['fitness_certificates'] if cert['status'] == 'Valid')}")
        print(f"- Expired/Expiring: {sum(1 for cert in data['fitness_certificates'] if cert['status'] in ['Expired', 'Expiring-Soon'])}")
        print(f"Job Cards: {len(data['job_cards'])}")
        print(f"- Critical: {sum(1 for job in data['job_cards'] if job['priority'] == 'Critical')}")
        print(f"- Open: {sum(1 for job in data['job_cards'] if job['status'] == 'Open')}")
        print(f"Component Health: {len(data['component_health'])}")
        print(f"- Warning/Critical: {sum(1 for comp in data['component_health'] if comp['status'] in ['Warning', 'Critical'])}")
        print(f"Branding Contracts: {len(data['branding_contracts'])}")
        print(f"- At Risk: {sum(1 for brand in data['branding_contracts'] if brand['status'] == 'At-Risk')}")
        
        # Optimization challenges
        challenges = []
        critical_jobs = sum(1 for job in data['job_cards'] if job['priority'] == 'Critical' and job['status'] == 'Open')
        if critical_jobs > 0:
            challenges.append(f"{critical_jobs} critical jobs blocking service")
        
        expired_certs = sum(1 for cert in data['fitness_certificates'] if cert['status'] == 'Expired')
        if expired_certs > 0:
            challenges.append(f"{expired_certs} expired certificates")
        
        at_risk_brands = sum(1 for brand in data['branding_contracts'] if brand['status'] == 'At-Risk')
        if at_risk_brands > 0:
            challenges.append(f"{at_risk_brands} underperforming brand contracts")
        
        if challenges:
            print(f"\n🎯 Optimization Challenges:")
            for challenge in challenges:
                print(f"  • {challenge}")
        
        print(f"\n🚀 Ready for optimization!")


# Usage example
if __name__ == "__main__":
    # Generate enhanced data
    generator = EnhancedMetroDataGenerator(num_trainsets=25, seed=42)  # Reproducible results
    data = generator.save_to_json("metro_enhanced_data.json")
    
    # Also generate original format for compatibility
    print(f"\n📁 Generating backward-compatible data...")
    from synthetic_base import MetroSyntheticDataGenerator
    original_gen = MetroSyntheticDataGenerator(num_trainsets=25)
    original_data = original_gen.save_to_json("metro_synthetic_data.json")
    
    print(f"\n✅ Both datasets generated:")
    print(f"  • metro_enhanced_data.json (Enhanced with realistic correlations)")
    print(f"  • metro_synthetic_data.json (Original format for compatibility)")