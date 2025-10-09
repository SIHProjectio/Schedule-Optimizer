import json
import random
from datetime import datetime, timedelta
from typing import Dict, List
import uuid

class MetroSyntheticDataGenerator:
    """Generate synthetic data for metro trainset scheduling system"""
    
    def __init__(self, num_trainsets: int = 25):
        self.num_trainsets = num_trainsets
        self.trainset_ids = [f"TS-{str(i+1).zfill(3)}" for i in range(num_trainsets)]
        self.departments = ["Rolling Stock", "Signalling", "Telecom"]
        self.brands = ["Brand-A", "Brand-B", "Brand-C", "Brand-D", "Brand-E"]
        
    def generate_trainset_status(self) -> List[Dict]:
        """Generate current operational status for all trainsets"""
        statuses = []
        for ts_id in self.trainset_ids:
            status = {
                "trainset_id": ts_id,
                "current_location": random.choice(["Depot-A", "Depot-B", "In-Service", "IBL", "Cleaning-Bay"]),
                "operational_status": random.choice(["Available", "In-Service", "Maintenance", "Standby"]),
                "last_service_date": (datetime.now() - timedelta(days=random.randint(1, 30))).isoformat(),
                "total_mileage_km": random.randint(50000, 200000),
                "daily_mileage_km": random.randint(200, 400),
                "operational_hours": random.randint(5000, 15000),
                "last_updated": datetime.now().isoformat()
            }
            statuses.append(status)
        return statuses
    
    def generate_fitness_certificates(self) -> List[Dict]:
        """Generate fitness certificates from different departments"""
        certificates = []
        for ts_id in self.trainset_ids:
            for dept in self.departments:
                cert = {
                    "certificate_id": str(uuid.uuid4()),
                    "trainset_id": ts_id,
                    "department": dept,
                    "issue_date": (datetime.now() - timedelta(days=random.randint(1, 60))).isoformat(),
                    "expiry_date": (datetime.now() + timedelta(days=random.randint(-5, 90))).isoformat(),
                    "status": random.choice(["Valid", "Valid", "Valid", "Expired", "Expiring-Soon"]),
                    "inspector_id": f"INS-{random.randint(100, 999)}",
                    "compliance_score": random.randint(75, 100),
                    "remarks": random.choice(["All systems operational", "Minor issues noted", "Requires follow-up", ""])
                }
                certificates.append(cert)
        return certificates
    
    def generate_job_cards(self) -> List[Dict]:
        """Generate IBM Maximo job cards"""
        job_types = ["Preventive", "Corrective", "Breakdown", "Inspection"]
        priorities = ["Critical", "High", "Medium", "Low"]
        
        job_cards = []
        for ts_id in self.trainset_ids:
            # Random number of job cards per trainset
            num_jobs = random.randint(0, 5)
            for _ in range(num_jobs):
                job = {
                    "job_card_id": f"JC-{random.randint(10000, 99999)}",
                    "trainset_id": ts_id,
                    "work_order_number": f"WO-{random.randint(100000, 999999)}",
                    "job_type": random.choice(job_types),
                    "priority": random.choice(priorities),
                    "status": random.choice(["Open", "Open", "Closed", "In-Progress", "Pending-Parts"]),
                    "created_date": (datetime.now() - timedelta(days=random.randint(1, 30))).isoformat(),
                    "estimated_completion": (datetime.now() + timedelta(hours=random.randint(2, 48))).isoformat(),
                    "assigned_technician": f"TECH-{random.randint(100, 999)}",
                    "component": random.choice(["Brakes", "HVAC", "Doors", "Bogies", "Pantograph", "Electrical"]),
                    "description": "Routine maintenance required",
                    "estimated_hours": random.randint(2, 24),
                    "cost_estimate": random.randint(5000, 50000)
                }
                job_cards.append(job)
        return job_cards
    
    def generate_component_health(self) -> List[Dict]:
        """Generate IoT sensor data for component health"""
        components = {
            "Bogie": {"wear_threshold": 80, "unit": "% wear"},
            "Brake_Pad": {"wear_threshold": 70, "unit": "% remaining"},
            "HVAC": {"wear_threshold": 85, "unit": "% efficiency"},
            "Door_System": {"wear_threshold": 90, "unit": "cycles"},
            "Pantograph": {"wear_threshold": 75, "unit": "% condition"},
            "Battery": {"wear_threshold": 80, "unit": "% capacity"}
        }
        
        health_data = []
        for ts_id in self.trainset_ids:
            for comp, meta in components.items():
                health = {
                    "trainset_id": ts_id,
                    "component": comp,
                    "health_score": random.randint(60, 100),
                    "wear_level": random.randint(0, 100),
                    "threshold": meta["wear_threshold"],
                    "unit": meta["unit"],
                    "status": random.choice(["Good", "Good", "Good", "Fair", "Warning"]),
                    "next_maintenance_km": random.randint(1000, 5000),
                    "last_maintenance_date": (datetime.now() - timedelta(days=random.randint(1, 60))).isoformat(),
                    "predicted_failure_date": (datetime.now() + timedelta(days=random.randint(30, 180))).isoformat(),
                    "timestamp": datetime.now().isoformat()
                }
                health_data.append(health)
        return health_data
    
    def generate_iot_sensors(self) -> List[Dict]:
        """Generate real-time IoT sensor readings"""
        sensor_data = []
        for ts_id in self.trainset_ids:
            sensors = {
                "trainset_id": ts_id,
                "timestamp": datetime.now().isoformat(),
                "vibration": {
                    "bogie_1": round(random.uniform(0.5, 3.5), 2),
                    "bogie_2": round(random.uniform(0.5, 3.5), 2),
                    "unit": "mm/s"
                },
                "temperature": {
                    "motor_1": round(random.uniform(45, 85), 1),
                    "motor_2": round(random.uniform(45, 85), 1),
                    "brake_disc": round(random.uniform(25, 120), 1),
                    "cabin": round(random.uniform(18, 28), 1),
                    "unit": "°C"
                },
                "pressure": {
                    "brake_system": round(random.uniform(5.5, 8.5), 2),
                    "pneumatic_doors": round(random.uniform(6.0, 8.0), 2),
                    "unit": "bar"
                },
                "electrical": {
                    "voltage": round(random.uniform(730, 770), 1),
                    "current": round(random.uniform(100, 400), 1),
                    "power_consumption": round(random.uniform(200, 600), 1),
                    "battery_voltage": round(random.uniform(70, 85), 1)
                },
                "door_cycles": {
                    "door_1": random.randint(50000, 200000),
                    "door_2": random.randint(50000, 200000),
                    "door_3": random.randint(50000, 200000),
                    "door_4": random.randint(50000, 200000)
                },
                "gps": {
                    "latitude": round(random.uniform(9.9, 10.1), 6),
                    "longitude": round(random.uniform(76.2, 76.4), 6),
                    "speed_kmh": round(random.uniform(0, 80), 1)
                }
            }
            sensor_data.append(sensors)
        return sensor_data
    
    def generate_branding_contracts(self) -> List[Dict]:
        """Generate branding/advertisement contract data"""
        contracts = []
        for ts_id in random.sample(self.trainset_ids, random.randint(10, 15)):
            contract = {
                "trainset_id": ts_id,
                "brand": random.choice(self.brands),
                "contract_id": f"ADV-{random.randint(1000, 9999)}",
                "start_date": (datetime.now() - timedelta(days=random.randint(30, 180))).isoformat(),
                "end_date": (datetime.now() + timedelta(days=random.randint(30, 365))).isoformat(),
                "contracted_exposure_hours": random.randint(2000, 5000),
                "actual_exposure_hours": random.randint(1500, 4500),
                "daily_target_hours": random.randint(8, 12),
                "contract_value": random.randint(500000, 2000000),
                "penalty_per_hour_shortfall": random.randint(500, 2000),
                "status": random.choice(["Active", "Active", "Active", "At-Risk", "Compliant"]),
                "priority_level": random.choice(["High", "Medium", "Low"])
            }
            contracts.append(contract)
        return contracts
    
    def generate_maintenance_schedule(self) -> List[Dict]:
        """Generate planned maintenance schedules"""
        maintenance_types = ["A-Check", "B-Check", "C-Check", "D-Check", "Overhaul"]
        
        schedules = []
        for ts_id in self.trainset_ids:
            schedule = {
                "trainset_id": ts_id,
                "maintenance_type": random.choice(maintenance_types),
                "scheduled_date": (datetime.now() + timedelta(days=random.randint(1, 60))).isoformat(),
                "estimated_duration_hours": random.randint(4, 72),
                "bay_required": random.choice(["IBL-1", "IBL-2", "Cleaning-Bay", "Workshop"]),
                "priority": random.choice(["Mandatory", "Scheduled", "Optional"]),
                "km_since_last_maintenance": random.randint(5000, 20000),
                "days_since_last_maintenance": random.randint(15, 90),
                "status": random.choice(["Scheduled", "Pending", "Overdue"])
            }
            schedules.append(schedule)
        return schedules
    
    def generate_performance_metrics(self) -> List[Dict]:
        """Generate historical performance data"""
        metrics = []
        for ts_id in self.trainset_ids:
            # Last 30 days of performance
            for days_ago in range(30):
                date = datetime.now() - timedelta(days=days_ago)
                metric = {
                    "trainset_id": ts_id,
                    "date": date.date().isoformat(),
                    "service_availability": random.choice([True, True, True, True, False]),
                    "punctuality_percent": round(random.uniform(95, 100), 2),
                    "km_traveled": random.randint(150, 450),
                    "trips_completed": random.randint(15, 35),
                    "breakdown_count": random.randint(0, 2),
                    "delay_minutes": random.randint(0, 30),
                    "passenger_count": random.randint(5000, 15000),
                    "energy_consumed_kwh": round(random.uniform(300, 800), 2),
                    "average_speed_kmh": round(random.uniform(35, 55), 1)
                }
                metrics.append(metric)
        return metrics
    
    def generate_cleaning_slots(self) -> List[Dict]:
        """Generate cleaning bay availability and schedules"""
        bays = ["Cleaning-Bay-1", "Cleaning-Bay-2", "Cleaning-Bay-3"]
        shifts = ["Morning", "Afternoon", "Night"]
        
        slots = []
        for bay in bays:
            for shift in shifts:
                slot = {
                    "bay_name": bay,
                    "date": datetime.now().date().isoformat(),
                    "shift": shift,
                    "capacity": random.randint(2, 4),
                    "occupied": random.randint(0, 3),
                    "available": random.randint(0, 2),
                    "scheduled_trainsets": random.sample(self.trainset_ids, random.randint(0, 3)),
                    "manpower_available": random.randint(2, 6),
                    "estimated_duration_hours": random.randint(2, 4)
                }
                slots.append(slot)
        return slots
    
    def generate_manual_overrides(self) -> List[Dict]:
        """Generate supervisor manual override entries"""
        overrides = []
        for _ in range(random.randint(3, 8)):
            override = {
                "override_id": str(uuid.uuid4()),
                "trainset_id": random.choice(self.trainset_ids),
                "timestamp": datetime.now().isoformat(),
                "supervisor_id": f"SUP-{random.randint(100, 999)}",
                "action": random.choice(["Force-Induction", "Hold-Back", "Priority-Change", "IBL-Delay"]),
                "reason": random.choice([
                    "Emergency service requirement",
                    "VIP movement",
                    "Component inspection needed",
                    "Branding priority",
                    "Safety precaution"
                ]),
                "priority": random.choice(["Critical", "High", "Medium"]),
                "expiry": (datetime.now() + timedelta(hours=24)).isoformat()
            }
            overrides.append(override)
        return overrides
    
    def generate_external_factors(self) -> Dict:
        """Generate external factors affecting operations"""
        return {
            "date": datetime.now().date().isoformat(),
            "weather": {
                "temperature": round(random.uniform(20, 35), 1),
                "humidity": random.randint(60, 90),
                "rainfall_mm": round(random.uniform(0, 50), 1),
                "condition": random.choice(["Clear", "Cloudy", "Rainy", "Stormy"])
            },
            "special_events": random.choice([
                None,
                "Festival - High ridership expected",
                "VIP visit - Route restrictions",
                "Maintenance window - Track work"
            ]),
            "ridership_forecast": {
                "expected_passengers": random.randint(80000, 150000),
                "peak_hours": ["08:00-10:00", "17:00-20:00"],
                "load_factor": round(random.uniform(0.6, 0.9), 2)
            },
            "track_conditions": {
                "status": random.choice(["Normal", "Caution", "Restricted"]),
                "maintenance_zones": random.randint(0, 3),
                "speed_restrictions": random.randint(0, 2)
            }
        }
    
    def generate_complete_dataset(self) -> Dict:
        """Generate complete synthetic dataset for metro scheduling"""
        dataset = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "num_trainsets": self.num_trainsets,
                "system": "Kochi Metro Rail",
                "data_version": "1.0"
            },
            "trainset_status": self.generate_trainset_status(),
            "fitness_certificates": self.generate_fitness_certificates(),
            "job_cards": self.generate_job_cards(),
            "component_health": self.generate_component_health(),
            "iot_sensors": self.generate_iot_sensors(),
            "branding_contracts": self.generate_branding_contracts(),
            "maintenance_schedule": self.generate_maintenance_schedule(),
            "performance_metrics": self.generate_performance_metrics(),
            "cleaning_slots": self.generate_cleaning_slots(),
            "manual_overrides": self.generate_manual_overrides(),
            "external_factors": self.generate_external_factors()
        }
        return dataset
    
    def save_to_json(self, filename: str = "metro_synthetic_data.json"):
        """Save generated data to JSON file"""
        data = self.generate_complete_dataset()
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Synthetic data generated and saved to {filename}")
        return data


# Usage example
if __name__ == "__main__":
    generator = MetroSyntheticDataGenerator(num_trainsets=25)
    
    # Generate and save complete dataset
    data = generator.save_to_json("metro_synthetic_data.json")
    
    # Print summary
    print(f"\nDataset Summary:")
    print(f"Trainsets: {len(data['trainset_status'])}")
    print(f"Fitness Certificates: {len(data['fitness_certificates'])}")
    print(f"Job Cards: {len(data['job_cards'])}")
    print(f"Component Health Records: {len(data['component_health'])}")
    print(f"IoT Sensor Readings: {len(data['iot_sensors'])}")
    print(f"Branding Contracts: {len(data['branding_contracts'])}")
    print(f"Performance Metrics: {len(data['performance_metrics'])}")