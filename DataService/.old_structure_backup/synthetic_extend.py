import json
import random
from datetime import datetime, time, timedelta
from typing import Dict, List
import uuid


class TimeSlottedDataGenerator:
    """Extension to generate time-slotted scheduling data"""
    
    def __init__(self, num_trainsets: int = 25):
        self.num_trainsets = num_trainsets
        self.trainset_ids = [f"TS-{str(i+1).zfill(3)}" for i in range(num_trainsets)]
        
        # Define time slots
        self.time_slots = [
            {"id": "SLOT-1", "name": "Early Morning", "start": "05:30", "end": "08:00", "type": "peak"},
            {"id": "SLOT-2", "name": "Morning Peak", "start": "08:00", "end": "11:00", "type": "peak"},
            {"id": "SLOT-3", "name": "Mid-day", "start": "11:00", "end": "14:00", "type": "off-peak"},
            {"id": "SLOT-4", "name": "Afternoon", "start": "14:00", "end": "17:00", "type": "off-peak"},
            {"id": "SLOT-5", "name": "Evening Peak", "start": "17:00", "end": "20:00", "type": "peak"},
            {"id": "SLOT-6", "name": "Night", "start": "20:00", "end": "22:30", "type": "off-peak"}
        ]
    
    def generate_trainset_availability(self) -> List[Dict]:
        """Generate time-slot availability for each trainset"""
        availability = []
        
        for ts_id in self.trainset_ids:
            # Determine max service hours (some trains limited availability)
            max_hours = random.choice([
                2,   # Very limited (10% of fleet)
                4,   # Limited (15%)
                8,   # Partial day (20%)
                12,  # Most of day (25%)
                17   # Full day (30%)
            ])
            
            # Generate slot availability
            slots_available = []
            total_hours = 0
            
            for slot in self.time_slots:
                start = datetime.strptime(slot["start"], "%H:%M")
                end = datetime.strptime(slot["end"], "%H:%M")
                slot_duration = (end - start).seconds / 3600
                
                # Can this trainset work this slot?
                if total_hours + slot_duration <= max_hours:
                    available = random.choice([True, True, True, False])  # 75% available
                    if available:
                        total_hours += slot_duration
                else:
                    available = False
                
                slots_available.append({
                    "slot_id": slot["id"],
                    "slot_name": slot["name"],
                    "available": available,
                    "reliability_score": random.randint(85, 99)  # Slot-specific reliability
                })
            
            availability.append({
                "trainset_id": ts_id,
                "max_service_hours": max_hours,
                "actual_available_hours": total_hours,
                "slots": slots_available,
                "turnaround_time_minutes": random.randint(20, 45),
                "last_service_slot": random.choice([s["id"] for s in self.time_slots])
            })
        
        return availability
    
    def generate_demand_forecast(self) -> List[Dict]:
        """Generate passenger demand forecast per time slot"""
        demand = []
        
        for slot in self.time_slots:
            peak_multiplier = 1.5 if slot["type"] == "peak" else 1.0
            
            demand_data = {
                "slot_id": slot["id"],
                "slot_name": slot["name"],
                "start_time": slot["start"],
                "end_time": slot["end"],
                "type": slot["type"],
                "expected_passengers": int(random.randint(15000, 25000) * peak_multiplier),
                "required_trains": 20 if slot["type"] == "peak" else 15,
                "load_factor": round(random.uniform(0.7, 0.95) if slot["type"] == "peak" else random.uniform(0.4, 0.7), 2),
                "headway_minutes": 6 if slot["type"] == "peak" else 10,
                "priority_score": 10 if slot["type"] == "peak" else 5
            }
            demand.append(demand_data)
        
        return demand
    
    def generate_rotation_constraints(self) -> List[Dict]:
        """Generate constraints for train rotations between slots"""
        constraints = []
        
        for i, slot in enumerate(self.time_slots):
            if i < len(self.time_slots) - 1:
                next_slot = self.time_slots[i + 1]
                
                constraint = {
                    "from_slot": slot["id"],
                    "to_slot": next_slot["id"],
                    "minimum_turnaround_minutes": 30,
                    "depot_capacity_limit": 8,  # Max trains that can return simultaneously
                    "max_simultaneous_changeovers": 5,
                    "cleaning_required": random.choice([True, False]),
                    "quick_inspection_time_minutes": 15
                }
                constraints.append(constraint)
        
        return constraints
    
    def generate_slot_specific_branding(self) -> List[Dict]:
        """Generate branding contracts with slot-specific requirements"""
        brands = ["Brand-A", "Brand-B", "Brand-C", "Brand-D", "Brand-E"]
        contracts = []
        
        for brand in brands:
            # Each brand has slot preferences
            preferred_slots = random.sample([s["id"] for s in self.time_slots], k=random.randint(2, 4))
            
            contract = {
                "brand": brand,
                "contract_id": f"ADV-{random.randint(1000, 9999)}",
                "assigned_trainsets": random.sample(self.trainset_ids, k=random.randint(2, 4)),
                "preferred_slots": preferred_slots,
                "required_exposure_hours_per_slot": {
                    slot_id: random.randint(2, 4) for slot_id in preferred_slots
                },
                "total_daily_target_hours": random.randint(8, 15),
                "peak_hour_bonus": 1.5,  # More value during peak hours
                "contract_value_per_hour": random.randint(1000, 3000),
                "penalty_per_hour_shortfall": random.randint(500, 1500),
                "priority": random.choice(["High", "Medium", "Low"])
            }
            contracts.append(contract)
        
        return contracts
    
    def generate_maintenance_windows(self) -> List[Dict]:
        """Generate maintenance tasks with specific time window requirements"""
        windows = []
        
        for ts_id in random.sample(self.trainset_ids, k=random.randint(5, 10)):
            # Maintenance must happen in specific slots
            required_slots = random.randint(1, 3)
            blocked_slots = random.sample([s["id"] for s in self.time_slots], k=required_slots)
            
            window = {
                "trainset_id": ts_id,
                "maintenance_id": f"MAINT-{random.randint(1000, 9999)}",
                "type": random.choice(["Preventive", "Scheduled", "Urgent"]),
                "required_duration_hours": random.randint(2, 6),
                "blocked_slots": blocked_slots,
                "flexibility": random.choice(["Fixed", "Flexible", "Postponable"]),
                "deadline_slot": random.choice([s["id"] for s in self.time_slots]),
                "cost_if_delayed": random.randint(10000, 50000)
            }
            windows.append(window)
        
        return windows
    
    def generate_crew_availability(self) -> List[Dict]:
        """Generate crew/operator availability per slot"""
        crew = []
        
        for slot in self.time_slots:
            crew_data = {
                "slot_id": slot["id"],
                "slot_name": slot["name"],
                "available_operators": random.randint(18, 25),
                "required_operators_per_train": 2,
                "max_trains_by_crew": random.randint(18, 25) // 2,
                "overtime_rate": round(random.uniform(1.2, 1.8), 2) if slot["type"] == "peak" else 1.0,
                "crew_fatigue_factor": round(random.uniform(0.85, 1.0), 2)
            }
            crew.append(crew_data)
        
        return crew
    
    def generate_energy_optimization_data(self) -> List[Dict]:
        """Generate energy cost data per slot"""
        energy = []
        
        for slot in self.time_slots:
            energy_data = {
                "slot_id": slot["id"],
                "slot_name": slot["name"],
                "electricity_rate_per_kwh": round(random.uniform(6.0, 12.0), 2),
                "off_peak_discount": 0.7 if slot["type"] == "off-peak" else 1.0,
                "average_consumption_per_train_kwh": round(random.uniform(400, 700), 1),
                "regenerative_braking_efficiency": round(random.uniform(0.25, 0.35), 2),
                "grid_load_factor": round(random.uniform(0.6, 0.95), 2)
            }
            energy.append(energy_data)
        
        return energy
    
    def generate_historical_slot_performance(self) -> List[Dict]:
        """Generate historical performance data per slot"""
        history = []
        
        for days_ago in range(30):
            date = (datetime.now() - timedelta(days=days_ago)).date()
            
            for slot in self.time_slots:
                performance = {
                    "date": date.isoformat(),
                    "slot_id": slot["id"],
                    "trains_deployed": random.randint(15, 22),
                    "trains_planned": 20 if slot["type"] == "peak" else 15,
                    "actual_passengers": random.randint(12000, 28000),
                    "delays_minutes": random.randint(0, 15),
                    "breakdowns": random.randint(0, 2),
                    "service_quality_score": round(random.uniform(85, 99), 1),
                    "energy_consumed_kwh": round(random.uniform(8000, 14000), 1)
                }
                history.append(performance)
        
        return history
    
    def generate_complete_timeslot_dataset(self) -> Dict:
        """Generate complete time-slotted scheduling dataset"""
        dataset = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "num_trainsets": self.num_trainsets,
                "num_slots": len(self.time_slots),
                "scheduling_mode": "time-slotted",
                "version": "2.0"
            },
            "time_slots": self.time_slots,
            "trainset_availability": self.generate_trainset_availability(),
            "demand_forecast": self.generate_demand_forecast(),
            "rotation_constraints": self.generate_rotation_constraints(),
            "slot_branding_contracts": self.generate_slot_specific_branding(),
            "maintenance_windows": self.generate_maintenance_windows(),
            "crew_availability": self.generate_crew_availability(),
            "energy_optimization": self.generate_energy_optimization_data(),
            "historical_performance": self.generate_historical_slot_performance()
        }
        return dataset
    
    def save_to_json(self, filename: str = "metro_timeslot_data.json"):
        """Save time-slotted data to JSON"""
        data = self.generate_complete_timeslot_dataset()
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Time-slotted data generated: {filename}")
        return data


# Usage
if __name__ == "__main__":
    generator = TimeSlottedDataGenerator(num_trainsets=25)
    data = generator.save_to_json("metro_timeslot_data.json")
    
    print(f"\nDataset Summary:")
    print(f"Time Slots: {len(data['time_slots'])}")
    print(f"Trainset Availability Records: {len(data['trainset_availability'])}")
    print(f"Demand Forecasts: {len(data['demand_forecast'])}")
    print(f"Rotation Constraints: {len(data['rotation_constraints'])}")
    print(f"Branding Contracts: {len(data['slot_branding_contracts'])}")
    print(f"Maintenance Windows: {len(data['maintenance_windows'])}")
    print(f"Historical Records: {len(data['historical_performance'])}")