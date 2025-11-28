"""
Trainset scheduling evaluation module.
Handles constraint checking and objective function calculation.
"""
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from .models import OptimizationConfig, TrainsetConstraints


# Status normalization mappings (backend format -> internal format)
CERTIFICATE_STATUS_MAP = {
    'PENDING': 'Expiring-Soon',
    'IN_PROGRESS': 'Expiring-Soon',
    'ISSUED': 'Valid',
    'EXPIRED': 'Expired',
    'SUSPENDED': 'Suspended',
    'REVOKED': 'Expired',
    'RENEWED': 'Valid',
    'CANCELLED': 'Expired',
}

COMPONENT_STATUS_MAP = {
    'EXCELLENT': 'Good',
    'GOOD': 'Good',
    'FAIR': 'Fair',
    'POOR': 'Warning',
    'CRITICAL': 'Critical',
    'FAILED': 'Critical',
}


def normalize_certificate_status(status: str) -> str:
    """Normalize certificate status to internal format."""
    return CERTIFICATE_STATUS_MAP.get(status, status)


def normalize_component_status(status: str) -> str:
    """Normalize component status to internal format."""
    return COMPONENT_STATUS_MAP.get(status, status)


class TrainsetSchedulingEvaluator:
    """Multi-objective evaluator for trainset scheduling optimization."""
    
    def __init__(self, data: Dict, config: Optional[OptimizationConfig] = None):
        self.data = data
        self.config = config or OptimizationConfig()
        self.trainsets = [ts['trainset_id'] for ts in data['trainset_status']]
        self.num_trainsets = len(self.trainsets)
        
        # Build lookup dictionaries
        self._build_lookups()
        
    def _build_lookups(self):
        """Build fast lookup dictionaries for optimization."""
        self.status_map = {ts['trainset_id']: ts for ts in self.data['trainset_status']}
        
        # Fitness certificates by trainset and department
        self.fitness_map = {}
        for cert in self.data['fitness_certificates']:
            ts_id = cert['trainset_id']
            if ts_id not in self.fitness_map:
                self.fitness_map[ts_id] = {}
            self.fitness_map[ts_id][cert['department']] = cert
        
        # Job cards by trainset (optional - may be empty)
        self.job_map = {}
        for job in self.data.get('job_cards', []):
            ts_id = job['trainset_id']
            if ts_id not in self.job_map:
                self.job_map[ts_id] = []
            self.job_map[ts_id].append(job)
        
        # Component health by trainset
        self.health_map = {}
        for health in self.data['component_health']:
            ts_id = health['trainset_id']
            if ts_id not in self.health_map:
                self.health_map[ts_id] = []
            self.health_map[ts_id].append(health)
        
        # Branding contracts
        self.brand_map = {}
        for brand in self.data.get('branding_contracts', []):
            ts_id = brand['trainset_id']
            self.brand_map[ts_id] = brand
        
        # Maintenance schedule
        self.maint_map = {}
        for maint in self.data.get('maintenance_schedule', []):
            ts_id = maint['trainset_id']
            self.maint_map[ts_id] = maint
    
    def get_trainset_constraints(self, trainset_id: str) -> TrainsetConstraints:
        """Get all constraints for a specific trainset."""
        try:
            # Check fitness certificates
            has_valid_certs = True
            if trainset_id in self.fitness_map:
                for dept, cert in self.fitness_map[trainset_id].items():
                    # Normalize status to handle both legacy and backend formats
                    status = normalize_certificate_status(cert['status'])
                    if status in ['Expired']:
                        has_valid_certs = False
                        break
                    try:
                        expiry = datetime.fromisoformat(cert['expiry_date'])
                        if expiry < datetime.now():
                            has_valid_certs = False
                            break
                    except ValueError:
                        has_valid_certs = False
                        break
            else:
                has_valid_certs = False
            
            # Check critical jobs
            has_critical_jobs = False
            if trainset_id in self.job_map:
                for job in self.job_map[trainset_id]:
                    if job['status'] == 'Open' and job['priority'] == 'Critical':
                        has_critical_jobs = True
                        break
            
            # Check component warnings
            component_warnings = []
            if trainset_id in self.health_map:
                for health in self.health_map[trainset_id]:
                    # Normalize status to handle both legacy and backend formats
                    status = normalize_component_status(health['status'])
                    if status in ['Warning', 'Critical'] and health.get('wear_level', 0) > 90:
                        component_warnings.append(health['component'])
            
            # Check maintenance status
            maintenance_due = False
            if trainset_id in self.maint_map:
                maintenance_due = self.maint_map[trainset_id]['status'] == 'Overdue'
            
            # Get mileage and service info
            status = self.status_map.get(trainset_id, {})
            mileage = status.get('total_mileage_km', 0)
            
            # Calculate days since last service
            last_service_days = 0
            if 'last_service_date' in status:
                try:
                    last_service = datetime.fromisoformat(status['last_service_date'])
                    last_service_days = (datetime.now() - last_service).days
                except ValueError:
                    last_service_days = 999  # Unknown, assume old
            
            return TrainsetConstraints(
                has_valid_certificates=has_valid_certs,
                has_critical_jobs=has_critical_jobs,
                component_warnings=component_warnings,
                maintenance_due=maintenance_due,
                mileage=mileage,
                last_service_days=last_service_days
            )
        except Exception:
            # Return safe defaults if data is malformed
            return TrainsetConstraints(
                has_valid_certificates=False,
                has_critical_jobs=True,
                component_warnings=['Unknown'],
                maintenance_due=True,
                mileage=0,
                last_service_days=999
            )
    
    def check_hard_constraints(self, trainset_id: str) -> Tuple[bool, str]:
        """Check if trainset passes hard constraints for service."""
        constraints = self.get_trainset_constraints(trainset_id)
        
        if not constraints.has_valid_certificates:
            return False, "Invalid/expired certificates"
        
        if constraints.has_critical_jobs:
            return False, "Critical maintenance jobs pending"
        
        if constraints.component_warnings:
            return False, f"Critical component wear: {', '.join(constraints.component_warnings)}"
        
        return True, "Passes all constraints"
    
    def calculate_objectives(self, solution: np.ndarray) -> Dict[str, float]:
        """Calculate multiple objectives for a solution.
        
        Solution encoding: 0=Service, 1=Standby, 2=Maintenance
        """
        objectives = {
            'service_availability': 0.0,
            'maintenance_cost': 0.0,
            'branding_compliance': 0.0,
            'mileage_balance': 0.0,
            'constraint_penalty': 0.0
        }
        
        try:
            service_trains = []
            standby_trains = []
            maint_trains = []
            
            for idx, action in enumerate(solution):
                ts_id = self.trainsets[idx]
                if action == 0:
                    service_trains.append(ts_id)
                elif action == 1:
                    standby_trains.append(ts_id)
                else:
                    maint_trains.append(ts_id)
            
            # Objective 1: Service Availability (maximize)
            availability = len(service_trains) / self.config.required_service_trains
            if len(service_trains) < self.config.required_service_trains:
                objectives['constraint_penalty'] += (self.config.required_service_trains - len(service_trains)) * 100.0
            objectives['service_availability'] = min(availability, 1.0) * 100.0
            
            # Objective 2: Mileage Balance (maximize via minimizing std dev)
            mileages = [self.status_map[ts].get('total_mileage_km', 0) for ts in service_trains]
            if mileages and len(mileages) > 1:
                std_dev = float(np.std(mileages))
                objectives['mileage_balance'] = 100.0 - min(std_dev / 1000.0, 100.0)
            else:
                objectives['mileage_balance'] = 100.0
            
            # Objective 3: Branding Compliance (maximize)
            brand_scores = []
            for ts_id in service_trains:
                if ts_id in self.brand_map:
                    contract = self.brand_map[ts_id]
                    target = contract.get('daily_target_hours', 8)
                    actual = contract.get('actual_exposure_hours', 0) / 30.0  # Daily average
                    compliance = min(actual / target, 1.0) if target > 0 else 1.0
                    brand_scores.append(compliance)
            
            objectives['branding_compliance'] = float(np.mean(brand_scores)) * 100.0 if brand_scores else 100.0
            
            # Objective 4: Maintenance Cost (minimize)
            maint_cost = 0.0
            for ts_id in service_trains:
                if ts_id in self.maint_map:
                    if self.maint_map[ts_id].get('status') == 'Overdue':
                        maint_cost += 50.0
            objectives['maintenance_cost'] = 100.0 - min(maint_cost, 100.0)
            
            # Hard constraint violations
            for ts_id in service_trains:
                valid, _ = self.check_hard_constraints(ts_id)
                if not valid:
                    objectives['constraint_penalty'] += 200.0
            
            # Standby constraint
            if len(standby_trains) < self.config.min_standby:
                objectives['constraint_penalty'] += (self.config.min_standby - len(standby_trains)) * 50.0
            
        except Exception as e:
            # Penalize heavily for any errors during evaluation
            objectives['constraint_penalty'] += 1000.0
            print(f"Error in objective calculation: {e}")
        
        return objectives
    
    def fitness_function(self, solution: np.ndarray) -> float:
        """Aggregate fitness function for minimization."""
        obj = self.calculate_objectives(solution)
        
        # Weighted sum (convert maximization objectives to minimization)
        fitness = (
            -obj['service_availability'] * 2.0 +      # Maximize (negative weight)
            -obj['branding_compliance'] * 1.5 +        # Maximize
            -obj['mileage_balance'] * 1.0 +            # Maximize
            -obj['maintenance_cost'] * 1.0 +           # Maximize
            obj['constraint_penalty'] * 5.0            # Minimize (positive weight)
        )
        
        return fitness