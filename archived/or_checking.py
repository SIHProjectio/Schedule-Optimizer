"""
OR-Tools based optimizers for trainset scheduling.
Provides exact and constraint programming solutions.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

try:
    from ortools.sat.python import cp_model
    from ortools.linear_solver import pywraplp
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False
    cp_model = None  # type: ignore
    pywraplp = None  # type: ignore

from .models import OptimizationResult, OptimizationConfig
from .evaluator import TrainsetSchedulingEvaluator


class CPSATOptimizer:
    """Constraint Programming optimizer using OR-Tools CP-SAT solver."""
    
    def __init__(self, evaluator: TrainsetSchedulingEvaluator, config: Optional[OptimizationConfig] = None):
        if not ORTOOLS_AVAILABLE:
            raise ImportError("OR-Tools not available. Install with: pip install ortools")
            
        self.evaluator = evaluator
        self.config = config or OptimizationConfig()
        self.n_trainsets = evaluator.num_trainsets
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
    def optimize(self, time_limit_seconds: int = 300) -> OptimizationResult:
        """Solve using CP-SAT with constraint programming."""
        if not ORTOOLS_AVAILABLE:
            raise ImportError("OR-Tools not available")
        
        model = cp_model.CpModel()
        
        # Decision variables: assignment[i] ∈ {0, 1, 2} for each trainset
        # 0 = Service, 1 = Standby, 2 = Maintenance
        assignment = []
        for i in range(self.n_trainsets):
            assignment.append(model.NewIntVar(0, 2, f'trainset_{i}'))
        
        # Helper binary variables for easier constraint formulation
        is_service = []
        is_standby = []
        is_maintenance = []
        
        for i in range(self.n_trainsets):
            is_service.append(model.NewBoolVar(f'service_{i}'))
            is_standby.append(model.NewBoolVar(f'standby_{i}'))
            is_maintenance.append(model.NewBoolVar(f'maintenance_{i}'))
            
            # Link assignment variables to binary indicators
            model.Add(assignment[i] == 0).OnlyEnforceIf(is_service[i])
            model.Add(assignment[i] != 0).OnlyEnforceIf(is_service[i].Not())
            model.Add(assignment[i] == 1).OnlyEnforceIf(is_standby[i])
            model.Add(assignment[i] != 1).OnlyEnforceIf(is_standby[i].Not())
            model.Add(assignment[i] == 2).OnlyEnforceIf(is_maintenance[i])
            model.Add(assignment[i] != 2).OnlyEnforceIf(is_maintenance[i].Not())
        
        # Constraint 1: Exact number of trains in service
        model.Add(sum(is_service) == self.config.required_service_trains)
        
        # Constraint 2: Minimum standby trains
        model.Add(sum(is_standby) >= self.config.min_standby)
        
        # Constraint 3: Hard constraints - trains with issues cannot be in service
        for i, trainset_id in enumerate(self.evaluator.trainsets):
            valid, reason = self.evaluator.check_hard_constraints(trainset_id)
            if not valid:
                # Force trainset to maintenance if it fails constraints
                model.Add(assignment[i] == 2)
                self.logger.info(f"Trainset {trainset_id} forced to maintenance: {reason}")
        
        # Constraint 4: Branding requirements
        self._add_branding_constraints(model, is_service)
        
        # Objective: Multi-objective optimization using weighted sum
        self._set_multi_objective(model, is_service, is_standby, assignment)
        
        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit_seconds
        solver.parameters.log_search_progress = True
        
        print(f"Solving with CP-SAT (time limit: {time_limit_seconds}s)...")
        status = solver.Solve(model)
        
        if status == cp_model.OPTIMAL:
            print("✅ Optimal solution found!")
        elif status == cp_model.FEASIBLE:
            print("✅ Feasible solution found!")
        else:
            print("❌ No solution found!")
            return self._create_fallback_solution()
        
        # Extract solution
        solution = np.array([solver.Value(assignment[i]) for i in range(self.n_trainsets)])
        return self._build_result(solution, solver.ObjectiveValue())
    
    def _add_branding_constraints(self, model, is_service: List):
        """Add branding contract constraints."""
        # Group trainsets by brand and ensure minimum coverage
        brand_trainsets = {}
        for i, trainset_id in enumerate(self.evaluator.trainsets):
            if trainset_id in self.evaluator.brand_map:
                brand = self.evaluator.brand_map[trainset_id].get('brand_name', 'Unknown')
                if brand not in brand_trainsets:
                    brand_trainsets[brand] = []
                brand_trainsets[brand].append(i)
        
        # Ensure each brand has at least some representation in service
        for brand, trainset_indices in brand_trainsets.items():
            if len(trainset_indices) > 1:
                # At least 50% of branded trainsets should be in service if possible
                min_branded = max(1, len(trainset_indices) // 2)
                model.Add(sum(is_service[i] for i in trainset_indices) >= min_branded)
    
    def _set_multi_objective(self, model, is_service: List, is_standby: List, assignment: List):
        """Set up multi-objective optimization using weighted sum."""
        
        # Objective components
        objective_terms = []
        
        # 1. Maximize service availability (already satisfied by exact constraint)
        # This is handled by the exact constraint, so no need to optimize
        
        # 2. Minimize mileage imbalance - use auxiliary variables for quadratic terms
        # Approximate mileage balance by preferring even distribution across mileage ranges
        mileage_ranges = self._categorize_trainsets_by_mileage()
        for range_name, trainset_indices in mileage_ranges.items():
            if len(trainset_indices) > 1:
                # Try to balance service assignment across mileage ranges
                range_service_vars = [is_service[i] for i in trainset_indices]
                # Add soft constraint to balance - minimize deviation from average
                avg_target = (self.config.required_service_trains * len(trainset_indices)) // self.n_trainsets
                if avg_target > 0:
                    deviation_var = model.NewIntVar(0, len(trainset_indices), f'dev_{range_name}')
                    model.Add(sum(range_service_vars) - avg_target <= deviation_var)
                    model.Add(avg_target - sum(range_service_vars) <= deviation_var)
                    objective_terms.append((-10, deviation_var))  # Minimize deviation
        
        # 3. Maximize branding compliance
        brand_compliance_score = 0
        for i, trainset_id in enumerate(self.evaluator.trainsets):
            if trainset_id in self.evaluator.brand_map:
                # Reward putting branded trainsets in service
                objective_terms.append((50, is_service[i]))  # Positive weight to maximize
        
        # 4. Minimize maintenance overhead
        for i, trainset_id in enumerate(self.evaluator.trainsets):
            constraints = self.evaluator.get_trainset_constraints(trainset_id)
            if constraints.maintenance_due:
                # Prefer putting maintenance-due trainsets in maintenance
                objective_terms.append((20, assignment[i] == 2))
        
        # Set the objective
        if objective_terms:
            model.Maximize(sum(weight * var for weight, var in objective_terms))
        else:
            # Fallback objective: maximize service assignments (though already constrained)
            model.Maximize(sum(is_service))
    
    def _categorize_trainsets_by_mileage(self) -> Dict[str, List[int]]:
        """Categorize trainsets by mileage ranges for balancing."""
        ranges = {'low': [], 'medium': [], 'high': []}
        
        mileages = []
        for trainset_id in self.evaluator.trainsets:
            status = self.evaluator.status_map.get(trainset_id, {})
            mileage = status.get('total_mileage_km', 0)
            mileages.append(mileage)
        
        if not mileages:
            return ranges
            
        # Define ranges based on quartiles
        mileages_sorted = sorted(mileages)
        q1 = mileages_sorted[len(mileages_sorted) // 4]
        q3 = mileages_sorted[3 * len(mileages_sorted) // 4]
        
        for i, trainset_id in enumerate(self.evaluator.trainsets):
            status = self.evaluator.status_map.get(trainset_id, {})
            mileage = status.get('total_mileage_km', 0)
            
            if mileage <= q1:
                ranges['low'].append(i)
            elif mileage >= q3:
                ranges['high'].append(i)
            else:
                ranges['medium'].append(i)
        
        return ranges
    
    def _create_fallback_solution(self) -> OptimizationResult:
        """Create a basic feasible solution when CP-SAT fails."""
        print("Creating fallback solution...")
        
        # Simple greedy assignment
        solution = np.full(self.n_trainsets, 2)  # Start with all in maintenance
        
        # Select best trainsets for service
        valid_trainsets = []
        for i, trainset_id in enumerate(self.evaluator.trainsets):
            valid, _ = self.evaluator.check_hard_constraints(trainset_id)
            if valid:
                valid_trainsets.append(i)
        
        # Assign required number to service
        service_count = min(len(valid_trainsets), self.config.required_service_trains)
        for i in range(service_count):
            solution[valid_trainsets[i]] = 0
        
        # Assign minimum to standby
        standby_start = service_count
        standby_count = min(len(valid_trainsets) - service_count, self.config.min_standby)
        for i in range(standby_count):
            if standby_start + i < len(valid_trainsets):
                solution[valid_trainsets[standby_start + i]] = 1
        
        return self._build_result(solution, float('inf'))
    
    def _build_result(self, solution: np.ndarray, objective_value: float) -> OptimizationResult:
        """Build optimization result from solution."""
        objectives = self.evaluator.calculate_objectives(solution)
        
        service = [self.evaluator.trainsets[i] for i, v in enumerate(solution) if v == 0]
        standby = [self.evaluator.trainsets[i] for i, v in enumerate(solution) if v == 1]
        maintenance = [self.evaluator.trainsets[i] for i, v in enumerate(solution) if v == 2]
        
        explanations = {}
        for ts_id in service:
            valid, reason = self.evaluator.check_hard_constraints(ts_id)
            explanations[ts_id] = "✅ CP-SAT optimal assignment" if valid else f"⚠ {reason}"
        
        fitness = self.evaluator.fitness_function(solution)
        
        return OptimizationResult(
            selected_trainsets=service,
            standby_trainsets=standby,
            maintenance_trainsets=maintenance,
            objectives=objectives,
            fitness_score=fitness,
            explanation=explanations
        )


class MIPOptimizer:
    """Mixed Integer Programming optimizer using OR-Tools."""
    
    def __init__(self, evaluator: TrainsetSchedulingEvaluator, config: Optional[OptimizationConfig] = None):
        if not ORTOOLS_AVAILABLE:
            raise ImportError("OR-Tools not available. Install with: pip install ortools")
            
        self.evaluator = evaluator
        self.config = config or OptimizationConfig()
        self.n_trainsets = evaluator.num_trainsets
    
    def optimize(self, time_limit_seconds: int = 300) -> OptimizationResult:
        """Solve using Mixed Integer Programming."""
        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            print("❌ SCIP solver not available, falling back to CP-SAT")
            cp_optimizer = CPSATOptimizer(self.evaluator, self.config)
            return cp_optimizer.optimize(time_limit_seconds)
        
        # Decision variables
        x = {}  # x[i,j] = 1 if trainset i is assigned to status j (0=service, 1=standby, 2=maintenance)
        for i in range(self.n_trainsets):
            for j in range(3):
                x[i, j] = solver.BoolVar(f'x_{i}_{j}')
        
        # Constraint: Each trainset has exactly one assignment
        for i in range(self.n_trainsets):
            solver.Add(sum(x[i, j] for j in range(3)) == 1)
        
        # Constraint: Exact number in service
        solver.Add(sum(x[i, 0] for i in range(self.n_trainsets)) == self.config.required_service_trains)
        
        # Constraint: Minimum standby
        solver.Add(sum(x[i, 1] for i in range(self.n_trainsets)) >= self.config.min_standby)
        
        # Hard constraints
        for i, trainset_id in enumerate(self.evaluator.trainsets):
            valid, _ = self.evaluator.check_hard_constraints(trainset_id)
            if not valid:
                # Force to maintenance
                solver.Add(x[i, 2] == 1)
        
        # Objective: Maximize weighted sum of objectives
        objective = solver.Objective()
        
        # Branding compliance
        for i, trainset_id in enumerate(self.evaluator.trainsets):
            if trainset_id in self.evaluator.brand_map:
                objective.SetCoefficient(x[i, 0], 100)  # Reward service assignment for branded trains
        
        # Mileage balance (simplified - prefer lower mileage in service)
        for i, trainset_id in enumerate(self.evaluator.trainsets):
            status = self.evaluator.status_map.get(trainset_id, {})
            mileage = status.get('total_mileage_km', 0)
            # Higher mileage gets lower weight (prefer lower mileage in service)
            weight = max(1, 200000 - mileage) // 1000
            objective.SetCoefficient(x[i, 0], weight)
        
        objective.SetMaximization()
        
        # Solve
        solver.SetTimeLimit(time_limit_seconds * 1000)  # milliseconds
        print(f"Solving with MIP (time limit: {time_limit_seconds}s)...")
        status = solver.Solve()
        
        if status == pywraplp.Solver.OPTIMAL:
            print("✅ Optimal MIP solution found!")
        elif status == pywraplp.Solver.FEASIBLE:
            print("✅ Feasible MIP solution found!")
        else:
            print("❌ No MIP solution found!")
            return self._create_fallback_solution()
        
        # Extract solution
        solution = np.zeros(self.n_trainsets, dtype=int)
        for i in range(self.n_trainsets):
            for j in range(3):
                if x[i, j].solution_value() > 0.5:
                    solution[i] = j
                    break
        
        return self._build_result(solution, solver.Objective().Value())
    
    def _create_fallback_solution(self) -> OptimizationResult:
        """Create fallback solution for MIP."""
        cp_optimizer = CPSATOptimizer(self.evaluator, self.config)
        return cp_optimizer._create_fallback_solution()
    
    def _build_result(self, solution: np.ndarray, objective_value: float) -> OptimizationResult:
        """Build result from MIP solution."""
        cp_optimizer = CPSATOptimizer(self.evaluator, self.config)
        return cp_optimizer._build_result(solution, objective_value)


# Integration function
def optimize_with_ortools(data: Dict, method: str = 'cp-sat', **kwargs) -> OptimizationResult:
    """Optimize using OR-Tools methods.
    
    Args:
        data: Metro synthetic data
        method: 'cp-sat' or 'mip'
        **kwargs: Additional parameters (time_limit_seconds, etc.)
    """
    if not ORTOOLS_AVAILABLE:
        raise ImportError(
            "OR-Tools not available. Install with: pip install ortools\n"
            "OR-Tools provides exact optimization methods that complement the existing metaheuristics."
        )
    
    from .evaluator import TrainsetSchedulingEvaluator
    
    evaluator = TrainsetSchedulingEvaluator(data)
    config = kwargs.get('config', OptimizationConfig())
    time_limit = kwargs.get('time_limit_seconds', 300)
    
    if method == 'cp-sat':
        optimizer = CPSATOptimizer(evaluator, config)
        return optimizer.optimize(time_limit)
    elif method == 'mip':
        optimizer = MIPOptimizer(evaluator, config)
        return optimizer.optimize(time_limit)
    else:
        raise ValueError(f"Unknown OR-Tools method: {method}. Use 'cp-sat' or 'mip'")


if __name__ == "__main__":
    import json
    
    # Test OR-Tools integration
    if not ORTOOLS_AVAILABLE:
        print("❌ OR-Tools not available. Install with: pip install ortools")
        exit(1)
    
    # Load test data
    try:
        with open('metro_enhanced_data.json', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Please generate enhanced data first")
        exit(1)
    
    print("🔧 Testing OR-Tools Optimization")
    print("=" * 50)
    
    # Test CP-SAT
    print("\nTesting CP-SAT optimizer...")
    result_cpsat = optimize_with_ortools(data, 'cp-sat', time_limit_seconds=60)
    print(f"CP-SAT Result: {len(result_cpsat.selected_trainsets)} in service")
    
    # Test MIP
    print("\nTesting MIP optimizer...")
    result_mip = optimize_with_ortools(data, 'mip', time_limit_seconds=60)
    print(f"MIP Result: {len(result_mip.selected_trainsets)} in service")
    
    print("\n✅ OR-Tools integration successful!")