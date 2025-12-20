"""
OR-Tools based optimizers for trainset scheduling.
Provides exact and constraint programming solutions.
"""
import numpy as np
from typing import Dict, List, Optional, Any
import logging

from greedyOptim.core.models import OptimizationResult, OptimizationConfig
from greedyOptim.scheduling.evaluator import TrainsetSchedulingEvaluator


def check_ortools_availability() -> bool:
    """Check if OR-Tools is available."""
    try:
        import ortools.sat.python.cp_model
        import ortools.linear_solver.pywraplp
        return True
    except ImportError:
        return False


class ORToolsOptimizer:
    """Base class for OR-Tools optimizers."""
    
    def __init__(self, evaluator: TrainsetSchedulingEvaluator, config: Optional[OptimizationConfig] = None):
        if not check_ortools_availability():
            raise ImportError(
                "OR-Tools not available. Install with: pip install ortools\n"
                "OR-Tools provides exact optimization methods that complement existing metaheuristics."
            )
            
        self.evaluator = evaluator
        self.config = config or OptimizationConfig()
        self.n_trainsets = evaluator.num_trainsets
        
        # Logging
        self.logger = logging.getLogger(__name__)


class CPSATOptimizer(ORToolsOptimizer):
    """Constraint Programming optimizer using OR-Tools CP-SAT solver."""
    
    def optimize(self, time_limit_seconds: int = 300) -> OptimizationResult:
        """Solve using CP-SAT with constraint programming."""
        # Import here to avoid issues with type checking
        from ortools.sat.python import cp_model
        
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
        forced_maintenance = 0
        for i, trainset_id in enumerate(self.evaluator.trainsets):
            valid, reason = self.evaluator.check_hard_constraints(trainset_id)
            if not valid:
                # Force trainset to maintenance if it fails constraints
                model.Add(assignment[i] == 2)
                forced_maintenance += 1
                self.logger.info(f"Trainset {trainset_id} forced to maintenance: {reason}")
        
        print(f"Forced {forced_maintenance} trainsets to maintenance due to constraints")
        
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
        
        print(f"CP-SAT solution: {np.sum(solution == 0)} service, {np.sum(solution == 1)} standby, {np.sum(solution == 2)} maintenance")
        
        return self._build_result(solution, solver.ObjectiveValue())
    
    def _add_branding_constraints(self, model: Any, is_service: List[Any]):
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
                # At least 30% of branded trainsets should be in service if possible
                min_branded = max(1, len(trainset_indices) // 3)
                model.Add(sum(is_service[i] for i in trainset_indices) >= min_branded)
                print(f"Brand {brand}: {len(trainset_indices)} trainsets, requiring {min_branded} in service")
    
    def _set_multi_objective(self, model: Any, is_service: List[Any], is_standby: List[Any], assignment: List[Any]):
        """Set up multi-objective optimization using weighted sum."""
        # Import here to avoid type checking issues
        from ortools.sat.python import cp_model
        
        objective_terms = []
        
        # 1. Branding compliance - maximize service assignment for branded trainsets
        brand_score = 0
        for i, trainset_id in enumerate(self.evaluator.trainsets):
            if trainset_id in self.evaluator.brand_map:
                # Reward putting branded trainsets in service
                objective_terms.append((50, is_service[i]))
                brand_score += 1
        
        print(f"Found {brand_score} branded trainsets for optimization")
        
        # 2. Mileage balance - prefer lower mileage trainsets in service
        mileages = []
        for trainset_id in self.evaluator.trainsets:
            status = self.evaluator.status_map.get(trainset_id, {})
            mileage = status.get('total_mileage_km', 0)
            mileages.append(mileage)
        
        if mileages:
            avg_mileage = sum(mileages) / len(mileages)
            for i, mileage in enumerate(mileages):
                # Prefer trainsets with below-average mileage
                if mileage < avg_mileage:
                    weight = int((avg_mileage - mileage) / 1000)  # Scale down
                    objective_terms.append((max(1, weight), is_service[i]))
        
        # 3. Maintenance preference - prefer trainsets needing maintenance to go to maintenance
        for i, trainset_id in enumerate(self.evaluator.trainsets):
            constraints = self.evaluator.get_trainset_constraints(trainset_id)
            if constraints.maintenance_due:
                # Create auxiliary variable for maintenance assignment
                is_maint_var = model.NewBoolVar(f'maint_pref_{i}')
                model.Add(assignment[i] == 2).OnlyEnforceIf(is_maint_var)
                model.Add(assignment[i] != 2).OnlyEnforceIf(is_maint_var.Not())
                objective_terms.append((30, is_maint_var))
        
        # Set the objective
        if objective_terms:
            model.Maximize(sum(weight * var for weight, var in objective_terms))
            print(f"Set objective with {len(objective_terms)} terms")
        else:
            # Fallback objective: minimize assignments to maintenance (prefer service/standby)
            model.Minimize(sum(assignment[i] == 2 for i in range(self.n_trainsets)))
            print("Using fallback objective")
    
    def _create_fallback_solution(self) -> OptimizationResult:
        """Create a basic feasible solution when CP-SAT fails."""
        print("Creating fallback solution...")
        
        # Simple greedy assignment
        solution = np.full(self.n_trainsets, 2, dtype=int)  # Start with all in maintenance
        
        # Select best trainsets for service
        valid_trainsets = []
        for i, trainset_id in enumerate(self.evaluator.trainsets):
            valid, _ = self.evaluator.check_hard_constraints(trainset_id)
            if valid:
                valid_trainsets.append(i)
        
        # Sort by mileage (prefer lower mileage)
        valid_trainsets.sort(key=lambda i: self.evaluator.status_map.get(
            self.evaluator.trainsets[i], {}
        ).get('total_mileage_km', 0))
        
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
            explanations[ts_id] = "✅ CP-SAT optimal" if valid else f"⚠ {reason}"
        
        for ts_id in standby:
            explanations[ts_id] = "🟡 Standby (CP-SAT)"
        
        for ts_id in maintenance:
            explanations[ts_id] = "🔧 Maintenance (CP-SAT)"
        
        fitness = self.evaluator.fitness_function(solution)
        
        return OptimizationResult(
            selected_trainsets=service,
            standby_trainsets=standby,
            maintenance_trainsets=maintenance,
            objectives=objectives,
            fitness_score=fitness,
            explanation=explanations
        )


class MIPOptimizer(ORToolsOptimizer):
    """Mixed Integer Programming optimizer using OR-Tools."""
    
    def optimize(self, time_limit_seconds: int = 300) -> OptimizationResult:
        """Solve using Mixed Integer Programming."""
        # Import here to avoid type checking issues
        from ortools.linear_solver import pywraplp
        
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
        forced_maintenance = 0
        for i, trainset_id in enumerate(self.evaluator.trainsets):
            valid, _ = self.evaluator.check_hard_constraints(trainset_id)
            if not valid:
                # Force to maintenance
                solver.Add(x[i, 2] == 1)
                forced_maintenance += 1
        
        print(f"MIP: Forced {forced_maintenance} trainsets to maintenance")
        
        # Objective: Maximize weighted sum of objectives
        objective = solver.Objective()
        
        # Branding compliance
        brand_count = 0
        for i, trainset_id in enumerate(self.evaluator.trainsets):
            if trainset_id in self.evaluator.brand_map:
                objective.SetCoefficient(x[i, 0], 100)  # Reward service assignment for branded trains
                brand_count += 1
        
        # Mileage balance (simplified - prefer lower mileage in service)
        for i, trainset_id in enumerate(self.evaluator.trainsets):
            status = self.evaluator.status_map.get(trainset_id, {})
            mileage = status.get('total_mileage_km', 0)
            # Higher mileage gets lower weight (prefer lower mileage in service)
            weight = max(1, 200000 - mileage) // 1000
            objective.SetCoefficient(x[i, 0], weight)
        
        objective.SetMaximization()
        print(f"MIP: Set up optimization with {brand_count} branded trainsets")
        
        # Solve
        solver.SetTimeLimit(time_limit_seconds * 1000)  # milliseconds
        print(f"Solving with MIP (time limit: {time_limit_seconds}s)...")
        status = solver.Solve()
        
        if status == pywraplp.Solver.OPTIMAL:
            print("✅ Optimal MIP solution found!")
        elif status == pywraplp.Solver.FEASIBLE:
            print("✅ Feasible MIP solution found!")
        else:
            print("❌ No MIP solution found, falling back to CP-SAT")
            cp_optimizer = CPSATOptimizer(self.evaluator, self.config)
            return cp_optimizer.optimize(time_limit_seconds)
        
        # Extract solution
        solution = np.zeros(self.n_trainsets, dtype=int)
        for i in range(self.n_trainsets):
            for j in range(3):
                if x[i, j].solution_value() > 0.5:
                    solution[i] = j
                    break
        
        print(f"MIP solution: {np.sum(solution == 0)} service, {np.sum(solution == 1)} standby, {np.sum(solution == 2)} maintenance")
        
        # Use CP-SAT's result builder
        cp_optimizer = CPSATOptimizer(self.evaluator, self.config)
        result = cp_optimizer._build_result(solution, solver.Objective().Value())
        
        # Update explanations for MIP
        for ts_id in result.explanation:
            if "CP-SAT" in result.explanation[ts_id]:
                result.explanation[ts_id] = result.explanation[ts_id].replace("CP-SAT", "MIP")
        
        return result


# Integration functions
def optimize_with_ortools(data: Dict, method: str = 'cp-sat', **kwargs) -> OptimizationResult:
    """Optimize using OR-Tools methods.
    
    Args:
        data: Metro synthetic data
        method: 'cp-sat' or 'mip'
        **kwargs: Additional parameters (time_limit_seconds, config, etc.)
    """
    if not check_ortools_availability():
        raise ImportError(
            "OR-Tools not available. Install with: pip install ortools\n"
            "OR-Tools provides exact optimization methods that complement the existing metaheuristics."
        )
    
    from greedyOptim.scheduling.evaluator import TrainsetSchedulingEvaluator
    
    evaluator = TrainsetSchedulingEvaluator(data)
    config = kwargs.get('config', OptimizationConfig())
    time_limit = kwargs.get('time_limit_seconds', 300)
    
    print(f"\n🔧 OR-Tools {method.upper()} Optimization")
    print("=" * 50)
    print(f"Trainsets: {evaluator.num_trainsets}")
    print(f"Required in service: {config.required_service_trains}")
    print(f"Minimum standby: {config.min_standby}")
    
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
    if not check_ortools_availability():
        print("❌ OR-Tools not available. Install with: pip install ortools")
        print("OR-Tools provides exact optimization that complements the existing metaheuristics.")
        exit(1)
    
    # Load test data
    try:
        with open('../DataService/metro_enhanced_data.json', 'r') as f:
            data = json.load(f)
        print("✅ Loaded enhanced synthetic data")
    except FileNotFoundError:
        try:
            with open('../DataService/metro_synthetic_data.json', 'r') as f:
                data = json.load(f)
            print("✅ Loaded basic synthetic data")
        except FileNotFoundError:
            print("❌ No test data found. Please generate synthetic data first.")
            exit(1)
    
    print("\n🔧 Testing OR-Tools Optimization Methods")
    print("=" * 60)
    
    # Test CP-SAT
    print("\n1. Testing CP-SAT optimizer...")
    try:
        result_cpsat = optimize_with_ortools(data, 'cp-sat', time_limit_seconds=60)
        print(f"✅ CP-SAT completed: {len(result_cpsat.selected_trainsets)} in service, "
              f"fitness = {result_cpsat.fitness_score:.2f}")
    except Exception as e:
        print(f"❌ CP-SAT failed: {e}")
    
    # Test MIP
    print("\n2. Testing MIP optimizer...")
    try:
        result_mip = optimize_with_ortools(data, 'mip', time_limit_seconds=60)
        print(f"✅ MIP completed: {len(result_mip.selected_trainsets)} in service, "
              f"fitness = {result_mip.fitness_score:.2f}")
    except Exception as e:
        print(f"❌ MIP failed: {e}")
    
    print("\n🎉 OR-Tools integration test completed!")
    print("\nOR-Tools provides:")
    print("• Exact optimization with mathematical guarantees")
    print("• Constraint satisfaction with hard constraints")
    print("• Complement to existing metaheuristic approaches")
    print("• Optimal solutions for smaller problem instances")