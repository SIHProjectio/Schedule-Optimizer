"""
Main optimization interface for trainset scheduling.
Provides a unified API for different optimization algorithms.
"""
import json
from typing import Dict, Optional, List

from .models import OptimizationResult, OptimizationConfig
from .evaluator import TrainsetSchedulingEvaluator
from .genetic_algorithm import GeneticAlgorithmOptimizer
from .advanced_optimizers import CMAESOptimizer, ParticleSwarmOptimizer, SimulatedAnnealingOptimizer
from .hybrid_optimizers import (
    MultiObjectiveOptimizer, AdaptiveOptimizer, EnsembleOptimizer,
    HyperParameterOptimizer, optimize_with_hybrid_methods
)

# Optional OR-Tools import
try:
    from .ortools_optimizers import optimize_with_ortools, check_ortools_availability
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False


class TrainsetSchedulingOptimizer:
    """Main interface for trainset scheduling optimization."""
    
    AVAILABLE_METHODS = {
        'ga': 'Genetic Algorithm',
        'cmaes': 'CMA-ES (Covariance Matrix Adaptation)',
        'pso': 'Particle Swarm Optimization',
        'sa': 'Simulated Annealing',
        'nsga2': 'NSGA-II Multi-Objective',
        'adaptive': 'Adaptive Algorithm Selection',
        'ensemble': 'Ensemble (Parallel)',
        'auto-tune': 'Auto-tuned Genetic Algorithm',
        'cp-sat': 'OR-Tools CP-SAT (Constraint Programming)',
        'mip': 'OR-Tools MIP (Mixed Integer Programming)'
    }
    
    def __init__(self, data: Dict, config: Optional[OptimizationConfig] = None):
        """Initialize optimizer with metro data."""
        self.data = data
        self.config = config or OptimizationConfig()
        self.evaluator = TrainsetSchedulingEvaluator(data, self.config)
        
        print(f"Initialized optimizer with {self.evaluator.num_trainsets} trainsets")
        print(f"Required in service: {self.config.required_service_trains}")
        print(f"Minimum standby: {self.config.min_standby}")
    
    def optimize(self, method: str = 'ga', **kwargs) -> OptimizationResult:
        """Run optimization with specified method.
        
        Args:
            method: Optimization method ('ga', 'cmaes', 'pso', 'sa')
            **kwargs: Method-specific parameters
            
        Returns:
            OptimizationResult containing the solution
        """
        if method not in self.AVAILABLE_METHODS:
            raise ValueError(f"Unknown method: {method}. Available: {list(self.AVAILABLE_METHODS.keys())}")
        
        print(f"\\nStarting optimization with {self.AVAILABLE_METHODS[method]}...")
        
        try:
            if method == 'ga':
                optimizer = GeneticAlgorithmOptimizer(self.evaluator, self.config)
                result = optimizer.optimize()
            elif method == 'cmaes':
                optimizer = CMAESOptimizer(self.evaluator, self.config)
                generations = kwargs.get('generations', 150)
                result = optimizer.optimize(generations)
            elif method == 'pso':
                optimizer = ParticleSwarmOptimizer(self.evaluator, self.config)
                generations = kwargs.get('generations', 200)
                result = optimizer.optimize(generations)
            elif method == 'sa':
                optimizer = SimulatedAnnealingOptimizer(self.evaluator, self.config)
                max_iterations = kwargs.get('max_iterations', 10000)
                result = optimizer.optimize(max_iterations)
            elif method == 'nsga2':
                optimizer = MultiObjectiveOptimizer(self.evaluator, self.config)
                result = optimizer.optimize()
            elif method == 'adaptive':
                optimizer = AdaptiveOptimizer(self.evaluator, self.config)
                max_iterations = kwargs.get('max_iterations', 5)
                result = optimizer.optimize(max_iterations)
            elif method == 'ensemble':
                optimizer = EnsembleOptimizer(self.evaluator, self.config)
                result = optimizer.optimize()
            elif method == 'auto-tune':
                tuner = HyperParameterOptimizer(self.evaluator)
                trials = kwargs.get('trials', 10)
                best_config = tuner.optimize_ga_params(trials)
                optimizer = GeneticAlgorithmOptimizer(self.evaluator, best_config)
                result = optimizer.optimize()
            elif method in ['cp-sat', 'mip']:
                if not ORTOOLS_AVAILABLE:
                    raise ImportError(f"OR-Tools not available for {method}. Install with: pip install ortools")
                from .ortools_optimizers import optimize_with_ortools
                time_limit = kwargs.get('time_limit_seconds', 300)
                result = optimize_with_ortools(self.data, method, config=self.config, time_limit_seconds=time_limit)
            else:
                raise ValueError(f"Method {method} not implemented")
            
            self._print_results(result, method)
            return result
            
        except Exception as e:
            print(f"Optimization failed: {e}")
            raise
    
    def compare_methods(self, methods: Optional[List[str]] = None, **kwargs) -> Dict[str, OptimizationResult]:
        """Compare multiple optimization methods.
        
        Args:
            methods: List of methods to compare (default: all available)
            **kwargs: Method-specific parameters
            
        Returns:
            Dictionary mapping method names to results
        """
        if methods is None:
            methods = list(self.AVAILABLE_METHODS.keys())
        
        results = {}
        print(f"\\nComparing optimization methods: {methods}")
        print("=" * 60)
        
        for method in methods:
            try:
                print(f"\\nRunning {self.AVAILABLE_METHODS[method]}...")
                result = self.optimize(method, **kwargs)
                results[method] = result
                print(f"Completed {method}: Fitness = {result.fitness_score:.2f}")
            except Exception as e:
                print(f"Failed {method}: {e}")
                results[method] = None
        
        # Print comparison summary
        self._print_comparison(results)
        return results
    
    def _print_results(self, result: OptimizationResult, method: str):
        """Print detailed optimization results."""
        print("\\n" + "="*60)
        print(f"OPTIMIZATION RESULTS - {self.AVAILABLE_METHODS[method].upper()}")
        print("="*60)
        
        print(f"\\nService Trainsets ({len(result.selected_trainsets)}):")
        for i, ts in enumerate(result.selected_trainsets[:5]):
            status = result.explanation.get(ts, 'N/A')
            print(f"  {ts}: {status}")
        if len(result.selected_trainsets) > 5:
            print(f"  ... and {len(result.selected_trainsets) - 5} more")
        
        print(f"\\nStandby Trainsets ({len(result.standby_trainsets)}):")
        for ts in result.standby_trainsets:
            print(f"  {ts}")
        
        print(f"\\nMaintenance Trainsets ({len(result.maintenance_trainsets)}):")
        for ts in result.maintenance_trainsets[:5]:
            print(f"  {ts}")
        if len(result.maintenance_trainsets) > 5:
            print(f"  ... and {len(result.maintenance_trainsets) - 5} more")
        
        print(f"\\nObjective Scores:")
        for key, value in result.objectives.items():
            print(f"  {key.replace('_', ' ').title()}: {value:.2f}")
        
        print(f"\\nOverall Fitness Score: {result.fitness_score:.2f}")
        
        # Constraint violations summary
        violations = sum(1 for exp in result.explanation.values() if '⚠' in exp)
        if violations > 0:
            print(f"\\n⚠ Warning: {violations} service trainsets have constraint violations")
    
    def _print_comparison(self, results: Dict[str, OptimizationResult]):
        """Print comparison summary of different methods."""
        print("\\n" + "="*60)
        print("METHOD COMPARISON SUMMARY")
        print("="*60)
        
        valid_results = {k: v for k, v in results.items() if v is not None}
        
        if not valid_results:
            print("No valid results to compare")
            return
        
        # Sort by fitness score (lower is better)
        sorted_results = sorted(valid_results.items(), key=lambda x: x[1].fitness_score)
        
        print(f"{'Method':<15} {'Fitness':<10} {'Service':<8} {'Standby':<8} {'Maint':<8} {'Violations':<10}")
        print("-" * 70)
        
        for method, result in sorted_results:
            violations = sum(1 for exp in result.explanation.values() if '⚠' in exp)
            print(f"{self.AVAILABLE_METHODS[method]:<15} "
                  f"{result.fitness_score:<10.2f} "
                  f"{len(result.selected_trainsets):<8} "
                  f"{len(result.standby_trainsets):<8} "
                  f"{len(result.maintenance_trainsets):<8} "
                  f"{violations:<10}")
        
        # Highlight best method
        best_method, best_result = sorted_results[0]
        print(f"\\n🏆 Best Method: {self.AVAILABLE_METHODS[best_method]} "
              f"(Fitness: {best_result.fitness_score:.2f})")
    
    def get_trainset_analysis(self) -> Dict:
        """Get detailed analysis of all trainsets."""
        analysis = {}
        
        for ts_id in self.evaluator.trainsets:
            constraints = self.evaluator.get_trainset_constraints(ts_id)
            valid, reason = self.evaluator.check_hard_constraints(ts_id)
            
            analysis[ts_id] = {
                'valid_for_service': valid,
                'constraint_reason': reason,
                'certificates_valid': constraints.has_valid_certificates,
                'critical_jobs': constraints.has_critical_jobs,
                'component_warnings': constraints.component_warnings,
                'maintenance_due': constraints.maintenance_due,
                'total_mileage_km': constraints.mileage,
                'days_since_service': constraints.last_service_days
            }
        
        return analysis


def optimize_trainset_schedule(data: Dict, 
                              method: str = 'ga', 
                              config: Optional[OptimizationConfig] = None,
                              **kwargs) -> OptimizationResult:
    """Convenience function for single optimization run.
    
    Args:
        data: Metro synthetic data dictionary
        method: Optimization method ('ga', 'cmaes', 'pso', 'sa')
        config: Optimization configuration
        **kwargs: Method-specific parameters
        
    Returns:
        OptimizationResult
    """
    optimizer = TrainsetSchedulingOptimizer(data, config)
    return optimizer.optimize(method, **kwargs)


def compare_optimization_methods(data: Dict,
                               methods: Optional[List[str]] = None,
                               config: Optional[OptimizationConfig] = None,
                               **kwargs) -> Dict[str, OptimizationResult]:
    """Convenience function for comparing multiple methods.
    
    Args:
        data: Metro synthetic data dictionary
        methods: List of methods to compare (default: all)
        config: Optimization configuration
        **kwargs: Method-specific parameters
        
    Returns:
        Dictionary mapping method names to results
    """
    optimizer = TrainsetSchedulingOptimizer(data, config)
    return optimizer.compare_methods(methods, **kwargs)


# Usage example
if __name__ == "__main__":
    # Load synthetic data
    try:
        with open('metro_synthetic_data.json', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: metro_synthetic_data.json not found. Please generate synthetic data first.")
        exit(1)
    
    # Custom configuration
    config = OptimizationConfig(
        required_service_trains=20,
        min_standby=3,
        population_size=50,
        generations=100
    )
    
    # Run single optimization
    result = optimize_trainset_schedule(data, method='ga', config=config)
    
    # Compare all methods
    # comparison = compare_optimization_methods(data, config=config)