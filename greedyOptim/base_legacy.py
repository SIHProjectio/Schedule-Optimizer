"""
Legacy base.py module - maintained for backward compatibility.
New code should use the modular structure in the greedyOptim package.

This file provides the same interface as the original base.py but uses
the new modular, improved implementation under the hood.
"""
import json
import warnings
from typing import Dict

# Import from new modular structure
from .scheduler import (
    optimize_trainset_schedule,
    compare_optimization_methods,
    TrainsetSchedulingOptimizer as NewOptimizer
)
from .models import OptimizationResult, OptimizationConfig
from .evaluator import TrainsetSchedulingEvaluator


# Legacy class names for backward compatibility
class TrainsetSchedulingOptimizer:
    """Legacy optimizer class - use new modular structure instead."""
    
    def __init__(self, data: Dict):
        warnings.warn(
            "Legacy TrainsetSchedulingOptimizer is deprecated. Use the new modular structure.",
            DeprecationWarning,
            stacklevel=2
        )
        self.data = data
        self.evaluator = TrainsetSchedulingEvaluator(data)
        self.trainsets = self.evaluator.trainsets
        self.num_trainsets = self.evaluator.num_trainsets
        self.required_service_trains = 20
        self.min_standby = 2
    
    def check_hard_constraints(self, trainset_id: str):
        """Check hard constraints for a trainset."""
        return self.evaluator.check_hard_constraints(trainset_id)
    
    def calculate_objectives(self, solution):
        """Calculate objectives for a solution."""
        return self.evaluator.calculate_objectives(solution)
    
    def fitness_function(self, solution):
        """Calculate fitness for a solution."""
        return self.evaluator.fitness_function(solution)


class GeneticAlgorithmOptimizer:
    """Legacy GA optimizer - use new version instead."""
    
    def __init__(self, evaluator, population_size=100, generations=200):
        warnings.warn(
            "Legacy GeneticAlgorithmOptimizer is deprecated. Use genetic_algorithm.GeneticAlgorithmOptimizer instead.",
            DeprecationWarning,
            stacklevel=2
        )
        from .genetic_algorithm import GeneticAlgorithmOptimizer as NewGA
        
        config = OptimizationConfig(
            population_size=population_size,
            generations=generations
        )
        self.new_optimizer = NewGA(evaluator, config)
    
    def optimize(self):
        """Run optimization."""
        return self.new_optimizer.optimize()


class CMAESOptimizer:
    """Legacy CMA-ES optimizer - use new version instead."""
    
    def __init__(self, evaluator, population_size=50):
        warnings.warn(
            "Legacy CMAESOptimizer is deprecated. Use advanced_optimizers.CMAESOptimizer instead.",
            DeprecationWarning,
            stacklevel=2
        )
        from .advanced_optimizers import CMAESOptimizer as NewCMAES
        
        config = OptimizationConfig(population_size=population_size)
        self.new_optimizer = NewCMAES(evaluator, config)
    
    def optimize(self, generations=150):
        """Run optimization."""
        return self.new_optimizer.optimize(generations)


def optimize_trainset_schedule_legacy(data: Dict, method: str = 'ga') -> OptimizationResult:
    """Legacy function - use optimize_trainset_schedule instead."""
    warnings.warn(
        "This function is deprecated. Use optimize_trainset_schedule instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return optimize_trainset_schedule(data, method)


# Main function with the original signature for compatibility
def optimize_trainset_schedule_original(data: Dict, method: str = 'ga') -> OptimizationResult:
    """
    Main optimization function with original signature.
    
    Args:
        data: Synthetic metro data dictionary
        method: 'ga' for Genetic Algorithm, 'cmaes' for CMA-ES
    
    Returns:
        OptimizationResult
    """
    return optimize_trainset_schedule(data, method)


# Usage example (original style)
if __name__ == "__main__":
    # Load synthetic data
    try:
        with open('metro_synthetic_data.json', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: metro_synthetic_data.json not found. Please generate synthetic data first.")
        exit(1)
    
    # Run optimization with Genetic Algorithm (original interface)
    result_ga = optimize_trainset_schedule_original(data, method='ga')
    
    # Run optimization with CMA-ES (original interface)  
    result_cmaes = optimize_trainset_schedule_original(data, method='cmaes')
    
    print("\\nLegacy interface still works, but consider using the new modular structure!")
    print("Example:")
    print("from greedyOptim import optimize_trainset_schedule, OptimizationConfig")
    print("config = OptimizationConfig(required_service_trains=20)")
    print("result = optimize_trainset_schedule(data, 'pso', config=config)")