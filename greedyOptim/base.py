"""
Refactored base.py - now uses modular structure for better maintainability.

This module has been restructured into separate files:
- models.py: Data classes and configurations
- evaluator.py: Constraint checking and objective evaluation  
- genetic_algorithm.py: Genetic Algorithm implementation
- advanced_optimizers.py: CMA-ES, PSO, and Simulated Annealing
- scheduler.py: Main interface and comparison tools

For backward compatibility, the main functions are still available here.
"""
import json
from typing import Dict

# Import from new modular structure
from .scheduler import optimize_trainset_schedule, compare_optimization_methods
from .models import OptimizationResult, OptimizationConfig


# For backward compatibility, expose the main function with original signature
def optimize_trainset_schedule_main(data: Dict, method: str = 'ga') -> OptimizationResult:
    """Multi-objective optimizer for trainset scheduling using genetic algorithm.
    
    This is a backward compatibility wrapper around the new modular system.
    
    Args:
        data: Metro synthetic data dictionary
        method: Optimization method ('ga', 'cmaes', 'pso', 'sa', etc.)
        
    Returns:
        OptimizationResult with selected trainsets and performance metrics
    """
    # Use the new modular system
    config = OptimizationConfig()
    return optimize_trainset_schedule(data, method, config)


# Usage example
if __name__ == "__main__":
    # Load synthetic data
    try:
        with open('metro_synthetic_data.json', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Please generate synthetic data first")
        exit(1)
    
    # Run optimization with Genetic Algorithm (backward compatibility)
    result_ga = optimize_trainset_schedule_main(data, method='ga')
    
    print(f"\nOptimization completed!")
    print(f"Service trainsets: {len(result_ga.selected_trainsets)}")
    print(f"Standby trainsets: {len(result_ga.standby_trainsets)}")  
    print(f"Maintenance trainsets: {len(result_ga.maintenance_trainsets)}")
    print(f"Final fitness score: {result_ga.fitness_score:.2f}")
    
    # You can also use the new modular interface
    print(f"\nFor more advanced features, use:")
    print(f"from greedyOptim import optimize_trainset_schedule, compare_optimization_methods")
    print(f"from greedyOptim import optimize_with_hybrid_methods")
