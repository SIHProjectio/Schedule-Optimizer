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


def optimize_trainset_schedule_main(data: Dict, method: str = 'ga') -> OptimizationResult:
    """
    Main optimization function with original signature for backward compatibility.
    
    Args:
        data: Synthetic metro data dictionary
        method: 'ga' for Genetic Algorithm, 'cmaes' for CMA-ES, 'pso' for PSO, 'sa' for SA
    
    Returns:
        OptimizationResult containing the optimized schedule
    """
    return optimize_trainset_schedule(data, method)


def compare_all_methods(data: Dict) -> Dict[str, OptimizationResult]:
    """
    Compare all available optimization methods.
    
    Args:
        data: Synthetic metro data dictionary
    
    Returns:
        Dictionary mapping method names to OptimizationResults
    """
    return compare_optimization_methods(data)


# Usage example with the new structure
if __name__ == "__main__":
    # Load synthetic data
    try:
        with open('metro_synthetic_data.json', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: metro_synthetic_data.json not found.")
        print("Please run the synthetic data generator first:")
        print("python DataService/synthetic_base.py")
        exit(1)
    
    print("Metro Trainset Scheduling Optimization")
    print("=" * 50)
    
    # Option 1: Run single optimization
    print("\n1. Running Genetic Algorithm optimization...")
    result_ga = optimize_trainset_schedule_main(data, method='ga')
    
    # Option 2: Compare all methods
    print("\n2. Comparing all optimization methods...")
    all_results = compare_all_methods(data)
    
    print("\n✅ Optimization complete!")
    print("\nNew modular structure provides:")
    print("- Better code organization")
    print("- More optimization algorithms (GA, CMA-ES, PSO, SA)")
    print("- Improved error handling")
    print("- Configurable parameters")
    print("- Detailed result analysis")
    
    print("\nExample usage of new interface:")
    print("from greedyOptim import optimize_trainset_schedule, OptimizationConfig")
    print("config = OptimizationConfig(required_service_trains=20, min_standby=3)")
    print("result = optimize_trainset_schedule(data, 'pso', config=config)")