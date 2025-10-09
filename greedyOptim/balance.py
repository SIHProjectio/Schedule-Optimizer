"""
Load balancing for trainset optimization.
Simple interface for OR-Tools integration.
"""

from .scheduler import optimize_trainset_schedule
from .models import OptimizationConfig

# Check for OR-Tools availability
try:
    from .ortools_optimizers import optimize_with_ortools, check_ortools_availability
    ORTOOLS_AVAILABLE = check_ortools_availability()
except ImportError:
    ORTOOLS_AVAILABLE = False
    optimize_with_ortools = None


def balance_trainset_load(data, method='cp-sat', time_limit=300):
    """Balance trainset load using optimization methods."""
    if ORTOOLS_AVAILABLE and method in ['cp-sat', 'mip'] and optimize_with_ortools is not None:
        config = OptimizationConfig(required_service_trains=20, min_standby=3)
        return optimize_with_ortools(data, method, config=config, time_limit_seconds=time_limit)
    else:
        print("Using genetic algorithm fallback")
        config = OptimizationConfig(required_service_trains=20, min_standby=3)
        return optimize_trainset_schedule(data, 'ga', config)


def compare_exact_vs_metaheuristic(data):
    """Compare optimization methods."""
    results = {}
    
    # Try OR-Tools methods
    if ORTOOLS_AVAILABLE:
        for method in ['cp-sat', 'mip']:
            try:
                results[method] = balance_trainset_load(data, method, 120)
                print(f"✅ {method.upper()} completed")
            except Exception as e:
                print(f"❌ {method.upper()} failed: {e}")
    
    # Try metaheuristic methods
    config = OptimizationConfig(generations=50, population_size=30)
    for method in ['ga', 'pso']:
        try:
            results[method] = optimize_trainset_schedule(data, method, config)
            print(f"✅ {method.upper()} completed")
        except Exception as e:
            print(f"❌ {method.upper()} failed: {e}")
    
    return results


def get_ortools_status():
    """Get OR-Tools status."""
    return {
        'available': ORTOOLS_AVAILABLE,
        'methods': ['cp-sat', 'mip'] if ORTOOLS_AVAILABLE else []
    }
