"""
Test Constraint Satisfaction Benchmark
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from DataService.generators.enhanced_generator import EnhancedMetroDataGenerator
from greedyOptim.scheduling.scheduler import TrainsetSchedulingOptimizer
from greedyOptim.core.models import OptimizationConfig
from greedyOptim.scheduling.service_blocks import create_service_blocks_for_schedule
from benchmarks.constraint_satisfaction import run_constraint_benchmark


def create_schedule_with_service_blocks(result, method_name):
    """Create a schedule with service blocks from optimization result."""
    service_blocks_map = create_service_blocks_for_schedule(result.selected_trainsets)
    
    trainsets = []
    
    # Service trains with blocks
    for ts_id in result.selected_trainsets:
        trainsets.append({
            'trainset_id': ts_id,
            'status': 'REVENUE_SERVICE',
            'service_blocks': service_blocks_map.get(ts_id, []),
            'daily_km_allocation': 350,
            'cumulative_km': 50000
        })
    
    # Standby trains
    for ts_id in result.standby_trainsets:
        trainsets.append({
            'trainset_id': ts_id,
            'status': 'STANDBY',
            'service_blocks': []
        })
    
    # Maintenance trains
    for ts_id in result.maintenance_trainsets:
        trainsets.append({
            'trainset_id': ts_id,
            'status': 'MAINTENANCE',
            'service_blocks': []
        })
    
    return {
        'method': method_name,
        'trainsets': trainsets
    }


def main():
    print("=" * 80)
    print("CONSTRAINT SATISFACTION BENCHMARK TEST")
    print("=" * 80)
    print()
    
    # Generate data
    print("Generating synthetic metro data...")
    generator = EnhancedMetroDataGenerator(num_trainsets=25, seed=42)
    data = generator.generate_complete_enhanced_dataset()
    print(f"Generated data for {len(data['trainset_status'])} trainsets")
    print()
    
    # Create optimizer configuration
    config = OptimizationConfig(
        required_service_trains=15,
        min_standby=2,
        population_size=50,
        generations=100
    )
    
    # Generate schedules using different methods
    methods = ['ga', 'pso', 'sa', 'cmaes', 'nsga2', 'adaptive', 'ensemble']
    schedules = []
    
    print(f"Generating {len(methods)} schedules using different optimization methods...")
    print()
    
    for method in methods:
        print(f"Optimizing with {method.upper()}...")
        optimizer = TrainsetSchedulingOptimizer(data, config)
        result = optimizer.optimize(method=method)
        
        # Create schedule with service blocks
        schedule = create_schedule_with_service_blocks(result, method)
        schedules.append(schedule)
        
        print(f"  Service trains: {len(result.selected_trainsets)}")
        print(f"  Fitness score: {result.fitness_score:.2f}")
        print()
    
    # Run constraint satisfaction benchmark
    print("=" * 80)
    print("RUNNING CONSTRAINT SATISFACTION BENCHMARK")
    print("=" * 80)
    print()
    
    results = run_constraint_benchmark(
        schedules=schedules,
        data=data,
        output_file="constraint_satisfaction_results.json",
        verbose=True
    )
    
    print()
    print("=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
    print(f"Results saved to: constraint_satisfaction_results.json")
    print(f"Overall Constraint Score: {results['aggregate_metrics']['avg_overall_score']:.2f}/100")


if __name__ == "__main__":
    main()
