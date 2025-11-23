"""
Test Service Quality Benchmark
Direct test of the service quality benchmarking system.
"""
import sys
import os

from DataService.enhanced_generator import EnhancedMetroDataGenerator
from greedyOptim.scheduler import TrainsetSchedulingOptimizer
from greedyOptim.models import OptimizationConfig
from greedyOptim.service_blocks import create_service_blocks_for_schedule
from benchmarks.service_quality import run_service_quality_benchmark


def create_schedule_with_service_blocks(result, method_name):
    """Create a schedule with service blocks from optimization result."""
    # Generate service blocks for selected trains
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
    print("SERVICE QUALITY BENCHMARK TEST")
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
    methods = ['ga', 'pso', 'sa']
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
    
    # Run service quality benchmark
    print("=" * 80)
    print("RUNNING SERVICE QUALITY BENCHMARK")
    print("=" * 80)
    print()
    
    results = run_service_quality_benchmark(
        schedules=schedules,
        output_file="service_quality_benchmark_results.json",
        verbose=True
    )
    
    print()
    print("=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
    print(f"Results saved to: service_quality_benchmark_results.json")
    print(f"Overall Quality Score: {results['aggregate_metrics']['avg_overall_score']:.2f}/100")


if __name__ == "__main__":
    main()
