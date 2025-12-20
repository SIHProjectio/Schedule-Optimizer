"""
Service Quality Benchmark Script
Runs comprehensive benchmarks for headway consistency, wait times, and coverage.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from DataService.enhanced_generator import EnhancedMetroDataGenerator
from greedyOptim.scheduler import TrainsetSchedulingOptimizer
from greedyOptim.models import OptimizationConfig
from greedyOptim.service_blocks import create_service_blocks_for_schedule

import time
import json
from typing import Dict, List
from datetime import datetime

from benchmarks.service_quality import ServiceQualityAnalyzer, ServiceQualityMetrics


def run_service_quality_benchmark(
    schedules: List[Dict],
    output_file: str = "service_quality_benchmark_results.json",
    output_dir: str = None,
    verbose: bool = True
) -> Dict:
    """Run service quality benchmark on multiple schedules.
    
    Args:
        schedules: List of schedule dictionaries to analyze
        output_file: Path to save results JSON
        verbose: Whether to print detailed output
        
    Returns:
        Dictionary with benchmark results
    """
    if verbose:
        print("=" * 80)
        print("SERVICE QUALITY BENCHMARK")
        print("=" * 80)
        print(f"Analyzing {len(schedules)} schedules...")
        print()
    
    analyzer = ServiceQualityAnalyzer()
    start_time = time.time()
    
    results = []
    
    # Analyze each schedule
    for i, schedule in enumerate(schedules, 1):
        if verbose:
            print(f"Schedule {i}/{len(schedules)}:")
        
        schedule_start = time.time()
        metrics = analyzer.analyze_schedule(schedule)
        analysis_time = (time.time() - schedule_start) * 1000  # ms
        
        result = {
            'schedule_id': i,
            'analysis_time_ms': round(analysis_time, 2),
            'metrics': {
                'real_world_applicability': {
                    'avg_speed_maintained': metrics.real_world_metrics.avg_speed_maintained,
                    'max_speed_respected': metrics.real_world_metrics.max_speed_respected,
                    'route_distance_covered': metrics.real_world_metrics.route_distance_covered,
                    'stations_serviced_count': metrics.real_world_metrics.stations_serviced_count,
                    'operational_hours_met': metrics.real_world_metrics.operational_hours_met,
                    'peak_headway_met': metrics.real_world_metrics.peak_headway_met,
                    'score': round(metrics.real_world_metrics.score, 2)
                },
                'headway_consistency': {
                    'peak_mean_minutes': round(metrics.peak_headway_mean, 2),
                    'peak_std_minutes': round(metrics.peak_headway_std, 2),
                    'peak_cv': round(metrics.peak_headway_coefficient_variation, 3),
                    'offpeak_mean_minutes': round(metrics.offpeak_headway_mean, 2),
                    'offpeak_std_minutes': round(metrics.offpeak_headway_std, 2),
                    'offpeak_cv': round(metrics.offpeak_headway_coefficient_variation, 3),
                    'score': round(metrics.headway_consistency_score, 2)
                },
                'wait_times': {
                    'avg_wait_peak_minutes': round(metrics.avg_wait_time_peak, 2),
                    'max_wait_peak_minutes': round(metrics.max_wait_time_peak, 2),
                    'avg_wait_offpeak_minutes': round(metrics.avg_wait_time_offpeak, 2),
                    'max_wait_offpeak_minutes': round(metrics.max_wait_time_offpeak, 2),
                    'reduction_vs_baseline_percent': round(metrics.wait_time_reduction_vs_baseline, 2),
                    'score': round(metrics.wait_time_score, 2)
                },
                'service_coverage': {
                    'operational_hours': round(metrics.operational_hours, 2),
                    'peak_hours_covered': round(metrics.peak_hours_covered, 2),
                    'offpeak_hours_covered': round(metrics.offpeak_hours_covered, 2),
                    'coverage_percent': round(metrics.service_coverage_percent, 2),
                    'peak_coverage_percent': round(metrics.peak_coverage_percent, 2),
                    'service_gaps': metrics.gaps_in_service,
                    'score': round(metrics.coverage_score, 2)
                },
                'overall_quality_score': round(metrics.overall_quality_score, 2)
            }
        }
        
        results.append(result)
        
        if verbose:
            print(f"  Real-World Applicability: {result['metrics']['real_world_applicability']['score']:.2f}/100")
            print(f"    Avg Speed Maintained: {'✓' if metrics.real_world_metrics.avg_speed_maintained else '✗'}")
            print(f"    Max Speed Respected: {'✓' if metrics.real_world_metrics.max_speed_respected else '✗'}")
            print(f"    Route Distance: {'✓' if metrics.real_world_metrics.route_distance_covered else '✗'}")
            print(f"    Stations Serviced: {metrics.real_world_metrics.stations_serviced_count}/22")
            print(f"    Operational Hours: {'✓' if metrics.real_world_metrics.operational_hours_met else '✗'}")
            print(f"    Peak Headway (5-7m): {'✓' if metrics.real_world_metrics.peak_headway_met else '✗'}")
            print(f"  Headway Consistency Score: {result['metrics']['headway_consistency']['score']:.2f}/100")
            print(f"    Peak: {metrics.peak_headway_mean:.1f}min ± {metrics.peak_headway_std:.1f}min (CV: {metrics.peak_headway_coefficient_variation:.3f})")
            print(f"    Off-Peak: {metrics.offpeak_headway_mean:.1f}min ± {metrics.offpeak_headway_std:.1f}min (CV: {metrics.offpeak_headway_coefficient_variation:.3f})")
            print(f"  Wait Time Score: {result['metrics']['wait_times']['score']:.2f}/100")
            print(f"    Peak avg wait: {metrics.avg_wait_time_peak:.1f}min (max: {metrics.max_wait_time_peak:.1f}min)")
            print(f"    Off-peak avg wait: {metrics.avg_wait_time_offpeak:.1f}min (max: {metrics.max_wait_time_offpeak:.1f}min)")
            print(f"    Improvement vs baseline: {metrics.wait_time_reduction_vs_baseline:.1f}%")
            print(f"  Coverage Score: {result['metrics']['service_coverage']['score']:.2f}/100")
            print(f"    Peak coverage: {metrics.peak_coverage_percent:.1f}%")
            print(f"    Overall coverage: {metrics.service_coverage_percent:.1f}%")
            print(f"    Service gaps: {metrics.gaps_in_service}")
            print(f"  OVERALL QUALITY: {metrics.overall_quality_score:.2f}/100")
            print(f"  Analysis time: {analysis_time:.2f}ms")
            print()
    
    total_time = time.time() - start_time
    
    # Calculate aggregate statistics
    aggregate = _calculate_aggregate_stats(results)
    
    # Prepare final output
    benchmark_results = {
        'benchmark_info': {
            'timestamp': datetime.now().isoformat(),
            'total_schedules': len(schedules),
            'total_time_seconds': round(total_time, 3),
            'avg_analysis_time_ms': round(aggregate['avg_analysis_time_ms'], 2)
        },
        'aggregate_metrics': aggregate,
        'individual_results': results
    }
    
    # Print summary
    if verbose:
        print("=" * 80)
        print("AGGREGATE RESULTS")
        print("=" * 80)
        print(f"Schedules analyzed: {len(schedules)}")
        print(f"Total benchmark time: {total_time:.2f}s")
        print()
        print("Average Scores:")
        print(f"  Real-World Applicability: {aggregate['avg_real_world_score']:.2f}/100")
        print(f"  Headway Consistency: {aggregate['avg_headway_score']:.2f}/100")
        print(f"  Wait Time Quality: {aggregate['avg_wait_score']:.2f}/100")
        print(f"  Service Coverage: {aggregate['avg_coverage_score']:.2f}/100")
        print(f"  Overall Quality: {aggregate['avg_overall_score']:.2f}/100")
        print()
        print("Best Performers:")
        print(f"  Best headway consistency: Schedule {aggregate['best_headway_schedule']} ({aggregate['best_headway_score']:.2f})")
        print(f"  Best wait times: Schedule {aggregate['best_wait_schedule']} ({aggregate['best_wait_score']:.2f})")
        print(f"  Best coverage: Schedule {aggregate['best_coverage_schedule']} ({aggregate['best_coverage_score']:.2f})")
        print(f"  Best overall: Schedule {aggregate['best_overall_schedule']} ({aggregate['best_overall_score']:.2f})")
        print()
        print("Service Quality Metrics:")
        print(f"  Avg peak headway: {aggregate['avg_peak_headway']:.2f} ± {aggregate['avg_peak_headway_std']:.2f} minutes")
        print(f"  Avg off-peak headway: {aggregate['avg_offpeak_headway']:.2f} ± {aggregate['avg_offpeak_headway_std']:.2f} minutes")
        print(f"  Avg peak wait time: {aggregate['avg_peak_wait']:.2f} minutes")
        print(f"  Avg wait time reduction: {aggregate['avg_wait_reduction']:.1f}%")
        print(f"  Avg service coverage: {aggregate['avg_coverage_percent']:.1f}%")
        print(f"  Avg service gaps: {aggregate['avg_service_gaps']:.1f}")
        print()
    
    # Save to file in benchmark_output/ at project root
    if output_dir is None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        output_dir = os.path.join(project_root, "benchmark_output")
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)
    
    with open(output_path, 'w') as f:
        json.dump(benchmark_results, f, indent=2)
    
    if verbose:
        print(f"Results saved to: {output_path}")
        print("=" * 80)
    
    return benchmark_results


def _calculate_aggregate_stats(results: List[Dict]) -> Dict:
    """Calculate aggregate statistics across all results."""
    if not results:
        return {}
    
    # Extract all metrics
    real_world_scores = [r['metrics']['real_world_applicability']['score'] for r in results]
    headway_scores = [r['metrics']['headway_consistency']['score'] for r in results]
    wait_scores = [r['metrics']['wait_times']['score'] for r in results]
    coverage_scores = [r['metrics']['service_coverage']['score'] for r in results]
    overall_scores = [r['metrics']['overall_quality_score'] for r in results]
    
    peak_headways = [r['metrics']['headway_consistency']['peak_mean_minutes'] for r in results]
    peak_headway_stds = [r['metrics']['headway_consistency']['peak_std_minutes'] for r in results]
    offpeak_headways = [r['metrics']['headway_consistency']['offpeak_mean_minutes'] for r in results]
    offpeak_headway_stds = [r['metrics']['headway_consistency']['offpeak_std_minutes'] for r in results]
    peak_waits = [r['metrics']['wait_times']['avg_wait_peak_minutes'] for r in results]
    wait_reductions = [r['metrics']['wait_times']['reduction_vs_baseline_percent'] for r in results]
    coverage_percents = [r['metrics']['service_coverage']['coverage_percent'] for r in results]
    service_gaps = [r['metrics']['service_coverage']['service_gaps'] for r in results]
    analysis_times = [r['analysis_time_ms'] for r in results]
    
    # Find best performers
    best_headway_idx = headway_scores.index(max(headway_scores))
    best_wait_idx = wait_scores.index(max(wait_scores))
    best_coverage_idx = coverage_scores.index(max(coverage_scores))
    best_overall_idx = overall_scores.index(max(overall_scores))
    
    return {
        'avg_real_world_score': round(sum(real_world_scores) / len(real_world_scores), 2),
        'avg_headway_score': round(sum(headway_scores) / len(headway_scores), 2),
        'avg_wait_score': round(sum(wait_scores) / len(wait_scores), 2),
        'avg_coverage_score': round(sum(coverage_scores) / len(coverage_scores), 2),
        'avg_overall_score': round(sum(overall_scores) / len(overall_scores), 2),
        
        'best_headway_schedule': best_headway_idx + 1,
        'best_headway_score': round(max(headway_scores), 2),
        'best_wait_schedule': best_wait_idx + 1,
        'best_wait_score': round(max(wait_scores), 2),
        'best_coverage_schedule': best_coverage_idx + 1,
        'best_coverage_score': round(max(coverage_scores), 2),
        'best_overall_schedule': best_overall_idx + 1,
        'best_overall_score': round(max(overall_scores), 2),
        
        'avg_peak_headway': round(sum(peak_headways) / len(peak_headways), 2),
        'avg_peak_headway_std': round(sum(peak_headway_stds) / len(peak_headway_stds), 2),
        'avg_offpeak_headway': round(sum(offpeak_headways) / len(offpeak_headways), 2),
        'avg_offpeak_headway_std': round(sum(offpeak_headway_stds) / len(offpeak_headway_stds), 2),
        'avg_peak_wait': round(sum(peak_waits) / len(peak_waits), 2),
        'avg_wait_reduction': round(sum(wait_reductions) / len(wait_reductions), 2),
        'avg_coverage_percent': round(sum(coverage_percents) / len(coverage_percents), 2),
        'avg_service_gaps': round(sum(service_gaps) / len(service_gaps), 2),
        'avg_analysis_time_ms': round(sum(analysis_times) / len(analysis_times), 2)
    }


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
    
    # Compute output path for display
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    output_dir = os.path.join(project_root, "benchmark_output")
    output_path = os.path.join(output_dir, "service_quality_benchmark_results.json")
    
    print(f"Results saved to: {output_path}")
    print(f"Overall Quality Score: {results['aggregate_metrics']['avg_overall_score']:.2f}/100")


if __name__ == "__main__":
    main()
