#!/usr/bin/env python3
"""
Comprehensive benchmark for schedule generation performance.
Tests MetroScheduleOptimizer and greedyOptim methods across different fleet sizes.
"""
import time
import statistics
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import DataService components
from DataService.schedule_optimizer import MetroScheduleOptimizer
from DataService.metro_data_generator import MetroDataGenerator
from DataService.enhanced_generator import EnhancedMetroDataGenerator

# Import greedyOptim components
from greedyOptim.scheduler import optimize_trainset_schedule
from greedyOptim.models import OptimizationConfig


class SchedulePerformanceBenchmark:
    """Benchmark schedule generation performance"""
    
    def __init__(self):
        self.results = {
            "benchmark_info": {
                "date": datetime.now().isoformat(),
                "description": "Metro Schedule Generation Performance Analysis",
                "test_type": "Schedule Generation Time & Computational Efficiency"
            },
            "configurations": [],
            "detailed_results": [],
            "summary": {}
        }
    
    def benchmark_schedule_generation(
        self,
        num_trains: int,
        num_stations: int = 22,
        num_runs: int = 3
    ) -> Dict[str, Any]:
        """Benchmark the MetroScheduleOptimizer"""
        print(f"\n{'='*70}")
        print(f"Benchmarking Schedule Generation")
        print(f"Fleet Size: {num_trains} trains | Stations: {num_stations}")
        print(f"{'='*70}")
        
        run_times = []
        success_count = 0
        schedule_stats = []
        
        for run in range(num_runs):
            print(f"\nRun {run + 1}/{num_runs}...", end=" ")
            
            try:
                # Generate data
                generator = MetroDataGenerator(num_trains=num_trains)
                
                route = generator.generate_route(route_name="Aluva-Pettah Line")
                
                train_health = generator.generate_train_health_statuses()
                
                # Time the schedule generation
                start_time = time.perf_counter()
                
                optimizer = MetroScheduleOptimizer(
                    date="2025-11-06",
                    num_trains=num_trains,
                    route=route,
                    train_health=train_health
                )
                
                schedule = optimizer.optimize_schedule()
                
                end_time = time.perf_counter()
                elapsed = end_time - start_time
                
                run_times.append(elapsed)
                success_count += 1
                
                # Collect schedule statistics
                stats = {
                    "num_trainsets": len(schedule.trainsets),
                    "num_in_service": len([t for t in schedule.trainsets if t.status == "IN_SERVICE"]),
                    "num_standby": len([t for t in schedule.trainsets if t.status == "STANDBY"]),
                    "num_maintenance": len([t for t in schedule.trainsets if t.status == "UNDER_MAINTENANCE"]),
                    "total_service_blocks": sum(len(t.service_blocks) for t in schedule.trainsets),
                }
                schedule_stats.append(stats)
                
                print(f"✓ {elapsed:.4f}s | In Service: {stats['num_in_service']}/{stats['num_trainsets']}")
                
            except Exception as e:
                print(f"✗ Failed: {str(e)[:60]}")
                continue
        
        # Calculate statistics
        if run_times:
            avg_stats = {}
            if schedule_stats:
                for key in schedule_stats[0].keys():
                    values = [s[key] for s in schedule_stats]
                    avg_stats[key] = {
                        "mean": statistics.mean(values),
                        "min": min(values),
                        "max": max(values)
                    }
            
            result = {
                "optimizer": "MetroScheduleOptimizer",
                "fleet_size": num_trains,
                "num_stations": num_stations,
                "num_runs": num_runs,
                "successful_runs": success_count,
                "success_rate": f"{(success_count/num_runs)*100:.1f}%",
                "execution_time": {
                    "min_seconds": min(run_times),
                    "max_seconds": max(run_times),
                    "mean_seconds": statistics.mean(run_times),
                    "median_seconds": statistics.median(run_times),
                    "stdev_seconds": statistics.stdev(run_times) if len(run_times) > 1 else 0,
                    "all_runs_seconds": run_times
                },
                "schedule_statistics": avg_stats
            }
        else:
            result = {
                "optimizer": "MetroScheduleOptimizer",
                "fleet_size": num_trains,
                "num_stations": num_stations,
                "num_runs": num_runs,
                "successful_runs": 0,
                "success_rate": "0%",
                "error": "All runs failed"
            }
        
        print(f"\nSummary:")
        print(f"  Success Rate: {result['success_rate']}")
        if run_times:
            print(f"  Mean Time: {result['execution_time']['mean_seconds']:.4f}s")
            print(f"  Std Dev: {result['execution_time']['stdev_seconds']:.4f}s")
        
        return result
    
    def benchmark_greedy_optimizers(
        self,
        num_trains: int,
        methods: List[str] = ['ga', 'cmaes', 'pso'],
        num_runs: int = 3
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Benchmark greedyOptim methods"""
        print(f"\n{'='*70}")
        print(f"Benchmarking Greedy Optimization Methods")
        print(f"Fleet Size: {num_trains} trains | Methods: {methods}")
        print(f"{'='*70}")
        
        results_by_method = {}
        
        # Generate complete synthetic data using EnhancedMetroDataGenerator
        print(f"Generating complete synthetic data for {num_trains} trains...")
        try:
            # Use EnhancedMetroDataGenerator for complete, realistic data
            generator = EnhancedMetroDataGenerator(num_trainsets=num_trains)
            synthetic_data = generator.generate_complete_enhanced_dataset()
            
            print(f"  ✓ Generated {len(synthetic_data['trainset_status'])} trainset statuses")
            print(f"  ✓ Generated {len(synthetic_data['fitness_certificates'])} fitness certificates")
            print(f"  ✓ Generated {len(synthetic_data['job_cards'])} job cards")
            
        except Exception as e:
            print(f"✗ Failed to generate synthetic data: {e}")
            import traceback
            traceback.print_exc()
            return results_by_method
        
        for method in methods:
            print(f"\n--- Testing Method: {method.upper()} ---")
            
            run_times = []
            success_count = 0
            results = []
            
            for run in range(num_runs):
                print(f"Run {run + 1}/{num_runs}...", end=" ")
                
                try:
                    config = OptimizationConfig()
                    
                    start_time = time.perf_counter()
                    result = optimize_trainset_schedule(synthetic_data, method, config)
                    end_time = time.perf_counter()
                    
                    elapsed = end_time - start_time
                    run_times.append(elapsed)
                    success_count += 1
                    results.append(result)
                    
                    print(f"✓ {elapsed:.4f}s | Score: {result.fitness_score:.4f}")
                    
                except Exception as e:
                    print(f"✗ Failed: {str(e)[:50]}")
                    continue
            
            if run_times:
                method_result = {
                    "method": method,
                    "optimizer_family": "GreedyOptim",
                    "fleet_size": num_trains,
                    "num_runs": num_runs,
                    "successful_runs": success_count,
                    "success_rate": f"{(success_count/num_runs)*100:.1f}%",
                    "execution_time": {
                        "min_seconds": min(run_times),
                        "max_seconds": max(run_times),
                        "mean_seconds": statistics.mean(run_times),
                        "median_seconds": statistics.median(run_times),
                        "stdev_seconds": statistics.stdev(run_times) if len(run_times) > 1 else 0,
                    },
                    "optimization_scores": {
                        "mean": statistics.mean([r.fitness_score for r in results]),
                        "min": min([r.fitness_score for r in results]),
                        "max": max([r.fitness_score for r in results]),
                    }
                }
                results_by_method[method] = method_result
            
        return results_by_method
    
    def run_comprehensive_benchmark(
        self,
        fleet_sizes: List[int] = [10, 20, 30],
        greedy_methods: List[str] = ['ga', 'cmaes', 'pso'],
        num_runs: int = 3
    ):
        """Run comprehensive performance benchmark"""
        print("\n" + "="*70)
        print("COMPREHENSIVE SCHEDULE GENERATION PERFORMANCE BENCHMARK")
        print("="*70)
        print(f"Fleet Sizes: {fleet_sizes}")
        print(f"Greedy Methods: {greedy_methods}")
        print(f"Runs per Configuration: {num_runs}")
        print("="*70)
        
        # Store configurations
        self.results["configurations"].append({
            "fleet_sizes": fleet_sizes,
            "greedy_methods": greedy_methods,
            "runs_per_config": num_runs,
            "station_count": 22
        })
        
        all_results = []
        
        for fleet_size in fleet_sizes:
            print(f"\n{'#'*70}")
            print(f"# FLEET SIZE: {fleet_size} TRAINS")
            print(f"{'#'*70}")
            
            # Benchmark Schedule Generation
            schedule_result = self.benchmark_schedule_generation(
                num_trains=fleet_size,
                num_runs=num_runs
            )
            all_results.append(schedule_result)
            
            # Benchmark Greedy Optimizers
            greedy_results = self.benchmark_greedy_optimizers(
                num_trains=fleet_size,
                methods=greedy_methods,
                num_runs=num_runs
            )
            
            for method, result in greedy_results.items():
                all_results.append(result)
            
            time.sleep(0.5)  # Brief pause between fleet sizes
        
        self.results["detailed_results"] = all_results
        self._generate_performance_summary()
    
    def _generate_performance_summary(self):
        """Generate comparative performance summary"""
        print("\n" + "="*70)
        print("PERFORMANCE SUMMARY")
        print("="*70)
        
        # Group by fleet size
        fleet_sizes = sorted(set(
            r["fleet_size"] for r in self.results["detailed_results"]
            if "fleet_size" in r
        ))
        
        summary_by_fleet = {}
        
        for fleet_size in fleet_sizes:
            fleet_results = [
                r for r in self.results["detailed_results"]
                if r.get("fleet_size") == fleet_size and "execution_time" in r
            ]
            
            print(f"\n{'Fleet Size:':<20} {fleet_size} trains")
            print("-" * 70)
            print(f"{'Optimizer':<30} {'Mean Time (s)':<15} {'Success Rate':<15}")
            print("-" * 70)
            
            fleet_summary = []
            for result in fleet_results:
                name = result.get("optimizer") or result.get("method", "Unknown")
                mean_time = result["execution_time"]["mean_seconds"]
                success = result["success_rate"]
                
                print(f"{name:<30} {mean_time:<15.4f} {success:<15}")
                fleet_summary.append({
                    "optimizer": name,
                    "mean_time_seconds": mean_time,
                    "success_rate": success
                })
            
            summary_by_fleet[fleet_size] = fleet_summary
        
        # Overall rankings
        print("\n" + "="*70)
        print("OVERALL PERFORMANCE RANKINGS (by average time)")
        print("="*70)
        
        optimizer_times = {}
        for result in self.results["detailed_results"]:
            if "execution_time" not in result:
                continue
            
            name = result.get("optimizer") or result.get("method", "Unknown")
            if name not in optimizer_times:
                optimizer_times[name] = []
            optimizer_times[name].append(result["execution_time"]["mean_seconds"])
        
        rankings = [
            (name, statistics.mean(times))
            for name, times in optimizer_times.items()
        ]
        rankings.sort(key=lambda x: x[1])
        
        print(f"\n{'Rank':<8} {'Optimizer/Method':<30} {'Avg Time (s)':<15}")
        print("-" * 70)
        for rank, (name, avg_time) in enumerate(rankings, 1):
            print(f"{rank:<8} {name:<30} {avg_time:<15.4f}")
        
        self.results["summary"] = {
            "by_fleet_size": summary_by_fleet,
            "overall_rankings": {
                name: {"rank": rank, "avg_time_seconds": avg_time}
                for rank, (name, avg_time) in enumerate(rankings, 1)
            },
            "fastest_optimizer": rankings[0][0] if rankings else None,
            "fastest_time_seconds": rankings[0][1] if rankings else None
        }
    
    def save_results(self, filename: Optional[str] = None):
        """Save benchmark results to JSON file"""
        if filename is None:
            filename = f"schedule_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n{'='*70}")
        print(f"Results saved to: {filename}")
        print(f"{'='*70}")
        
        return filename
    
    def generate_report(self):
        """Generate formatted text report"""
        report_filename = f"schedule_performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(report_filename, 'w') as f:
            f.write("="*80 + "\n")
            f.write("METRO SCHEDULE GENERATION PERFORMANCE REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Test Type: Schedule Generation Time & Computational Efficiency\n\n")
            
            # Executive Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-"*80 + "\n\n")
            
            if "summary" in self.results and "fastest_optimizer" in self.results["summary"]:
                f.write(f"Fastest Optimizer: {self.results['summary']['fastest_optimizer']}\n")
                f.write(f"Best Average Time: {self.results['summary']['fastest_time_seconds']:.4f} seconds\n\n")
            
            # Rankings
            if "summary" in self.results and "overall_rankings" in self.results["summary"]:
                f.write("Overall Performance Rankings:\n")
                for name, data in sorted(
                    self.results["summary"]["overall_rankings"].items(),
                    key=lambda x: x[1]["rank"]
                ):
                    f.write(f"  {data['rank']}. {name}: {data['avg_time_seconds']:.4f}s\n")
                f.write("\n")
            
            # Detailed Results
            f.write("\nDETAILED RESULTS\n")
            f.write("-"*80 + "\n\n")
            
            for result in self.results["detailed_results"]:
                name = result.get("optimizer") or result.get("method", "Unknown")
                f.write(f"Optimizer/Method: {name}\n")
                f.write(f"Fleet Size: {result.get('fleet_size', 'N/A')} trains\n")
                f.write(f"Success Rate: {result.get('success_rate', 'N/A')}\n")
                
                if "execution_time" in result:
                    f.write(f"Execution Time Statistics:\n")
                    f.write(f"  Mean:   {result['execution_time']['mean_seconds']:.4f}s\n")
                    f.write(f"  Median: {result['execution_time']['median_seconds']:.4f}s\n")
                    f.write(f"  Min:    {result['execution_time']['min_seconds']:.4f}s\n")
                    f.write(f"  Max:    {result['execution_time']['max_seconds']:.4f}s\n")
                    f.write(f"  StdDev: {result['execution_time']['stdev_seconds']:.4f}s\n")
                
                if "optimization_scores" in result:
                    f.write(f"Optimization Scores:\n")
                    f.write(f"  Mean: {result['optimization_scores']['mean']:.4f}\n")
                    f.write(f"  Min:  {result['optimization_scores']['min']:.4f}\n")
                    f.write(f"  Max:  {result['optimization_scores']['max']:.4f}\n")
                
                f.write("\n" + "-"*80 + "\n\n")
        
        print(f"Performance report saved to: {report_filename}")
        return report_filename


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Benchmark metro schedule generation performance"
    )
    parser.add_argument(
        "--fleet-sizes",
        nargs="+",
        type=int,
        default=[10, 20, 30],
        help="Fleet sizes to test (default: 10 20 30)"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=['ga', 'cmaes', 'pso'],
        help="Greedy optimization methods to test (default: ga cmaes pso)"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs per configuration (default: 3)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test with minimal configurations"
    )
    
    args = parser.parse_args()
    
    if args.quick:
        fleet_sizes = [10, 20]
        methods = ['ga']
        num_runs = 2
        print("\n*** QUICK BENCHMARK MODE ***")
    else:
        fleet_sizes = args.fleet_sizes
        methods = args.methods
        num_runs = args.runs
    
    # Run benchmark
    benchmark = SchedulePerformanceBenchmark()
    benchmark.run_comprehensive_benchmark(
        fleet_sizes=fleet_sizes,
        greedy_methods=methods,
        num_runs=num_runs
    )
    
    # Save results
    json_file = benchmark.save_results()
    report_file = benchmark.generate_report()
    
    print("\n" + "="*70)
    print("BENCHMARK COMPLETE")
    print("="*70)
    print(f"JSON Results: {json_file}")
    print(f"Text Report:  {report_file}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
