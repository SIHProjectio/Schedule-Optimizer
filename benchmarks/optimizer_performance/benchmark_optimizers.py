#!/usr/bin/env python3
"""
Benchmark script for comparing optimizer performance
Measures schedule generation time and computational efficiency
"""
import time
import json
import statistics
from datetime import datetime, date
from typing import Dict, List, Any, Optional
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from DataService.metro_data_generator import MetroDataGenerator
from DataService.schedule_optimizer import MetroScheduleOptimizer
from greedyOptim.scheduler import TrainsetSchedulingOptimizer
from greedyOptim.genetic_algorithm import GeneticAlgorithmOptimizer
from greedyOptim.ortools_optimizers import ORToolsOptimizer
from greedyOptim.models import OptimizationConfig


class OptimizerBenchmark:
    """Benchmark different optimization algorithms"""
    
    def __init__(self, num_trains: int = 25, num_stations: int = 22):
        self.results = {
            "benchmark_info": {
                "date": datetime.now().isoformat(),
                "description": "Metro Schedule Optimization Performance Comparison"
            },
            "test_configurations": [],
            "results": []
        }
        self.data_generator = MetroDataGenerator(num_trains=num_trains, num_stations=num_stations)
    
    def generate_test_data(self):
        """Generate consistent test data for all optimizers"""
        route = self.data_generator.generate_route(
            route_name="Aluva-Pettah Line"
        )
        
        trains = self.data_generator.generate_train_health_statuses()
        # Limit to requested number
        trains = trains[:num_trains]
        
        return route, trains
    
    def benchmark_optimizer(
        self,
        optimizer_name: str,
        optimizer_class,
        num_trains: int,
        num_stations: int = 22,
        num_runs: int = 3
    ) -> Dict[str, Any]:
        """Benchmark a single optimizer"""
        print(f"\n{'='*70}")
        print(f"Benchmarking: {optimizer_name}")
        print(f"Fleet Size: {num_trains} trains | Stations: {num_stations}")
        print(f"{'='*70}")
        
        run_times = []
        success_count = 0
        schedules_generated = []
        
        for run in range(num_runs):
            print(f"\nRun {run + 1}/{num_runs}...", end=" ")
            
            try:
                # Generate fresh data for each run
                route, trains = self.generate_test_data(num_trains, num_stations)
                
                # Create request
                request = ScheduleRequest(
                    date=date.today(),
                    num_trains=num_trains,
                    route=route,
                    trains=trains
                )
                
                # Time the optimization
                start_time = time.perf_counter()
                
                optimizer = optimizer_class()
                schedule = optimizer.optimize(request)
                
                end_time = time.perf_counter()
                elapsed = end_time - start_time
                
                run_times.append(elapsed)
                success_count += 1
                schedules_generated.append(schedule)
                
                print(f"✓ Completed in {elapsed:.4f}s")
                
            except Exception as e:
                print(f"✗ Failed: {str(e)[:50]}")
                continue
        
        # Calculate statistics
        if run_times:
            result = {
                "optimizer": optimizer_name,
                "fleet_size": num_trains,
                "num_stations": num_stations,
                "num_runs": num_runs,
                "successful_runs": success_count,
                "success_rate": f"{(success_count/num_runs)*100:.1f}%",
                "execution_times": {
                    "min_seconds": min(run_times),
                    "max_seconds": max(run_times),
                    "mean_seconds": statistics.mean(run_times),
                    "median_seconds": statistics.median(run_times),
                    "stdev_seconds": statistics.stdev(run_times) if len(run_times) > 1 else 0,
                    "all_runs": run_times
                },
                "schedule_quality": self._analyze_schedules(schedules_generated) if schedules_generated else None
            }
        else:
            result = {
                "optimizer": optimizer_name,
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
            print(f"  Average Time: {result['execution_times']['mean_seconds']:.4f}s")
            print(f"  Std Dev: {result['execution_times']['stdev_seconds']:.4f}s")
        
        return result
    
    def _analyze_schedules(self, schedules: List) -> Dict[str, Any]:
        """Analyze quality metrics of generated schedules"""
        if not schedules:
            return None
        
        total_trips_list = []
        trains_used_list = []
        
        for schedule in schedules:
            if hasattr(schedule, 'trips'):
                total_trips_list.append(len(schedule.trips))
            if hasattr(schedule, 'train_schedules'):
                trains_used_list.append(len(schedule.train_schedules))
        
        quality = {}
        
        if total_trips_list:
            quality["trips"] = {
                "mean": statistics.mean(total_trips_list),
                "min": min(total_trips_list),
                "max": max(total_trips_list)
            }
        
        if trains_used_list:
            quality["trains_utilized"] = {
                "mean": statistics.mean(trains_used_list),
                "min": min(trains_used_list),
                "max": max(trains_used_list)
            }
        
        return quality if quality else None
    
    def run_comprehensive_benchmark(
        self,
        fleet_sizes: List[int] = [5, 10, 15, 20, 25, 30],
        num_runs: int = 3
    ):
        """Run comprehensive benchmark across all optimizers and fleet sizes"""
        print("\n" + "="*70)
        print("COMPREHENSIVE OPTIMIZER BENCHMARK")
        print("="*70)
        print(f"Fleet Sizes to Test: {fleet_sizes}")
        print(f"Runs per Configuration: {num_runs}")
        print("="*70)
        
        # Define optimizers to test
        optimizers = [
            ("Greedy Optimizer", GreedyScheduleOptimizer),
            ("Genetic Algorithm", GeneticScheduleOptimizer),
            ("OR-Tools CP-SAT", ORToolsScheduleOptimizer),
        ]
        
        # Store test configurations
        self.results["test_configurations"] = [
            {
                "fleet_sizes": fleet_sizes,
                "num_stations": 22,
                "runs_per_config": num_runs,
                "optimizers": [name for name, _ in optimizers]
            }
        ]
        
        # Run benchmarks
        for fleet_size in fleet_sizes:
            print(f"\n{'#'*70}")
            print(f"# FLEET SIZE: {fleet_size} TRAINS")
            print(f"{'#'*70}")
            
            for optimizer_name, optimizer_class in optimizers:
                result = self.benchmark_optimizer(
                    optimizer_name,
                    optimizer_class,
                    fleet_size,
                    num_runs=num_runs
                )
                self.results["results"].append(result)
                
                # Small delay between tests
                time.sleep(0.5)
        
        # Generate comparison summary
        self._generate_summary()
    
    def _generate_summary(self):
        """Generate comparative summary of results"""
        print("\n" + "="*70)
        print("BENCHMARK SUMMARY")
        print("="*70)
        
        # Group by fleet size
        fleet_sizes = sorted(set(r["fleet_size"] for r in self.results["results"]))
        
        summary = {
            "by_fleet_size": {},
            "overall_rankings": {}
        }
        
        for fleet_size in fleet_sizes:
            fleet_results = [r for r in self.results["results"] if r["fleet_size"] == fleet_size]
            
            print(f"\nFleet Size: {fleet_size} trains")
            print("-" * 70)
            print(f"{'Optimizer':<25} {'Avg Time (s)':<15} {'Success Rate':<15}")
            print("-" * 70)
            
            fleet_summary = []
            for result in fleet_results:
                optimizer = result["optimizer"]
                avg_time = result["execution_times"]["mean_seconds"] if "execution_times" in result else "N/A"
                success = result["success_rate"]
                
                if avg_time != "N/A":
                    print(f"{optimizer:<25} {avg_time:<15.4f} {success:<15}")
                    fleet_summary.append({
                        "optimizer": optimizer,
                        "avg_time": avg_time,
                        "success_rate": success
                    })
                else:
                    print(f"{optimizer:<25} {'FAILED':<15} {success:<15}")
            
            summary["by_fleet_size"][fleet_size] = fleet_summary
        
        # Overall performance ranking
        print("\n" + "="*70)
        print("OVERALL PERFORMANCE RANKING")
        print("="*70)
        
        optimizer_avg_times = {}
        for result in self.results["results"]:
            if "execution_times" in result:
                optimizer = result["optimizer"]
                if optimizer not in optimizer_avg_times:
                    optimizer_avg_times[optimizer] = []
                optimizer_avg_times[optimizer].append(result["execution_times"]["mean_seconds"])
        
        rankings = []
        for optimizer, times in optimizer_avg_times.items():
            avg = statistics.mean(times)
            rankings.append((optimizer, avg))
        
        rankings.sort(key=lambda x: x[1])
        
        print(f"\n{'Rank':<8} {'Optimizer':<25} {'Avg Time (s)':<15}")
        print("-" * 70)
        for rank, (optimizer, avg_time) in enumerate(rankings, 1):
            print(f"{rank:<8} {optimizer:<25} {avg_time:<15.4f}")
            summary["overall_rankings"][optimizer] = {
                "rank": rank,
                "avg_time_seconds": avg_time
            }
        
        self.results["summary"] = summary
    
    def save_results(self, filename: str = None):
        """Save benchmark results to JSON file"""
        if filename is None:
            filename = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n{'='*70}")
        print(f"Results saved to: {filename}")
        print(f"{'='*70}")
        
        return filename
    
    def generate_performance_report(self):
        """Generate a formatted performance report"""
        report_filename = f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(report_filename, 'w') as f:
            f.write("="*80 + "\n")
            f.write("METRO SCHEDULE OPTIMIZER PERFORMANCE REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-"*80 + "\n")
            if "summary" in self.results and "overall_rankings" in self.results["summary"]:
                f.write("\nOverall Performance Ranking:\n")
                for optimizer, data in self.results["summary"]["overall_rankings"].items():
                    f.write(f"  {data['rank']}. {optimizer}: {data['avg_time_seconds']:.4f}s average\n")
            f.write("\n")
            
            # Detailed Results
            f.write("\nDETAILED RESULTS BY FLEET SIZE\n")
            f.write("-"*80 + "\n\n")
            
            for result in self.results["results"]:
                f.write(f"Optimizer: {result['optimizer']}\n")
                f.write(f"Fleet Size: {result['fleet_size']} trains\n")
                f.write(f"Success Rate: {result['success_rate']}\n")
                
                if "execution_times" in result:
                    f.write(f"Execution Time:\n")
                    f.write(f"  Mean:   {result['execution_times']['mean_seconds']:.4f}s\n")
                    f.write(f"  Median: {result['execution_times']['median_seconds']:.4f}s\n")
                    f.write(f"  Min:    {result['execution_times']['min_seconds']:.4f}s\n")
                    f.write(f"  Max:    {result['execution_times']['max_seconds']:.4f}s\n")
                    f.write(f"  StdDev: {result['execution_times']['stdev_seconds']:.4f}s\n")
                
                if result.get("schedule_quality"):
                    f.write(f"Schedule Quality:\n")
                    if "trips" in result["schedule_quality"]:
                        f.write(f"  Trips Generated (avg): {result['schedule_quality']['trips']['mean']:.1f}\n")
                    if "trains_utilized" in result["schedule_quality"]:
                        f.write(f"  Trains Utilized (avg): {result['schedule_quality']['trains_utilized']['mean']:.1f}\n")
                
                f.write("\n" + "-"*80 + "\n\n")
            
            # Recommendations
            f.write("\nRECOMMENDATIONS\n")
            f.write("-"*80 + "\n")
            f.write("Based on the benchmark results:\n\n")
            
            if "summary" in self.results and "overall_rankings" in self.results["summary"]:
                rankings = sorted(
                    self.results["summary"]["overall_rankings"].items(),
                    key=lambda x: x[1]["rank"]
                )
                if rankings:
                    fastest = rankings[0]
                    f.write(f"• {fastest[0]} showed the best overall performance\n")
                    f.write(f"  with an average execution time of {fastest[1]['avg_time_seconds']:.4f}s\n\n")
            
            f.write("• Consider using faster optimizers for real-time scheduling\n")
            f.write("• Slower optimizers may provide better solution quality for offline planning\n")
            f.write("• Test with your specific constraints and requirements\n\n")
        
        print(f"Performance report saved to: {report_filename}")
        return report_filename


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark metro schedule optimizers")
    parser.add_argument(
        "--fleet-sizes",
        nargs="+",
        type=int,
        default=[5, 10, 15, 20, 25, 30],
        help="Fleet sizes to test (default: 5 10 15 20 25 30)"
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
        help="Quick test with fewer configurations"
    )
    
    args = parser.parse_args()
    
    if args.quick:
        fleet_sizes = [10, 20, 30]
        num_runs = 2
        print("\n*** QUICK BENCHMARK MODE ***")
    else:
        fleet_sizes = args.fleet_sizes
        num_runs = args.runs
    
    # Run benchmark
    benchmark = OptimizerBenchmark()
    benchmark.run_comprehensive_benchmark(
        fleet_sizes=fleet_sizes,
        num_runs=num_runs
    )
    
    # Save results
    json_file = benchmark.save_results()
    report_file = benchmark.generate_performance_report()
    
    print("\n" + "="*70)
    print("BENCHMARK COMPLETE")
    print("="*70)
    print(f"JSON Results: {json_file}")
    print(f"Text Report:  {report_file}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
