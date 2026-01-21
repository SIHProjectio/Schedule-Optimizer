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

from DataService.generators.enhanced_generator import EnhancedMetroDataGenerator
from DataService.optimizers.schedule_optimizer import MetroScheduleOptimizer
from greedyOptim.scheduler import TrainsetSchedulingOptimizer
from DataService.core.models import Route, TrainHealthStatus

# --- Adapters for Uniform Interface ---

class OptimizerAdapter:
    """Base adapter for different optimizers"""
    def optimize(self, data: Dict) -> Any:
        raise NotImplementedError

class GeneticAdapter(OptimizerAdapter):
    """Adapter for Genetic Algorithm"""
    def optimize(self, data: Dict) -> Any:
        optimizer = TrainsetSchedulingOptimizer(data)
        return optimizer.optimize(method='ga')

class PSOAdapter(OptimizerAdapter):
    """Adapter for Particle Swarm Optimization"""
    def optimize(self, data: Dict) -> Any:
        optimizer = TrainsetSchedulingOptimizer(data)
        return optimizer.optimize(method='pso')

class SAAdapter(OptimizerAdapter):
    """Adapter for Simulated Annealing"""
    def optimize(self, data: Dict) -> Any:
        optimizer = TrainsetSchedulingOptimizer(data)
        return optimizer.optimize(method='sa')

class CMAESAdapter(OptimizerAdapter):
    """Adapter for CMA-ES"""
    def optimize(self, data: Dict) -> Any:
        optimizer = TrainsetSchedulingOptimizer(data)
        return optimizer.optimize(method='cmaes')

class NSGA2Adapter(OptimizerAdapter):
    """Adapter for NSGA-II"""
    def optimize(self, data: Dict) -> Any:
        optimizer = TrainsetSchedulingOptimizer(data)
        return optimizer.optimize(method='nsga2')

class AdaptiveAdapter(OptimizerAdapter):
    """Adapter for Adaptive Algorithm"""
    def optimize(self, data: Dict) -> Any:
        optimizer = TrainsetSchedulingOptimizer(data)
        return optimizer.optimize(method='adaptive')

class EnsembleAdapter(OptimizerAdapter):
    """Adapter for Ensemble Method"""
    def optimize(self, data: Dict) -> Any:
        optimizer = TrainsetSchedulingOptimizer(data)
        return optimizer.optimize(method='ensemble')

class ORToolsAdapter(OptimizerAdapter):
    """Adapter for OR-Tools CP-SAT"""
    def optimize(self, data: Dict) -> Any:
        optimizer = TrainsetSchedulingOptimizer(data)
        return optimizer.optimize(method='cp-sat')


class OptimizerBenchmark:
    """Benchmark different optimization algorithms"""
    
    def __init__(self):
        self.results = {
            "benchmark_info": {
                "date": datetime.now().isoformat(),
                "description": "Metro Schedule Optimization Performance Comparison"
            },
            "test_configurations": [],
            "results": []
        }
    
    def generate_test_data(self, num_trains: int) -> Dict:
        """Generate consistent test data for all optimizers"""
        generator = EnhancedMetroDataGenerator(num_trainsets=num_trains)
        # We need the full dataset as expected by TrainsetSchedulingEvaluator
        full_data = generator.generate_complete_enhanced_dataset()
        return full_data
    
    def benchmark_optimizer(
        self,
        optimizer_name: str,
        adapter_class,
        num_trains: int,
        num_runs: int = 3
    ) -> Dict[str, Any]:
        """Benchmark a single optimizer"""
        print(f"\n{'='*70}")
        print(f"Benchmarking: {optimizer_name}")
        print(f"Fleet Size: {num_trains} trains")
        print(f"{'='*70}")
        
        run_times = []
        success_count = 0
        
        for run in range(num_runs):
            print(f"Run {run + 1}/{num_runs}...", end=" ", flush=True)
            
            try:
                # Generate fresh data for each run
                data = self.generate_test_data(num_trains)
                
                # Time the optimization
                start_time = time.perf_counter()
                
                adapter = adapter_class()
                result = adapter.optimize(data)
                
                end_time = time.perf_counter()
                elapsed = end_time - start_time
                
                run_times.append(elapsed)
                success_count += 1
                
                print(f"✓ Completed in {elapsed:.4f}s | Fitness: {result.fitness_score:.2f}")
                
            except Exception as e:
                print(f"✗ Failed: {str(e)[:100]}")
                # import traceback
                # traceback.print_exc()
        
        # Calculate statistics
        if run_times:
            result = {
                "optimizer": optimizer_name,
                "fleet_size": num_trains,
                "num_runs": num_runs,
                "successful_runs": success_count,
                "success_rate": f"{(success_count/num_runs)*100:.1f}%",
                "execution_times": {
                    "min_seconds": min(run_times),
                    "max_seconds": max(run_times),
                    "mean_seconds": statistics.mean(run_times),
                    "stdev_seconds": statistics.stdev(run_times) if len(run_times) > 1 else 0
                }
            }
        else:
            result = {
                "optimizer": optimizer_name,
                "fleet_size": num_trains,
                "num_runs": num_runs,
                "successful_runs": 0,
                "success_rate": "0%",
                "error": "All runs failed"
            }
        
        print(f"\nSummary:")
        print(f"  Success Rate: {result['success_rate']}")
        if run_times:
            print(f"  Average Time: {result['execution_times']['mean_seconds']:.4f}s")
        
        return result

    def run_comprehensive_benchmark(
        self,
        fleet_sizes: List[int] = [10, 20, 30],
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
            ("Genetic Algorithm", GeneticAdapter),
            ("Particle Swarm", PSOAdapter),
            ("Simulated Annealing", SAAdapter),
            ("CMA-ES", CMAESAdapter),
            ("NSGA-II", NSGA2Adapter),
            ("Adaptive Algorithm", AdaptiveAdapter),
            ("Ensemble Method", EnsembleAdapter),
            # ("OR-Tools CP-SAT", ORToolsAdapter), # Uncomment if OR-Tools is installed
        ]
        
        # Run benchmarks
        for fleet_size in fleet_sizes:
            print(f"\n{'#'*70}")
            print(f"# FLEET SIZE: {fleet_size} TRAINS")
            print(f"{'#'*70}")
            
            for optimizer_name, adapter_class in optimizers:
                result = self.benchmark_optimizer(
                    optimizer_name,
                    adapter_class,
                    fleet_size,
                    num_runs=num_runs
                )
                self.results["results"].append(result)
                
                # Small delay between tests
                time.sleep(0.5)
        
        # Generate comparison summary
        self._generate_summary()
        
        # Save results to benchmark_output/ at project root
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        output_dir = os.path.join(project_root, "benchmark_output")
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"optimizer_benchmark_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to: {filepath}")
    
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
                
                if isinstance(avg_time, float):
                    time_str = f"{avg_time:.4f}"
                else:
                    time_str = str(avg_time)
                    
                print(f"{optimizer:<25} {time_str:<15} {success:<15}")
                
                if isinstance(avg_time, float):
                    fleet_summary.append({
                        "optimizer": optimizer,
                        "time": avg_time
                    })
            
            # Rank for this fleet size
            fleet_summary.sort(key=lambda x: x["time"])
            summary["by_fleet_size"][fleet_size] = fleet_summary
            
            # Update overall stats
            for item in fleet_summary:
                opt = item["optimizer"]
                if opt not in summary["overall_rankings"]:
                    summary["overall_rankings"][opt] = []
                summary["overall_rankings"][opt].append(item["time"])
        
        # Print overall rankings
        print("\n" + "="*70)
        print("OVERALL PERFORMANCE RANKINGS (by average time)")
        print("="*70)
        print(f"{'Rank':<8} {'Optimizer/Method':<30} {'Avg Time (s)':<15}")
        print("-" * 70)
        
        overall_stats = []
        for opt, times in summary["overall_rankings"].items():
            if times:
                overall_stats.append({
                    "optimizer": opt,
                    "avg_time": statistics.mean(times)
                })
        
        overall_stats.sort(key=lambda x: x["avg_time"])
        
        for i, stat in enumerate(overall_stats):
            print(f"{i+1:<8} {stat['optimizer']:<30} {stat['avg_time']:.4f}")
        
        # Save report to text file in benchmark_output/ at project root
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        output_dir = os.path.join(project_root, "benchmark_output")
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(output_dir, f"optimizer_performance_report_{timestamp}.txt")
        
        with open(report_file, "w") as f:
            f.write("OPTIMIZER PERFORMANCE BENCHMARK REPORT\n")
            f.write(f"Date: {datetime.now().isoformat()}\n")
            f.write("="*70 + "\n\n")
            
            for fleet_size in fleet_sizes:
                f.write(f"Fleet Size: {fleet_size} trains\n")
                f.write("-" * 70 + "\n")
                f.write(f"{'Optimizer':<25} {'Avg Time (s)':<15} {'Success Rate':<15}\n")
                f.write("-" * 70 + "\n")
                
                fleet_results = [r for r in self.results["results"] if r["fleet_size"] == fleet_size]
                for result in fleet_results:
                    optimizer = result["optimizer"]
                    avg_time = result["execution_times"]["mean_seconds"] if "execution_times" in result else "N/A"
                    success = result["success_rate"]
                    
                    if isinstance(avg_time, float):
                        time_str = f"{avg_time:.4f}"
                    else:
                        time_str = str(avg_time)
                        
                    f.write(f"{optimizer:<25} {time_str:<15} {success:<15}\n")
                f.write("\n")
            
            f.write("="*70 + "\n")
            f.write("OVERALL RANKINGS\n")
            f.write("="*70 + "\n")
            for i, stat in enumerate(overall_stats):
                f.write(f"{i+1}. {stat['optimizer']}: {stat['avg_time']:.4f}s\n")
                
        print(f"\nPerformance report saved to: {report_file}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark metro schedule optimizers")
    parser.add_argument("--fleet-sizes", type=int, nargs="+", default=[10, 20, 30],
                        help="Fleet sizes to test (default: 10 20 30)")
    parser.add_argument("--runs", type=int, default=3,
                        help="Number of runs per configuration (default: 3)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test with fewer configurations")
    
    args = parser.parse_args()
    
    if args.quick:
        print("\n*** QUICK BENCHMARK MODE ***")
        fleet_sizes = [10, 20]
        runs = 1
    else:
        fleet_sizes = args.fleet_sizes
        runs = args.runs
    
    benchmark = OptimizerBenchmark()
    benchmark.run_comprehensive_benchmark(
        fleet_sizes=fleet_sizes,
        num_runs=runs
    )

if __name__ == "__main__":
    main()
