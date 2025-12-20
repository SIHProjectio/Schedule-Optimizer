#!/usr/bin/env python3
"""
Comprehensive Fleet Utilization Benchmark
Generates data for research paper Results section.
"""
import json
import sys
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import statistics

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from benchmarks.fleet_utilization.fleet_analyzer import (
    FleetUtilizationAnalyzer,
    FleetUtilizationMetrics,
    format_metrics_report
)


class FleetUtilizationBenchmark:
    """Benchmark fleet utilization across different configurations"""
    
    def __init__(self):
        self.analyzer = FleetUtilizationAnalyzer()
        self.results = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "system": "Kochi Metro Rail",
                "analysis_type": "Fleet Utilization"
            },
            "configuration": {
                "route_length_km": self.analyzer.route_length_km,
                "avg_speed_kmh": self.analyzer.avg_speed_kmh,
                "service_hours": self.analyzer.total_service_hours,
                "peak_headway_minutes": self.analyzer.peak_headway_target,
                "offpeak_headway_minutes": self.analyzer.offpeak_headway_target
            },
            "fleet_analyses": [],
            "comparative_analysis": {},
            "optimal_fleet": {}
        }
    
    def run_comprehensive_analysis(
        self,
        fleet_sizes: Optional[List[int]] = None,
        maintenance_rate: float = 0.1
    ):
        """
        Run comprehensive fleet utilization analysis.
        
        Args:
            fleet_sizes: List of fleet sizes to test (default: 10-40 by 5)
            maintenance_rate: Percentage of fleet in maintenance
        """
        if fleet_sizes is None:
            fleet_sizes = [10, 15, 20, 25, 30, 35, 40]
        
        print("="*70)
        print("COMPREHENSIVE FLEET UTILIZATION BENCHMARK")
        print("="*70)
        print(f"Fleet Sizes to Test: {fleet_sizes}")
        print(f"Maintenance Rate: {maintenance_rate*100:.0f}%")
        print("="*70)
        print()
        
        # Analyze each fleet size
        for i, size in enumerate(fleet_sizes, 1):
            print(f"[{i}/{len(fleet_sizes)}] Analyzing fleet size: {size} trains...")
            
            maintenance_trains = max(1, int(size * maintenance_rate))
            metrics = self.analyzer.analyze_fleet_configuration(size, maintenance_trains)
            
            # Store results
            result_dict = {
                "fleet_size": metrics.fleet_size,
                "minimum_required_trains": metrics.minimum_required_trains,
                "trains_in_service_peak": metrics.trains_in_service_peak,
                "trains_in_service_offpeak": metrics.trains_in_service_offpeak,
                "trains_in_standby": metrics.trains_in_standby,
                "trains_in_maintenance": metrics.trains_in_maintenance,
                "peak_demand_coverage_percent": metrics.peak_demand_coverage_percent,
                "offpeak_demand_coverage_percent": metrics.offpeak_demand_coverage_percent,
                "overall_coverage_percent": metrics.overall_coverage_percent,
                "avg_operational_hours_per_train": metrics.avg_operational_hours_per_train,
                "avg_idle_hours_per_train": metrics.avg_idle_hours_per_train,
                "utilization_rate_percent": metrics.utilization_rate_percent,
                "fleet_efficiency_score": metrics.fleet_efficiency_score,
                "cost_efficiency_score": metrics.cost_efficiency_score
            }
            
            self.results["fleet_analyses"].append(result_dict)
            
            print(f"  ✓ Coverage: {metrics.overall_coverage_percent:.1f}%")
            print(f"  ✓ Utilization: {metrics.utilization_rate_percent:.1f}%")
            print(f"  ✓ Efficiency: {metrics.fleet_efficiency_score:.1f}/100")
            print()
        
        # Comparative analysis
        self._generate_comparative_analysis()
        
        # Find optimal fleet
        self._find_optimal_configuration()
        
        print("="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
    
    def _generate_comparative_analysis(self):
        """Generate comparative statistics across all fleet sizes"""
        analyses = self.results["fleet_analyses"]
        
        if not analyses:
            return
        
        # Extract metrics
        coverage = [a["overall_coverage_percent"] for a in analyses]
        utilization = [a["utilization_rate_percent"] for a in analyses]
        efficiency = [a["fleet_efficiency_score"] for a in analyses]
        
        # Find best performers
        best_coverage_idx = coverage.index(max(coverage))
        best_utilization_idx = utilization.index(max(utilization))
        best_efficiency_idx = efficiency.index(max(efficiency))
        
        self.results["comparative_analysis"] = {
            "coverage_statistics": {
                "min": min(coverage),
                "max": max(coverage),
                "mean": statistics.mean(coverage),
                "median": statistics.median(coverage),
                "stdev": statistics.stdev(coverage) if len(coverage) > 1 else 0
            },
            "utilization_statistics": {
                "min": min(utilization),
                "max": max(utilization),
                "mean": statistics.mean(utilization),
                "median": statistics.median(utilization),
                "stdev": statistics.stdev(utilization) if len(utilization) > 1 else 0
            },
            "efficiency_statistics": {
                "min": min(efficiency),
                "max": max(efficiency),
                "mean": statistics.mean(efficiency),
                "median": statistics.median(efficiency),
                "stdev": statistics.stdev(efficiency) if len(efficiency) > 1 else 0
            },
            "best_performers": {
                "best_coverage": {
                    "fleet_size": analyses[best_coverage_idx]["fleet_size"],
                    "coverage_percent": analyses[best_coverage_idx]["overall_coverage_percent"]
                },
                "best_utilization": {
                    "fleet_size": analyses[best_utilization_idx]["fleet_size"],
                    "utilization_percent": analyses[best_utilization_idx]["utilization_rate_percent"]
                },
                "best_efficiency": {
                    "fleet_size": analyses[best_efficiency_idx]["fleet_size"],
                    "efficiency_score": analyses[best_efficiency_idx]["fleet_efficiency_score"]
                }
            }
        }
    
    def _find_optimal_configuration(self):
        """Find and store optimal fleet configuration"""
        print("\nFinding optimal fleet configuration...")
        
        optimal_size, optimal_metrics = self.analyzer.find_optimal_fleet_size(
            min_coverage_required=95.0
        )
        
        self.results["optimal_fleet"] = {
            "optimal_fleet_size": optimal_size,
            "minimum_required_trains": optimal_metrics.minimum_required_trains,
            "coverage_percent": optimal_metrics.overall_coverage_percent,
            "utilization_percent": optimal_metrics.utilization_rate_percent,
            "efficiency_score": optimal_metrics.fleet_efficiency_score,
            "cost_efficiency_score": optimal_metrics.cost_efficiency_score,
            "operational_hours_per_train": optimal_metrics.avg_operational_hours_per_train,
            "idle_hours_per_train": optimal_metrics.avg_idle_hours_per_train
        }
        
        print(f"  ✓ Optimal Fleet Size: {optimal_size} trains")
        print(f"  ✓ Coverage: {optimal_metrics.overall_coverage_percent:.1f}%")
        print(f"  ✓ Efficiency: {optimal_metrics.fleet_efficiency_score:.1f}/100")
    
    def save_results(self, filename: Optional[str] = None, output_dir: Optional[str] = None) -> str:
        """Save results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"fleet_utilization_benchmark_{timestamp}.json"
        
        # Default to benchmark_output/ at project root
        if output_dir is None:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            output_dir = os.path.join(project_root, "benchmark_output")
        
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n✓ Results saved to: {filepath}")
        return filepath
    
    def generate_report(self, filename: Optional[str] = None, output_dir: Optional[str] = None) -> str:
        """Generate human-readable text report"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"fleet_utilization_report_{timestamp}.txt"
        
        # Default to benchmark_output/ at project root
        if output_dir is None:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            output_dir = os.path.join(project_root, "benchmark_output")
        
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write("="*70 + "\n")
            f.write("FLEET UTILIZATION BENCHMARK REPORT\n")
            f.write("="*70 + "\n\n")
            
            # Metadata
            f.write(f"Generated: {self.results['metadata']['generated_at']}\n")
            f.write(f"System: {self.results['metadata']['system']}\n\n")
            
            # Configuration
            f.write("Configuration:\n")
            f.write("-"*70 + "\n")
            config = self.results['configuration']
            f.write(f"  Route Length: {config['route_length_km']} km\n")
            f.write(f"  Average Speed: {config['avg_speed_kmh']} km/h\n")
            f.write(f"  Service Hours: {config['service_hours']} hours/day\n")
            f.write(f"  Peak Headway Target: {config['peak_headway_minutes']} minutes\n")
            f.write(f"  Off-Peak Headway Target: {config['offpeak_headway_minutes']} minutes\n\n")
            
            # Optimal Fleet
            f.write("OPTIMAL FLEET CONFIGURATION:\n")
            f.write("-"*70 + "\n")
            optimal = self.results['optimal_fleet']
            f.write(f"  Optimal Fleet Size: {optimal['optimal_fleet_size']} trains\n")
            f.write(f"  Minimum Required: {optimal['minimum_required_trains']} trains\n")
            f.write(f"  Coverage: {optimal['coverage_percent']:.1f}%\n")
            f.write(f"  Utilization Rate: {optimal['utilization_percent']:.1f}%\n")
            f.write(f"  Fleet Efficiency: {optimal['efficiency_score']:.1f}/100\n")
            f.write(f"  Operational Hours/Train: {optimal['operational_hours_per_train']:.2f} hrs/day\n")
            f.write(f"  Idle Hours/Train: {optimal['idle_hours_per_train']:.2f} hrs/day\n\n")
            
            # Comparative Analysis
            f.write("COMPARATIVE ANALYSIS:\n")
            f.write("-"*70 + "\n")
            comp = self.results['comparative_analysis']
            
            f.write("\nCoverage Statistics:\n")
            stats = comp['coverage_statistics']
            f.write(f"  Mean: {stats['mean']:.2f}%\n")
            f.write(f"  Median: {stats['median']:.2f}%\n")
            f.write(f"  Range: {stats['min']:.2f}% - {stats['max']:.2f}%\n")
            f.write(f"  Std Dev: {stats['stdev']:.2f}\n")
            
            f.write("\nUtilization Statistics:\n")
            stats = comp['utilization_statistics']
            f.write(f"  Mean: {stats['mean']:.2f}%\n")
            f.write(f"  Median: {stats['median']:.2f}%\n")
            f.write(f"  Range: {stats['min']:.2f}% - {stats['max']:.2f}%\n")
            f.write(f"  Std Dev: {stats['stdev']:.2f}\n")
            
            f.write("\nEfficiency Statistics:\n")
            stats = comp['efficiency_statistics']
            f.write(f"  Mean: {stats['mean']:.2f}/100\n")
            f.write(f"  Median: {stats['median']:.2f}/100\n")
            f.write(f"  Range: {stats['min']:.2f} - {stats['max']:.2f}\n")
            f.write(f"  Std Dev: {stats['stdev']:.2f}\n")
            
            # Best Performers
            f.write("\nBest Performers:\n")
            best = comp['best_performers']
            f.write(f"  Best Coverage: {best['best_coverage']['fleet_size']} trains ({best['best_coverage']['coverage_percent']:.1f}%)\n")
            f.write(f"  Best Utilization: {best['best_utilization']['fleet_size']} trains ({best['best_utilization']['utilization_percent']:.1f}%)\n")
            f.write(f"  Best Efficiency: {best['best_efficiency']['fleet_size']} trains ({best['best_efficiency']['efficiency_score']:.1f}/100)\n\n")
            
            # Detailed Results
            f.write("DETAILED FLEET ANALYSES:\n")
            f.write("="*70 + "\n\n")
            
            for analysis in self.results['fleet_analyses']:
                f.write(f"Fleet Size: {analysis['fleet_size']} trains\n")
                f.write("-"*70 + "\n")
                f.write(f"  Minimum Required: {analysis['minimum_required_trains']} trains\n")
                f.write(f"  Peak Service: {analysis['trains_in_service_peak']} trains\n")
                f.write(f"  Off-Peak Service: {analysis['trains_in_service_offpeak']} trains\n")
                f.write(f"  Standby: {analysis['trains_in_standby']} trains\n")
                f.write(f"  Maintenance: {analysis['trains_in_maintenance']} trains\n")
                f.write(f"  Peak Coverage: {analysis['peak_demand_coverage_percent']:.1f}%\n")
                f.write(f"  Off-Peak Coverage: {analysis['offpeak_demand_coverage_percent']:.1f}%\n")
                f.write(f"  Overall Coverage: {analysis['overall_coverage_percent']:.1f}%\n")
                f.write(f"  Operational Hours/Train: {analysis['avg_operational_hours_per_train']:.2f} hrs\n")
                f.write(f"  Idle Hours/Train: {analysis['avg_idle_hours_per_train']:.2f} hrs\n")
                f.write(f"  Utilization Rate: {analysis['utilization_rate_percent']:.1f}%\n")
                f.write(f"  Fleet Efficiency: {analysis['fleet_efficiency_score']:.1f}/100\n")
                f.write(f"  Cost Efficiency: {analysis['cost_efficiency_score']:.1f}/100\n")
                f.write("\n")
        
        print(f"✓ Report saved to: {filepath}")
        return filepath


def main():
    """Run comprehensive fleet utilization benchmark"""
    benchmark = FleetUtilizationBenchmark()
    
    # Run analysis for various fleet sizes
    benchmark.run_comprehensive_analysis(
        fleet_sizes=[10, 15, 20, 25, 30, 35, 40],
        maintenance_rate=0.1
    )
    
    # Save results
    benchmark.save_results()
    benchmark.generate_report()
    
    print("\n" + "="*70)
    print("BENCHMARK COMPLETE")
    print("="*70)
    print("\nFiles generated:")
    print("  - fleet_utilization_benchmark_TIMESTAMP.json")
    print("  - fleet_utilization_report_TIMESTAMP.txt")
    print("\nUse these results for your research paper Results section!")


if __name__ == "__main__":
    main()
