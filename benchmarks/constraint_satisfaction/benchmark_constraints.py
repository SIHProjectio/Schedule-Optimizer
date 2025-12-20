"""
Constraint Satisfaction Benchmark Script
Runs comprehensive benchmarks for maintenance, turnaround, and energy constraints.
"""
import time
import json
from typing import Dict, List
from datetime import datetime

from .constraint_analyzer import ConstraintAnalyzer, ConstraintMetrics


def run_constraint_benchmark(
    schedules: List[Dict],
    data: Dict,
    output_file: str = "constraint_satisfaction_results.json",
    output_dir: str = None,
    verbose: bool = True
) -> Dict:
    """Run constraint satisfaction benchmark on multiple schedules.
    
    Args:
        schedules: List of schedule dictionaries to analyze
        data: Original metro data (trainset status, certs, jobs, components)
        output_file: Path to save results JSON
        verbose: Whether to print detailed output
        
    Returns:
        Dictionary with benchmark results
    """
    if verbose:
        print("=" * 80)
        print("CONSTRAINT SATISFACTION BENCHMARK")
        print("=" * 80)
        print(f"Analyzing {len(schedules)} schedules...")
        print()
    
    analyzer = ConstraintAnalyzer()
    start_time = time.time()
    
    results = []
    
    # Analyze each schedule
    for i, schedule in enumerate(schedules, 1):
        if verbose:
            print(f"Schedule {i}/{len(schedules)}:")
        
        schedule_start = time.time()
        metrics = analyzer.analyze_schedule(schedule, data)
        analysis_time = (time.time() - schedule_start) * 1000  # ms
        
        result = {
            'schedule_id': i,
            'analysis_time_ms': round(analysis_time, 2),
            'metrics': {
                'maintenance': {
                    'trains_needing_maintenance': metrics.trains_needing_maintenance,
                    'trains_scheduled_maintenance': metrics.trains_scheduled_maintenance,
                    'compliance_rate': round(metrics.maintenance_compliance_rate, 2),
                    'overdue_count': metrics.overdue_maintenance_count,
                    'violations': metrics.maintenance_window_violations,
                    'avg_delay_days': round(metrics.avg_maintenance_delay_days, 2),
                    'score': round(metrics.maintenance_score, 2)
                },
                'turnaround': {
                    'total_turnarounds': metrics.total_turnarounds,
                    'compliant_turnarounds': metrics.compliant_turnarounds,
                    'compliance_rate': round(metrics.turnaround_compliance_rate, 2),
                    'avg_time_minutes': round(metrics.avg_turnaround_time_minutes, 2),
                    'min_time_minutes': round(metrics.min_turnaround_time_minutes, 2),
                    'violations': metrics.turnaround_violations,
                    'score': round(metrics.turnaround_score, 2)
                },
                'energy': {
                    'total_daily_energy_kwh': round(metrics.total_daily_energy_kwh, 2),
                    'peak_power_kw': round(metrics.peak_power_demand_kw, 2),
                    'efficiency_kwh_per_km': round(metrics.energy_efficiency_score, 2),
                    'range_violations': metrics.battery_range_violations,
                    'charging_opportunities': metrics.charging_opportunities,
                    'violations': metrics.energy_constraint_violations,
                    'score': round(metrics.energy_score, 2)
                },
                'certificates': {
                    'expired': metrics.trains_with_expired_certs,
                    'expiring_soon': metrics.trains_with_expiring_soon_certs,
                    'compliance_rate': round(metrics.certificate_compliance_rate, 2),
                    'score': round(metrics.certificate_score, 2)
                },
                'jobs': {
                    'critical_jobs': metrics.trains_with_critical_jobs,
                    'blocking_jobs': metrics.trains_with_blocking_jobs,
                    'violations': metrics.job_constraint_violations,
                    'score': round(metrics.job_score, 2)
                },
                'components': {
                    'critical': metrics.trains_with_critical_components,
                    'warning': metrics.trains_with_warning_components,
                    'violations': metrics.component_constraint_violations,
                    'score': round(metrics.component_score, 2)
                },
                'overall_constraint_score': round(metrics.overall_constraint_score, 2)
            }
        }
        
        results.append(result)
        
        if verbose:
            print(f"  Maintenance Score: {metrics.maintenance_score:.2f}/100")
            print(f"    Compliance: {metrics.maintenance_compliance_rate:.1f}% ({metrics.trains_scheduled_maintenance}/{metrics.trains_needing_maintenance})")
            print(f"    Overdue: {metrics.overdue_maintenance_count}, Violations: {metrics.maintenance_window_violations}")
            print(f"  Turnaround Score: {metrics.turnaround_score:.2f}/100")
            print(f"    Compliance: {metrics.turnaround_compliance_rate:.1f}% ({metrics.compliant_turnarounds}/{metrics.total_turnarounds})")
            print(f"    Avg time: {metrics.avg_turnaround_time_minutes:.1f}min, Min: {metrics.min_turnaround_time_minutes:.1f}min")
            print(f"  Energy Score: {metrics.energy_score:.2f}/100")
            print(f"    Total: {metrics.total_daily_energy_kwh:.1f} kWh, Efficiency: {metrics.energy_efficiency_score:.2f} kWh/km")
            print(f"    Violations: {metrics.energy_constraint_violations}")
            print(f"  Certificate Score: {metrics.certificate_score:.2f}/100")
            print(f"    Expired: {metrics.trains_with_expired_certs}, Expiring: {metrics.trains_with_expiring_soon_certs}")
            print(f"  Job Score: {metrics.job_score:.2f}/100")
            print(f"    Critical: {metrics.trains_with_critical_jobs}, Blocking: {metrics.trains_with_blocking_jobs}")
            print(f"  Component Score: {metrics.component_score:.2f}/100")
            print(f"    Critical: {metrics.trains_with_critical_components}, Warning: {metrics.trains_with_warning_components}")
            print(f"  OVERALL CONSTRAINT SCORE: {metrics.overall_constraint_score:.2f}/100")
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
        print(f"  Maintenance: {aggregate['avg_maintenance_score']:.2f}/100")
        print(f"  Turnaround: {aggregate['avg_turnaround_score']:.2f}/100")
        print(f"  Energy: {aggregate['avg_energy_score']:.2f}/100")
        print(f"  Certificates: {aggregate['avg_certificate_score']:.2f}/100")
        print(f"  Jobs: {aggregate['avg_job_score']:.2f}/100")
        print(f"  Components: {aggregate['avg_component_score']:.2f}/100")
        print(f"  Overall: {aggregate['avg_overall_score']:.2f}/100")
        print()
        print("Best Performers:")
        print(f"  Best maintenance: Schedule {aggregate['best_maintenance_schedule']} ({aggregate['best_maintenance_score']:.2f})")
        print(f"  Best turnaround: Schedule {aggregate['best_turnaround_schedule']} ({aggregate['best_turnaround_score']:.2f})")
        print(f"  Best energy: Schedule {aggregate['best_energy_schedule']} ({aggregate['best_energy_score']:.2f})")
        print(f"  Best overall: Schedule {aggregate['best_overall_schedule']} ({aggregate['best_overall_score']:.2f})")
        print()
        print("Constraint Violations:")
        print(f"  Avg maintenance overdue: {aggregate['avg_overdue_maintenance']:.1f}")
        print(f"  Avg turnaround violations: {aggregate['avg_turnaround_violations']:.1f}")
        print(f"  Avg energy violations: {aggregate['avg_energy_violations']:.1f}")
        print(f"  Avg certificate expired: {aggregate['avg_expired_certs']:.1f}")
        print(f"  Avg critical jobs: {aggregate['avg_critical_jobs']:.1f}")
        print(f"  Avg critical components: {aggregate['avg_critical_components']:.1f}")
        print()
    
    # Save to file in benchmark_output/ at project root
    if output_dir is None:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
    
    # Extract scores
    maint_scores = [r['metrics']['maintenance']['score'] for r in results]
    turnaround_scores = [r['metrics']['turnaround']['score'] for r in results]
    energy_scores = [r['metrics']['energy']['score'] for r in results]
    cert_scores = [r['metrics']['certificates']['score'] for r in results]
    job_scores = [r['metrics']['jobs']['score'] for r in results]
    comp_scores = [r['metrics']['components']['score'] for r in results]
    overall_scores = [r['metrics']['overall_constraint_score'] for r in results]
    
    # Extract violation counts
    overdue_maint = [r['metrics']['maintenance']['overdue_count'] for r in results]
    turnaround_viol = [r['metrics']['turnaround']['violations'] for r in results]
    energy_viol = [r['metrics']['energy']['violations'] for r in results]
    expired_certs = [r['metrics']['certificates']['expired'] for r in results]
    critical_jobs = [r['metrics']['jobs']['critical_jobs'] for r in results]
    critical_comps = [r['metrics']['components']['critical'] for r in results]
    
    analysis_times = [r['analysis_time_ms'] for r in results]
    
    # Find best performers
    best_maint_idx = maint_scores.index(max(maint_scores))
    best_turnaround_idx = turnaround_scores.index(max(turnaround_scores))
    best_energy_idx = energy_scores.index(max(energy_scores))
    best_overall_idx = overall_scores.index(max(overall_scores))
    
    return {
        'avg_maintenance_score': round(sum(maint_scores) / len(maint_scores), 2),
        'avg_turnaround_score': round(sum(turnaround_scores) / len(turnaround_scores), 2),
        'avg_energy_score': round(sum(energy_scores) / len(energy_scores), 2),
        'avg_certificate_score': round(sum(cert_scores) / len(cert_scores), 2),
        'avg_job_score': round(sum(job_scores) / len(job_scores), 2),
        'avg_component_score': round(sum(comp_scores) / len(comp_scores), 2),
        'avg_overall_score': round(sum(overall_scores) / len(overall_scores), 2),
        
        'best_maintenance_schedule': best_maint_idx + 1,
        'best_maintenance_score': round(max(maint_scores), 2),
        'best_turnaround_schedule': best_turnaround_idx + 1,
        'best_turnaround_score': round(max(turnaround_scores), 2),
        'best_energy_schedule': best_energy_idx + 1,
        'best_energy_score': round(max(energy_scores), 2),
        'best_overall_schedule': best_overall_idx + 1,
        'best_overall_score': round(max(overall_scores), 2),
        
        'avg_overdue_maintenance': round(sum(overdue_maint) / len(overdue_maint), 2),
        'avg_turnaround_violations': round(sum(turnaround_viol) / len(turnaround_viol), 2),
        'avg_energy_violations': round(sum(energy_viol) / len(energy_viol), 2),
        'avg_expired_certs': round(sum(expired_certs) / len(expired_certs), 2),
        'avg_critical_jobs': round(sum(critical_jobs) / len(critical_jobs), 2),
        'avg_critical_components': round(sum(critical_comps) / len(critical_comps), 2),
        
        'avg_analysis_time_ms': round(sum(analysis_times) / len(analysis_times), 2)
    }


if __name__ == "__main__":
    print("Constraint Satisfaction Benchmark Module")
    print("Import and use run_constraint_benchmark() to analyze schedules")
