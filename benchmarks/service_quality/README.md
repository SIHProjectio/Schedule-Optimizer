# Service Quality Benchmarking

Comprehensive benchmarking system for evaluating service quality metrics in metro train scheduling, focusing on **headway consistency**, **passenger wait times**, and **service coverage**.

## Overview

This module analyzes how well train schedules meet passenger service quality expectations by measuring:

1. **Headway Consistency**: How consistent are the intervals between trains?
2. **Passenger Wait Time**: How long do passengers wait on average and at maximum?
3. **Service Coverage**: What percentage of operational hours have adequate train frequency?

## Key Metrics

### 1. Headway Consistency

**Headway** = Time interval between consecutive train departures

**Metrics:**
- **Mean Headway** (peak & off-peak): Average minutes between trains
- **Standard Deviation**: Variability in headways
- **Coefficient of Variation (CV)**: Normalized measure of consistency (std/mean)
  - CV < 0.2 = Excellent consistency
  - CV 0.2-0.3 = Good consistency
  - CV > 0.3 = Poor consistency

**Why It Matters:**
- Consistent headways improve passenger experience
- Reduces crowding on platforms
- Makes service predictable for passengers

**Kochi Metro Standards:**
- **Peak hours (7-10 AM, 5-8 PM)**: Target 7.5 minutes
- **Off-peak hours**: Target 15 minutes

### 2. Passenger Wait Time

**Average Wait Time** = Headway / 2 (for random passenger arrival)

**Metrics:**
- **Average Wait Time** (peak & off-peak): Expected wait for random arrival
- **Maximum Wait Time**: Worst-case wait (longest headway)
- **Reduction vs Baseline**: Improvement over poor service (baseline: 12min peak, 25min off-peak)

**Why It Matters:**
- Directly impacts passenger satisfaction
- Longer waits = lost ridership
- Peak hour waits are especially critical

### 3. Service Coverage

**Coverage** = Percentage of operational hours with adequate train frequency

**Metrics:**
- **Operational Hours**: Total service hours (6 AM - 10 PM = 16 hours)
- **Peak Hours Covered**: Hours with ≥4 trains/hour (every 15 min)
- **Off-peak Hours Covered**: Hours with ≥2 trains/hour (every 30 min)
- **Coverage Percentage**: % of hours meeting frequency targets
- **Service Gaps**: Number of headways > 30 minutes (unacceptable gaps)

**Why It Matters:**
- Ensures consistent service throughout the day
- Avoids leaving passengers stranded
- Maintains public trust in the system

## Scoring System

Each metric is scored 0-100:

### Headway Consistency Score
```
Score = 100 - (CV * 200)
```
- CV 0.1 → 80 points
- CV 0.3 → 40 points
- CV 0.5 → 0 points

### Wait Time Score
```
Score = min(100, (target_wait / actual_wait) * 100)
```
- At target wait time → 100 points
- 2x target → 50 points
- 4x target → 25 points

### Coverage Score
```
Score = Coverage Percentage (0-100)
```
- Direct percentage of hours with adequate frequency

### Overall Quality Score
```
Overall = Headway_Score * 0.4 + Wait_Score * 0.3 + Coverage_Score * 0.3
```
- Headway consistency weighted highest (40%)
- Wait time and coverage each 30%

## Usage

### Basic Usage

```python
from benchmarks.service_quality import run_service_quality_benchmark

# Run benchmark on schedules
results = run_service_quality_benchmark(
    schedules=[schedule1, schedule2, schedule3],
    output_file="service_quality_results.json",
    verbose=True
)

print(f"Overall quality: {results['aggregate_metrics']['avg_overall_score']:.1f}/100")
```

### Analyzing Individual Schedule

```python
from benchmarks.service_quality import ServiceQualityAnalyzer

analyzer = ServiceQualityAnalyzer()
metrics = analyzer.analyze_schedule(schedule)

print(f"Peak headway: {metrics.peak_headway_mean:.1f} ± {metrics.peak_headway_std:.1f} min")
print(f"Peak wait time: {metrics.avg_wait_time_peak:.1f} min")
print(f"Service coverage: {metrics.service_coverage_percent:.1f}%")
print(f"Overall score: {metrics.overall_quality_score:.1f}/100")
```

### Running Example Benchmark

```bash
cd benchmarks/service_quality
python example_service_benchmark.py
```

This will:
1. Generate 5 sample schedules using different optimization methods
2. Analyze service quality for each
3. Compare results and identify best performers
4. Save detailed results to JSON

## Schedule Data Requirements

Schedules must include **service blocks** for trains in revenue service:

```python
{
    "trainsets": [
        {
            "trainset_id": "KMRL-01",
            "status": "REVENUE_SERVICE",
            "service_blocks": [
                {
                    "block_id": "BLK-M-1",
                    "departure_time": "07:00",  # HH:MM format
                    "origin": "Aluva",
                    "destination": "Pettah",
                    "trip_count": 8,
                    "estimated_km": 205
                },
                {
                    "block_id": "BLK-E-1",
                    "departure_time": "17:15",
                    "origin": "Pettah",
                    "destination": "Aluva",
                    "trip_count": 7,
                    "estimated_km": 179
                }
            ]
        }
    ]
}
```

**Service blocks** define when trains begin their operating shifts, creating the departure timeline needed for headway analysis.

## Output Format

### JSON Results

```json
{
  "benchmark_info": {
    "timestamp": "2025-11-23T10:30:00",
    "total_schedules": 5,
    "total_time_seconds": 2.45,
    "avg_analysis_time_ms": 489.2
  },
  "aggregate_metrics": {
    "avg_headway_score": 75.3,
    "avg_wait_score": 82.1,
    "avg_coverage_score": 68.5,
    "avg_overall_score": 75.9,
    "best_headway_schedule": 2,
    "best_wait_schedule": 3,
    "best_coverage_schedule": 1,
    "best_overall_schedule": 2,
    "avg_peak_headway": 8.2,
    "avg_peak_wait": 4.1,
    "avg_coverage_percent": 68.5,
    "avg_service_gaps": 2.4
  },
  "individual_results": [...]
}
```

### Individual Schedule Results

Each schedule gets detailed metrics:

```json
{
  "schedule_id": 1,
  "analysis_time_ms": 456.78,
  "metrics": {
    "headway_consistency": {
      "peak_mean_minutes": 7.8,
      "peak_std_minutes": 1.2,
      "peak_cv": 0.154,
      "offpeak_mean_minutes": 15.3,
      "offpeak_std_minutes": 2.1,
      "offpeak_cv": 0.137,
      "score": 73.2
    },
    "wait_times": {
      "avg_wait_peak_minutes": 3.9,
      "max_wait_peak_minutes": 10.2,
      "avg_wait_offpeak_minutes": 7.7,
      "max_wait_offpeak_minutes": 18.4,
      "reduction_vs_baseline_percent": 35.2,
      "score": 85.6
    },
    "service_coverage": {
      "operational_hours": 16.0,
      "peak_hours_covered": 5.5,
      "offpeak_hours_covered": 7.2,
      "coverage_percent": 79.4,
      "peak_coverage_percent": 91.7,
      "service_gaps": 2,
      "score": 79.4
    },
    "overall_quality_score": 78.5
  }
}
```

## Interpretation Guide

### Excellent Service (Score 80-100)
- Peak headway: 7-8 minutes with CV < 0.15
- Peak wait time: < 4 minutes average
- Coverage: > 90% of hours with adequate frequency
- Few or no service gaps

**Example:** "Passengers experience consistent, reliable service throughout the day with minimal waiting"

### Good Service (Score 60-79)
- Peak headway: 8-10 minutes with CV 0.15-0.25
- Peak wait time: 4-5 minutes average
- Coverage: 70-90% of hours adequate
- Occasional service gaps

**Example:** "Service is generally reliable but passengers may experience occasional longer waits"

### Fair Service (Score 40-59)
- Peak headway: 10-15 minutes with CV 0.25-0.35
- Peak wait time: 5-7 minutes average
- Coverage: 50-70% of hours adequate
- Multiple service gaps

**Example:** "Service quality is inconsistent; passengers cannot rely on regular intervals"

### Poor Service (Score < 40)
- Peak headway: > 15 minutes or highly variable (CV > 0.35)
- Peak wait time: > 7 minutes average
- Coverage: < 50% of hours adequate
- Many service gaps

**Example:** "Unreliable service with long, unpredictable waits"

## Algorithm Comparison

Use this benchmark to compare optimization algorithms:

```python
methods = ['ga', 'pso', 'cmaes', 'sa', 'ensemble']
schedules = []

for method in methods:
    optimizer = TrainsetSchedulingOptimizer(data, config)
    result = optimizer.optimize(method=method)
    schedule = create_schedule_with_blocks(result)
    schedules.append(schedule)

results = run_service_quality_benchmark(schedules)
# See which algorithm produces best service quality
```

## Integration with Other Benchmarks

Combine with other benchmark modules:

```python
from benchmarks.fleet_utilization import run_fleet_utilization_benchmark
from benchmarks.service_quality import run_service_quality_benchmark

# Analyze both fleet efficiency and service quality
fleet_results = run_fleet_utilization_benchmark(schedules)
service_results = run_service_quality_benchmark(schedules)

# Find schedules that excel in both
best_fleet_id = fleet_results['aggregate_metrics']['best_overall_schedule']
best_service_id = service_results['aggregate_metrics']['best_overall_schedule']

if best_fleet_id == best_service_id:
    print("Schedule achieves both excellent fleet utilization and service quality!")
```

## Advanced Features

### Custom Standards

Modify target headways for different metro systems:

```python
analyzer = ServiceQualityAnalyzer()
analyzer.TARGET_PEAK_HEADWAY = 10.0  # 10 minutes instead of 7.5
analyzer.TARGET_OFFPEAK_HEADWAY = 20.0  # 20 minutes instead of 15
```

### Time-Aware Analysis

The analyzer automatically classifies peak vs. off-peak:
- **Peak hours**: 7-10 AM, 5-8 PM (6 hours total)
- **Off-peak hours**: All other operational hours (10 hours)

### Service Gap Detection

Automatically flags unacceptable gaps:
- Any headway > 30 minutes is flagged as a service gap
- Critical for identifying schedule deficiencies

## Performance

- **Analysis time**: ~500ms per schedule (depends on number of service blocks)
- **Memory usage**: Minimal (<10 MB per schedule)
- **Scalability**: Can analyze hundreds of schedules efficiently

## Troubleshooting

### No service blocks found
**Problem:** Schedule has trains in REVENUE_SERVICE but no service_blocks
**Solution:** Ensure service blocks are populated with departure times

### All metrics are zero
**Problem:** No departures extracted from schedule
**Solution:** Check that departure_time is in "HH:MM" format

### Low coverage score
**Problem:** Not enough trains scheduled throughout the day
**Solution:** Increase required_service_trains or adjust service block allocation

## References

- **Headway Analysis**: Transport for London service quality standards
- **Wait Time Models**: Passenger arrival models (random vs. informed)
- **Coverage Metrics**: APTA (American Public Transportation Association) guidelines
- **Kochi Metro**: KMRL operational standards and specifications

## Future Enhancements

1. **Real-time Integration**: Monitor actual headways vs. scheduled
2. **Demand-Based Analysis**: Weight service quality by ridership patterns
3. **Multi-Route Support**: Analyze different metro lines separately
4. **Weather Impact**: Adjust targets based on weather conditions
5. **Special Events**: Handle irregular demand patterns

---

For questions or issues, refer to the main project documentation or raise an issue in the repository.
