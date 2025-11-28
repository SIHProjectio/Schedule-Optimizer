# Fitness Score Methodology

This document explains the fitness score calculation methodology used in the train scheduling optimization system.

## Overview

The fitness function is designed for **minimization** - more negative values indicate better solutions. This is standard practice for optimization algorithms (GA, PSO, CMA-ES, SA, NSGA-II) where we minimize an objective function.

## Solution Encoding

Each trainset is assigned one of three states:
- `0` = **Service** (Revenue Service)
- `1` = **Standby** (Available as backup)
- `2` = **Maintenance** (Under repair/inspection)

## Multi-Objective Evaluation

The fitness function combines 5 objectives with different weights and priorities:

### Objective 1: Service Availability (Weight: 5.0) - HIGHEST PRIORITY

**Goal**: Maximize the number of trains in revenue service for smooth operations.

**Calculation**:
```
If num_service < required_service_trains:
    service_availability = (num_service / required) * 100
    constraint_penalty += (required - num_service) * 200  # Heavy penalty
Else:
    bonus_trains = num_service - required
    max_bonus = required * 0.5  # Up to 50% more
    bonus_score = min(bonus_trains / max_bonus, 1.0) * 20.0
    service_availability = 100.0 + bonus_score
```

**Score Range**: 0-120 (100 = meeting minimum, 120 = 50% buffer)

**Rationale**: Having more trains than the minimum ensures:
- Coverage during peak hours
- Buffer for unexpected breakdowns
- Smoother operations throughout the day

---

### Objective 2: Mileage Balance (Weight: 1.5) - MEDIUM PRIORITY

**Goal**: Balance fleet wear by distributing service across trainsets with different mileages.

**Calculation**:
```
mileages = [total_mileage_km for each service train]
std_dev = standard_deviation(mileages)
mileage_balance = 100.0 - min(std_dev / 1000.0, 100.0)
```

**Score Range**: 0-100 (higher = more balanced)

**Rationale**: 
- Prevents overuse of specific trainsets
- Extends overall fleet lifespan
- Ensures even maintenance distribution

---

### Objective 3: Maintenance Cost (Weight: 1.0) - MEDIUM PRIORITY

**Goal**: Avoid scheduling trainsets with overdue maintenance.

**Calculation**:
```
maint_cost = 0
for each service train:
    if maintenance_status == 'Overdue':
        maint_cost += 50.0
maintenance_cost = 100.0 - min(maint_cost, 100.0)
```

**Score Range**: 0-100 (higher = lower maintenance risk)

**Rationale**:
- Reduces breakdown risk during service
- Ensures safety compliance
- Prevents cascading maintenance issues

---

### Objective 4: Branding Compliance (Weight: 0.2) - LOW PRIORITY

**Goal**: Meet advertising contract obligations (nice-to-have).

**Calculation**:
```
for each service train with branding contract:
    target = daily_target_hours
    actual = actual_exposure_hours / 30  # Daily average
    compliance = min(actual / target, 1.0)
branding_compliance = mean(compliance_scores) * 100
```

**Score Range**: 0-100 (higher = better contract compliance)

**Rationale**:
- Secondary consideration after operational needs
- Revenue generation from advertising
- Contract obligation fulfillment

---

### Objective 5: Constraint Penalty (Weight: 10.0) - CRITICAL

**Goal**: Heavily penalize hard constraint violations.

**Hard Constraints Checked**:

1. **Certificate Validity**
   - All fitness certificates must be valid and not expired
   - Penalty: +200 per violation

2. **Critical Jobs**
   - No open critical maintenance jobs allowed
   - Penalty: +200 per violation

3. **Component Health**
   - No components with Warning/Critical status AND wear > 90%
   - Penalty: +200 per violation

4. **Minimum Service Requirement**
   - Must meet `required_service_trains` (default: 15)
   - Penalty: +200 per missing train

5. **Minimum Standby Requirement**
   - Must meet `min_standby` (default: 2)
   - Penalty: +50 per missing standby

**Score Range**: 0 (best) to 1000+ (many violations)

---

## Final Fitness Calculation

```python
fitness = (
    -service_availability * 5.0 +    # Maximize (negative to minimize)
    -mileage_balance * 1.5 +          # Maximize
    -maintenance_cost * 1.0 +         # Maximize
    -branding_compliance * 0.2 +      # Maximize
    constraint_penalty * 10.0         # Minimize (already positive)
)
```

### Example Calculation(don't add section below this)

For a good solution with 21 service trains (min required: 15):

| Objective | Value | Weight | Contribution |
|-----------|-------|--------|--------------|
| service_availability | 117.1 | -5.0 | -585.7 |
| mileage_balance | 85.0 | -1.5 | -127.5 |
| maintenance_cost | 100.0 | -1.0 | -100.0 |
| branding_compliance | 90.0 | -0.2 | -18.0 |
| constraint_penalty | 0.0 | +10.0 | 0.0 |
| **Total Fitness** | | | **-831.2** |

More negative = better solution.

---

## Interpreting Fitness Scores

| Fitness Score Range | Interpretation |
|---------------------|----------------|
| -1200 to -1300 | Excellent (max service, no violations) |
| -1000 to -1200 | Very Good (high service, minimal issues) |
| -800 to -1000 | Good (meets requirements) |
| -500 to -800 | Acceptable (some objectives compromised) |
| -500 to 0 | Poor (significant issues) |
| > 0 (positive) | Bad (constraint violations) |

---

## Weight Rationale

The weights were tuned based on operational priorities:

1. **Service Availability (5.0)**: Core mission - maximize trains running
2. **Constraint Penalty (10.0)**: Safety critical - violations are unacceptable
3. **Mileage Balance (1.5)**: Long-term fleet health
4. **Maintenance Cost (1.0)**: Risk mitigation
5. **Branding Compliance (0.2)**: Nice-to-have, not operational

---

## Configuration Parameters

```python
OptimizationConfig(
    required_service_trains=15,  # Minimum trains for service
    min_standby=2,               # Minimum backup trains
    population_size=50,          # GA/PSO population
    generations=100,             # Evolution iterations
    mutation_rate=0.1,           # Exploration rate
    crossover_rate=0.8,          # Exploitation rate
)
```

---

## Summary

The fitness score methodology prioritizes:

1. **Safety First**: Hard constraints cannot be violated
2. **Operational Excellence**: Maximize trains in service with buffer
3. **Fleet Longevity**: Balance mileage across trainsets
4. **Cost Efficiency**: Avoid overdue maintenance risks
5. **Revenue Generation**: Meet branding contracts when possible

A score of **-1200** indicates an excellent schedule with 21+ trains in service, zero constraint violations, and well-balanced fleet utilization.
