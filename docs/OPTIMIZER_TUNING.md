# Optimizer Tuning Guide

This document describes the tuning changes made to optimize service train selection across all optimization methods.

## Problem Statement

The optimizers were initially selecting too few trains for service (as low as 1-13 trains), when 21-22 healthy trainsets were available. This was due to:

1. **Synthetic data issues**: Only 12% of trainsets were healthy
2. **Fitness function priorities**: Branding compliance weighted too heavily
3. **NSGA-II specific issues**: No elitism, random initialization, equal objective weights

## Changes Made

### 1. Synthetic Data Generator (`DataService/enhanced_generator.py`)

**Problem**: Only 3/25 trainsets (12%) were healthy enough for service.

**Solution**: Increased healthy trainset ratio to 85%.

```python
# Before: Equal probability of healthy/unhealthy components
# After: 85% healthy trainsets with wear capped at 60% of threshold

healthy_trainset_count = int(self.num_trainsets * 0.85)  # 85% healthy
max_healthy_wear = comp_info["wear_threshold"] * 0.60    # 60% cap
```

### 2. Fitness Function Weights (`greedyOptim/evaluator.py`)

**Problem**: Branding compliance was weighted too heavily, causing optimizers to prefer fewer trains with better branding over more trains in service.

**Solution**: Rebalanced weights to prioritize operational needs.

| Objective | Old Weight | New Weight | Priority |
|-----------|------------|------------|----------|
| service_availability | 2.0 | **5.0** | HIGHEST |
| constraint_penalty | 5.0 | **10.0** | CRITICAL |
| mileage_balance | 1.5 | 1.5 | Medium |
| maintenance_cost | 1.0 | 1.0 | Medium |
| branding_compliance | 1.5 | **0.2** | LOW |

**Buffer Bonus**: Added bonus for trains beyond minimum requirement:
```python
# Reward having more than minimum trains for smooth operations
buffer = max(0, len(service_trains) - self.config.required_service_trains)
objectives['service_availability'] += buffer * 3.0  # Bonus per extra train
```

### 3. NSGA-II Optimizer (`greedyOptim/hybrid_optimizers.py`)

#### 3.1 Weighted Dominance Comparison

**Problem**: All objectives treated equally in Pareto dominance.

**Solution**: Apply weights to objectives before dominance comparison.

```python
self.objective_weights = {
    'service_availability': 5.0,   # HIGHEST
    'mileage_balance': 1.5,
    'maintenance_cost': 1.0,
    'branding_compliance': 0.2,    # LOW
    'constraint_penalty': 10.0     # CRITICAL
}

# In dominates():
obj1 = [
    -solution1['service_availability'] * w['service_availability'],
    # ... other objectives with weights
]
```

#### 3.2 Smart Initialization

**Problem**: Random initialization created many invalid solutions.

**Solution**: Seed population with constraint-aware solutions.

```python
def _create_smart_initial_solution(self):
    solution = np.zeros(self.n_genes, dtype=int)  # All service
    for i, ts_id in enumerate(self.evaluator.trainsets):
        valid, _ = self.evaluator.check_hard_constraints(ts_id)
        if not valid:
            solution[i] = 2  # Maintenance for invalid
        elif standby_count < self.config.min_standby:
            solution[i] = 1  # Reserve for standby
    return solution

# Mix: 20% smart solutions + 80% biased random
```

#### 3.3 Biased Random Solutions

**Problem**: Equal probability for service/depot/maintenance (33% each).

**Solution**: Bias toward service assignment.

```python
# Initial population: 65% service, 20% depot, 15% maintenance
solution = np.random.choice([0, 1, 2], size=n, p=[0.65, 0.20, 0.15])

# Mutation: 55% service, 30% depot, 15% maintenance
child[i] = np.random.choice([0, 1, 2], p=[0.55, 0.30, 0.15])
```

#### 3.4 Elitism with Combined Population

**Problem**: Offspring replaced parents completely, losing good solutions.

**Solution**: Combine parents and offspring, then select best via non-dominated sorting.

```python
# Combine parents and offspring
combined_population = new_population + offspring

# Re-evaluate and sort
combined_fronts = self.fast_non_dominated_sort(combined_objectives)

# Select best from combined (preserves good solutions)
for front in combined_fronts:
    # Add to next generation up to population_size
```

#### 3.5 Service-Prioritized Final Selection

**Problem**: Random selection from Pareto front.

**Solution**: Explicitly select solution with highest service availability.

```python
# Among zero-penalty solutions, choose highest service_availability
valid_solutions = [(i, sol, obj) for i, (sol, obj) in enumerate(best_solutions)
                  if obj.get('constraint_penalty', 0) == 0]

if valid_solutions:
    best_idx = max(valid_solutions, 
                  key=lambda x: x[2].get('service_availability', 0))[0]
```

## Results

### Before Tuning
| Method | Service Trains | Notes |
|--------|---------------|-------|
| GA | 1-13 | Poor due to unhealthy data |
| PSO | 1-13 | Same issue |
| SA | 1-13 | Same issue |
| CMA-ES | 1-13 | Same issue |
| NSGA2 | 12-13 | Worst performer |

### After Tuning
| Method | Service Trains | Notes |
|--------|---------------|-------|
| GA | 21-22 | Excellent |
| SA | 21-22 | Excellent |
| CMA-ES | 19-20 | Good |
| NSGA2 | 21-22 | **Fixed!** |
| PSO | 15-18 | Acceptable |

## Recommendations

1. **Use GA or SA** for best results in single-objective optimization
2. **Use NSGA2** when you need to explore trade-offs between objectives
3. **PSO** may need further tuning for this problem domain
4. **CMA-ES** provides good balance between quality and exploration

## Configuration Parameters

Recommended settings for Kochi Metro (25 trainsets, 106 blocks):

```python
config = OptimizationConfig(
    required_service_trains=15,    # Minimum for service
    min_standby=2,                 # Safety buffer
    population_size=50,            # Larger = better but slower
    generations=100,               # More = better convergence
    mutation_rate=0.1,             # Standard
    crossover_rate=0.8,            # Standard
    optimize_block_assignment=True # Enable block optimization
)
```
