---
title: Train Schedule Optimization
emoji: 🐨
colorFrom: blue
colorTo: pink
sdk: docker
pinned: false
license: cc-by-4.0
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

Do any change below this, don't change anything in readme above this. Above for hosting the api on HF Space

# Approach

## Overview

The `greedyOptim` module is our scheduler for **Metro Trainset Optimization**. At a high level, it decides which trainsets should run in service, which should stay on standby, and which must go to maintenance—while trying to keep operations safe and balanced.

## Problem Statement

Given a fleet of metro trainsets with varying health conditions, maintenance requirements, mileage statistics, and operational constraints, the system must:

- Assign exactly `N` trainsets to revenue service (default: 20)
- Maintain at least `M` trainsets in standby for backup (default: 2)
- Send remaining trainsets to maintenance
- Assign service trainsets to specific service blocks (trips)
- Balance multiple competing objectives
- Respect hard constraints (safety, certificates, maintenance)

## Solution Approach

### 1. Multi-Objective Optimization Framework

The system optimizes trainset assignments using a **weighted fitness score** that balances:

| Objective | Description | Weight Priority |
|-----------|-------------|-----------------|
| **Service Availability** | Maximize number of operational trains | Highest (5.0) |
| **Constraint Compliance** | Penalize constraint violations | Critical (10.0) |
| **Mileage Balance** | Distribute workload evenly across fleet | Medium (1.5) |
| **Maintenance Cost** | Optimize maintenance scheduling | Medium (1.0) |
| **Branding Compliance** | Meet advertising contract obligations | Low (0.2) |

### 2. Constraint System

#### Hard Constraints (Must Pass for Service Assignment)

- **Certificate Validity**: All safety certificates must be valid and not expired
- **Critical Maintenance**: No critical/open job cards pending
- **Component Health**: No critical component wear (wear level > 90%)
- **Maintenance Status**: Avoid putting trainsets with overdue maintenance into service (penalized strongly)

#### Soft Constraints (Optimization Targets)

- Minimum standby trainsets: ≥ 2
- Required service trainsets: exactly 20
- Mileage balance across fleet
- Branding coverage when contracts exist (nice-to-have)

### 3. Optimization Algorithms

The module provides **10 different optimization methods** categorized into three types:

#### A. Metaheuristic Algorithms

1. **Genetic Algorithm (GA)** - Default method
   - Population-based evolutionary optimization
   - Features: Tournament selection, two-point crossover, adaptive mutation
   - Parameters: Population size (100), Generations (200), Mutation rate (0.1)
   - Smart seeding with 50% constraint-aware initial solutions

2. **CMA-ES (Covariance Matrix Adaptation Evolution Strategy)**
   - Advanced evolution strategy using covariance matrix adaptation
   - Self-adaptive step-size mechanism
   - Particularly effective for continuous optimization problems

3. **Particle Swarm Optimization (PSO)**
   - Swarm intelligence-based approach
   - Particles move through solution space influenced by personal and global best
   - Parameters: Inertia weight (0.7), Cognitive/Social parameters (1.5)

4. **Simulated Annealing (SA)**
   - Probabilistic optimization inspired by metallurgy
   - Temperature-based acceptance of worse solutions to escape local optima
   - Exponential cooling schedule

#### B. Exact/Constraint Programming Methods

5. **CP-SAT (Constraint Programming - SAT Solver)**
   - Google OR-Tools constraint programming solver
   - Finds optimal or near-optimal solutions with time limits
   - Enforces hard constraints exactly
   - Configurable time limit (default: 300s)

6. **MIP (Mixed Integer Programming)**
   - Linear programming with integer variables
   - Guarantees optimal solutions for smaller problems
   - OR-Tools based implementation

#### C. Hybrid/Advanced Methods

7. **NSGA-II Multi-Objective**
   - Non-dominated Sorting Genetic Algorithm II
   - True Pareto-front multi-objective optimization
   - Fast non-dominated sorting with crowding distance
   - Weighted dominance comparison prioritizing service availability

8. **Adaptive Algorithm Selection**
   - Dynamically switches between GA, PSO, and SA
   - Monitors convergence rates to select best-performing method
   - Automatically adapts to problem characteristics

9. **Ensemble Optimizer**
   - Runs multiple algorithms in parallel (GA, CMA-ES, PSO, SA)
   - Selects best solution across all methods
   - ThreadPoolExecutor for parallel execution
   - Combines strengths of different approaches

10. **Auto-Tuned GA**
    - Hyperparameter optimization using Bayesian techniques
    - Automatically tunes: population size, mutation rate, crossover rate
    - Tests multiple configurations to find optimal parameters

### 4. Solution Encoding

#### Trainset Assignment Representation

Solutions are encoded as integer arrays where each element represents a trainset's assignment:

```
Solution: [0, 2, 0, 1, 2, 0, ...]
          │  │  │  │  │  │
          │  │  │  │  │  └─ Trainset_6: Service (0)
          │  │  │  │  └──── Trainset_5: Maintenance (2)
          │  │  │  └─────── Trainset_4: Standby (1)
          │  │  └────────── Trainset_3: Service (0)
          │  └───────────── Trainset_2: Maintenance (2)
          └──────────────── Trainset_1: Service (0)

Encoding: 0 = Service, 1 = Standby, 2 = Maintenance
```

#### Block Assignment Representation

Service trainsets are further assigned to specific **service blocks** (operational trips):

```
Block Solution: [3, 7, 3, 12, 7, ...]
                 │  │  │   │  │
                 │  │  │   │  └─ Block_5 assigned to Trainset_7
                 │  │  │   └──── Block_4 assigned to Trainset_12
                 │  │  └──────── Block_3 assigned to Trainset_3
                 │  └─────────── Block_2 assigned to Trainset_7
                 └────────────── Block_1 assigned to Trainset_3
```

### 5. Service Block Generation

The system generates realistic **service blocks** representing train operations:

#### Route Configuration
- Loads station data from JSON configuration (`data/metro_stations.json`)
- Supports flexible route parameters:
  - Route length and terminals
  - Average speed and travel times
  - Dwell times and turnaround times
  - Peak/off-peak operational parameters

#### Block Types
- **Morning Peak** (07:00-10:00): 6-minute headways
- **Midday Off-Peak** (10:00-17:00): 15-minute headways
- **Evening Peak** (17:00-21:00): 6-minute headways
- **Late Evening** (21:00-23:00): 15-minute headways

#### Trip Structure
Each service block contains:
- Block ID and departure time
- Origin and destination terminals
- Trip count and estimated kilometers
- Detailed trip breakdown with:
  - Individual trip ID and number
  - Direction (UP/DOWN)
  - All station stops with arrival/departure times
  - Platform assignments
  - Distance from origin

### 6. Fitness Evaluation

The fitness function combines multiple objectives using weighted aggregation:

```
Fitness = Σ (weight_i × objective_i) + penalties

Where:
- Lower fitness = better solution
- Penalties heavily penalize constraint violations
- Objectives are normalized to 0-100 scale
```

#### Objective Calculations:

1. **Service Availability**
   - Base: `(actual_service / required_service) × 100`
   - Bonus: Extra trains beyond requirement (capped at 50% bonus)
   - Penalty: 200 points per missing service train

2. **Mileage Balance**
   - `100 - min(std_dev(mileages) / 1000, 100)`
   - Lower standard deviation = better balance

3. **Branding Compliance**
   - Average compliance across branded trainsets
   - `actual_exposure / target_exposure` per contract

4. **Maintenance Cost**
   - Rewards sending overdue trainsets to maintenance
   - Rewards using trainsets with recent service

5. **Constraint Penalty**
   - 200 points per constraint violation per trainset
   - Applied to service trainsets failing hard constraints

### 7. Genetic Algorithm Details

The default GA implementation includes several advanced features:

#### Population Initialization
- **Smart Seeding (50%)**: Constraint-aware initial solutions
  - Respects required service/standby counts
  - Balances assignments intelligently
- **Random Initialization (50%)**: Exploration diversity

#### Genetic Operators
- **Selection**: Tournament selection (size=5)
- **Crossover**: Two-point crossover (rate=0.8)
- **Mutation**: Random gene flipping (rate=0.1)
- **Elitism**: Preserves top 5 solutions

#### Repair Mechanisms
- Post-crossover/mutation repair ensures:
  - Sufficient service trainsets
  - Minimum standby count
  - Valid block assignments (only to service trains)

### 8. OR-Tools Integration

For problems requiring exact solutions or strong optimality guarantees:

#### CP-SAT Features
- Binary decision variables for each trainset
- Exact constraint satisfaction
- Multi-objective optimization via weighted sum
- Branding constraints: minimum brand representation
- Time-limited search with incumbent tracking

#### MIP Features
- Linear programming relaxation
- Integer variable constraints
- Branch-and-bound search
- Optimal solution guarantees (if found within time limit)

### 9. Evaluation System

`TrainsetSchedulingEvaluator` provides:

#### Constraint Checking
- Per-trainset constraint validation
- Lookup optimization using dictionaries
- Handles missing/malformed data gracefully

#### Data Structures
- Fast lookup maps for:
  - Trainset status
  - Fitness certificates
  - Job cards
  - Component health
  - Branding contracts
  - Maintenance schedules

#### Normalization
- Certificate status normalization (ISSUED→Valid, EXPIRED→Expired)
- Component status normalization (EXCELLENT→Good, CRITICAL→Critical)
- Operational status normalization

### 10. Key Features

- **Modular Design**: Separate modules for core logic, optimizers, scheduling, routing
- **Extensible**: Easy to add new optimization algorithms
- **Configurable**: Comprehensive `OptimizationConfig` class
- **Type-Safe**: Dataclasses with type hints throughout
- **Error Handling**: Graceful degradation with fallback solutions
- **Logging**: Detailed progress tracking and debugging
- **Testing**: Comprehensive test suites for each component
- **Performance**: Parallel execution support for ensemble methods
- **Benchmarking**: Built-in comparison tools for algorithm evaluation

### 11. Output Format

The system produces a structured `OptimizationResult` containing:

```python
{
    'selected_trainsets': ['T001', 'T003', ...],      # Service assignments
    'standby_trainsets': ['T004', 'T007', ...],       # Standby assignments
    'maintenance_trainsets': ['T002', 'T005', ...],   # Maintenance assignments
    'objectives': {                                    # Objective scores
        'service_availability': 110.5,
        'mileage_balance': 87.3,
        'maintenance_cost': 92.1,
        'branding_compliance': 78.4,
        'constraint_penalty': 0.0
    },
    'fitness_score': 145.2,                           # Overall fitness
    'explanation': {                                   # Per-trainset reasons
        'T001': '✓ Fit for service',
        'T002': '⚠ Critical maintenance jobs pending'
    },
    'service_block_assignments': {                    # Block assignments
        'T001': ['BLK_001', 'BLK_015', 'BLK_032'],
        'T003': ['BLK_002', 'BLK_016']
    }
}
```

### 12. Usage Example

```python
from greedyOptim import optimize_trainset_schedule, OptimizationConfig

# Configure optimization
config = OptimizationConfig(
    required_service_trains=20,
    min_standby=2,
    population_size=100,
    generations=200,
    optimize_block_assignment=True
)

# Run optimization
result = optimize_trainset_schedule(
    data=metro_data,
    method='ga',  # or 'cmaes', 'pso', 'sa', 'cp-sat', 'nsga2', etc.
    config=config
)

# Access results
print(f"Service trainsets: {len(result.selected_trainsets)}")
print(f"Fitness score: {result.fitness_score:.2f}")
```

### 13. Performance Characteristics

| Method | Speed | Solution Quality | Best For |
|--------|-------|------------------|----------|
| GA | Fast | Good | General-purpose, default choice |
| CMA-ES | Medium | Very Good | Continuous optimization |
| PSO | Fast | Good | Quick solutions |
| SA | Medium | Good | Escaping local optima |
| CP-SAT | Slow | Optimal/Near-optimal | When optimality matters |
| MIP | Slow | Optimal | Small problems |
| NSGA-II | Medium | Pareto-optimal | Multi-objective trade-offs |
| Adaptive | Medium | Very Good | Unknown problem types |
| Ensemble | Slow | Best | When time permits |
| Auto-Tune | Very Slow | Optimal GA config | Repeated similar problems |

---

## Architecture

```
greedyOptim/
├── core/                      # Core models and utilities
│   ├── models.py             # Data classes (configs, results, constraints)
│   ├── utils.py              # Helper functions (encoding, normalization)
│   └── error_handling.py     # Exception handling
│
├── optimizers/                # Optimization algorithms
│   ├── base_optimizer.py     # Abstract base class
│   ├── genetic_algorithm.py  # GA implementation
│   ├── advanced_optimizers.py # CMA-ES, PSO, SA
│   ├── ortools_optimizers.py # CP-SAT, MIP
│   └── hybrid_optimizers.py  # NSGA-II, Adaptive, Ensemble
│
├── scheduling/                # Scheduling logic
│   ├── evaluator.py          # Fitness evaluation & constraints
│   ├── scheduler.py          # Main optimization interface
│   ├── schedule_generator.py # Schedule output generation
│   └── service_blocks.py     # Service block generation
│
├── routing/                   # Route management
│   └── station_loader.py     # Load station/route configuration
│
├── balance.py                 # Load balancing utilities
└── base.py                    # Legacy API compatibility
```

