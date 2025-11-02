# Algorithms & Optimization Techniques

## Overview

This document describes all algorithms, optimization techniques, and machine learning models used in the Metro Train Scheduling Service.

---

## Table of Contents

1. [Machine Learning Algorithms](#machine-learning-algorithms)
2. [Optimization Algorithms](#optimization-algorithms)
3. [Hybrid Approach](#hybrid-approach)
4. [Feature Engineering](#feature-engineering)
5. [Performance Metrics](#performance-metrics)

---

## Machine Learning Algorithms

### Ensemble Learning Architecture

The system employs a **5-model ensemble** approach for schedule quality prediction:

#### 1. Gradient Boosting (Scikit-learn)
**Algorithm**: Sequential ensemble of weak learners (decision trees)

**Parameters**:
- `n_estimators`: 100 trees
- `learning_rate`: 0.001
- `loss function`: Least squares regression
- `max_depth`: Auto (unlimited)

**Strengths**:
- Excellent baseline performance
- Handles non-linear relationships well
- Robust to outliers

**Use Case**: Primary baseline model for schedule quality prediction

---

#### 2. Random Forest (Scikit-learn)
**Algorithm**: Bagging ensemble of decision trees

**Parameters**:
- `n_estimators`: 100 trees
- `max_features`: Auto (√n_features)
- `n_jobs`: -1 (parallel processing)
- `random_state`: 42

**Strengths**:
- Low variance through averaging
- Handles missing data well
- Feature importance ranking

**Use Case**: Robust predictions with feature importance insights

---

#### 3. XGBoost (Extreme Gradient Boosting)
**Algorithm**: Optimized distributed gradient boosting

**Parameters**:
- `n_estimators`: 100
- `learning_rate`: 0.001
- `objective`: reg:squarederror
- `tree_method`: Auto
- `verbosity`: 0

**Technical Details**:
- Uses second-order gradients (Newton-Raphson)
- L1/L2 regularization to prevent overfitting
- Parallel tree construction
- Cache-aware block structure

**Strengths**:
- Typically best single-model performance
- Fast training and prediction
- Built-in cross-validation

**Use Case**: High-performance predictions, often selected as best model

---

#### 4. LightGBM (Microsoft)
**Algorithm**: Gradient-based One-Side Sampling (GOSS) + Exclusive Feature Bundling (EFB)

**Parameters**:
- `n_estimators`: 100
- `learning_rate`: 0.001
- `boosting_type`: gbdt
- `verbose`: -1

**Technical Details**:
- **GOSS**: Keeps instances with large gradients, randomly samples small gradients
- **EFB**: Bundles mutually exclusive features to reduce dimensions
- Leaf-wise tree growth (vs level-wise)
- Histogram-based splitting

**Strengths**:
- Fastest training time
- Low memory usage
- Handles large datasets efficiently

**Use Case**: Fast iteration during development, efficient production inference

---

#### 5. CatBoost (Yandex)
**Algorithm**: Ordered boosting with categorical feature handling

**Parameters**:
- `iterations`: 100
- `learning_rate`: 0.001
- `loss_function`: RMSE
- `verbose`: False

**Technical Details**:
- **Ordered Boosting**: Prevents target leakage in gradient calculation
- **Symmetric Trees**: Balanced tree structure
- Native categorical feature support
- Minimal hyperparameter tuning needed

**Strengths**:
- Best out-of-the-box performance
- Robust to overfitting
- Excellent with categorical data

**Use Case**: Robust predictions with minimal tuning

---

### Ensemble Strategy

#### Weighted Voting
```python
# Weight calculation (performance-based)
weight_i = R²_score_i / Σ(R²_scores)

# Final prediction
prediction = Σ(weight_i × prediction_i)
```

**Example Weights**:
```json
{
  "xgboost": 0.215,      // Best performer
  "lightgbm": 0.208,
  "gradient_boosting": 0.195,
  "catboost": 0.195,
  "random_forest": 0.187
}
```

#### Confidence Calculation
```python
# Ensemble confidence based on model agreement
predictions = [model.predict(features) for model in models]
std_dev = np.std(predictions)

# High agreement → High confidence
confidence = max(0.5, min(1.0, 1.0 - (std_dev / 50)))
```

**Confidence Threshold**: 0.75 (75%)
- If confidence ≥ 75%: Use ML prediction
- If confidence < 75%: Fall back to optimization

---

## Optimization Algorithms

### Constraint Programming (OR-Tools)

**Algorithm**: Google OR-Tools CP-SAT Solver

**Problem Type**: Constraint Satisfaction Problem (CSP)

#### Variables
```python
# Decision variables for each trainset
for train in trainsets:
    for time_slot in operational_hours:
        is_assigned[train, time_slot] = BoolVar()
```

#### Constraints

**1. Fleet Coverage**
```
Σ(active_trains_at_time_t) ≥ min_service_trains
∀ t ∈ peak_hours
```

**2. Turnaround Time**
```
end_time[trip_i] + turnaround_time ≤ start_time[trip_i+1]
∀ consecutive trips of same train
```

**3. Maintenance Windows**
```
if train.status == MAINTENANCE:
    is_assigned[train, t] = False
    ∀ t ∈ maintenance_window
```

**4. Fitness Certificates**
```
if certificate_expired(train):
    is_assigned[train, t] = False
    ∀ t
```

**5. Mileage Balancing**
```
min_mileage ≤ daily_km[train] ≤ max_mileage
∀ trains in AVAILABLE status
```

**6. Depot Capacity**
```
Σ(trains_in_depot_at_t) ≤ depot_capacity
∀ t ∈ non_operational_hours
```

#### Objective Functions

**Multi-objective optimization** with weighted sum:

```python
objective = (
    0.35 × maximize(service_coverage) +
    0.25 × minimize(mileage_variance) +
    0.20 × maximize(availability_utilization) +
    0.10 × minimize(certificate_violations) +
    0.10 × maximize(branding_exposure)
)
```

**Component Details**:

1. **Service Coverage** (35% weight)
   - Maximize trains in service during peak hours
   - Ensure minimum standby capacity

2. **Mileage Variance** (25% weight)
   - Balance cumulative mileage across fleet
   - Prevent overuse of specific trainsets
   - Formula: `1 / (1 + coefficient_of_variation)`

3. **Availability Utilization** (20% weight)
   - Maximize usage of available healthy trains
   - Minimize idle time for service-ready trainsets

4. **Certificate Violations** (10% weight)
   - Minimize assignments with expiring certificates
   - Penalize near-expiry usage (< 30 days)

5. **Branding Exposure** (10% weight)
   - Prioritize branded trains during peak hours
   - Maximize visibility of high-priority advertisers

---

### Greedy Optimization

**Algorithm**: Priority-based greedy assignment

**Location**: `greedyOptim/` folder

#### Priority Scoring
```python
priority_score = (
    0.40 × readiness_score +
    0.25 × (1 - normalized_mileage) +
    0.20 × certificate_validity_days +
    0.10 × branding_priority +
    0.05 × maintenance_gap_days
)
```

#### Assignment Process

1. **Sort trains by priority** (descending)
2. **Iterate through time slots** (5 AM → 11 PM)
3. **For each slot**:
   - Select highest-priority available train
   - Check constraints (turnaround, capacity)
   - Assign if feasible
   - Update train state (location, mileage)
4. **Fallback**: If no train available, flag as gap

**Complexity**: O(n × t) where n = trains, t = time slots

**Advantages**:
- Fast execution (< 1 second for 40 trains)
- Interpretable decisions
- Good for real-time adjustments

**Disadvantages**:
- May not find global optimum
- Sensitive to initial priority weights

---

### Genetic Algorithm

**Algorithm**: Evolutionary optimization

**Location**: `greedyOptim/genetic_algorithm.py`

#### Parameters
- **Population size**: 100 schedules
- **Generations**: 50 iterations
- **Crossover rate**: 0.8
- **Mutation rate**: 0.1
- **Selection**: Tournament (k=3)

#### Chromosome Encoding
```python
# Each chromosome = complete schedule
chromosome = [train_id_for_trip_0, train_id_for_trip_1, ..., train_id_for_trip_n]
```

#### Fitness Function
```python
fitness = (
    service_quality_score -
    constraint_violations × penalty_weight
)
```

#### Genetic Operators

**1. Crossover (Single-point)**
```python
parent1 = [T1, T2, T3, T4, T5, T6]
parent2 = [T3, T1, T4, T2, T6, T5]
         ↓ crossover at position 3
child1  = [T1, T2, T3, T2, T6, T5]
child2  = [T3, T1, T4, T4, T5, T6]
```

**2. Mutation (Swap)**
```python
# Randomly swap two trip assignments
schedule = [T1, T2, T3, T4, T5]
         ↓ swap positions 1 and 3
mutated  = [T1, T4, T3, T2, T5]
```

**Termination**: Max generations or convergence (no improvement for 10 generations)

---

## Hybrid Approach

### Decision Flow

```
┌─────────────────────┐
│  Schedule Request   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────┐
│ Extract Features from Request   │
│ (num_trains, time, day, etc.)  │
└──────────┬──────────────────────┘
           │
           ▼
┌─────────────────────────────────┐
│  Ensemble ML Prediction         │
│  - All 5 models predict         │
│  - Weighted voting              │
│  - Calculate confidence         │
└──────────┬──────────────────────┘
           │
           ▼
      Confidence ≥ 75%?
           │
    ┌──────┴──────┐
    │             │
   YES            NO
    │             │
    ▼             ▼
┌───────┐   ┌──────────┐
│  Use  │   │   Use    │
│  ML   │   │OR-Tools  │
│Result │   │ Optimize │
└───────┘   └──────────┘
    │             │
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │  Schedule   │
    └─────────────┘
```

### When ML is Used

**Conditions**:
1. ✅ Models trained (≥100 schedules)
2. ✅ Confidence score ≥ 75%
3. ✅ Hybrid mode enabled

**Typical Scenarios**:
- Standard 30-train fleet
- Normal operational parameters
- No major disruptions

### When Optimization is Used

**Conditions**:
- ❌ Low ML confidence (< 75%)
- ❌ Models not trained
- ❌ Unusual parameters (edge cases)
- ❌ First-time scheduling

**Typical Scenarios**:
- Fleet size changes (25→40 trains)
- New route configurations
- Major maintenance events
- System initialization

---

## Feature Engineering

### Input Features (10 dimensions)

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `num_trains` | Integer | 25-40 | Total fleet size |
| `num_available` | Integer | 20-38 | Trains in service/standby |
| `avg_readiness_score` | Float | 0.0-1.0 | Average train health |
| `total_mileage` | Integer | 100K-500K | Fleet cumulative km |
| `mileage_variance` | Float | 0-50K | Std dev of mileage |
| `maintenance_count` | Integer | 0-10 | Trains in maintenance |
| `certificate_expiry_count` | Integer | 0-5 | Expiring certificates |
| `branding_priority_sum` | Integer | 0-100 | Total branding priority |
| `time_of_day` | Integer | 0-23 | Hour of day |
| `day_of_week` | Integer | 0-6 | Day (0=Monday) |

### Target Variable

**Schedule Quality Score** (0-100):

```python
score = (
    avg_readiness × 30 +        # Health (30 points)
    availability_% × 25 +        # Availability (25 points)
    (1 - mileage_var) × 20 +    # Balance (20 points)
    branding_sla × 15 +          # Branding (15 points)
    (10 - violations×2)          # Compliance (10 points)
)
```

### Feature Scaling

All features normalized to [0, 1] range before training:

```python
feature_normalized = (value - min) / (max - min)
```

---

## Performance Metrics

### Model Evaluation

**Primary Metric**: R² Score (Coefficient of Determination)
- Range: [0, 1], higher is better
- Typical ensemble R²: 0.85-0.92

**Secondary Metric**: RMSE (Root Mean Squared Error)
- Range: [0, ∞], lower is better
- Typical ensemble RMSE: 8-15

**Training Split**: 80% train, 20% test

### Optimization Quality

**Metrics Tracked**:

1. **Service Coverage**: % of required hours covered
   - Target: ≥ 95%

2. **Fleet Utilization**: % of available trains used
   - Target: 85-95%

3. **Mileage Balance**: Coefficient of variation
   - Target: < 0.15 (15%)

4. **Constraint Violations**: Count of hard constraint breaks
   - Target: 0

5. **Execution Time**: Algorithm runtime
   - ML: < 0.1 seconds
   - OR-Tools: 1-5 seconds
   - Genetic: 5-15 seconds

### Ensemble Performance Example

```json
{
  "gradient_boosting": {
    "train_r2": 0.8912,
    "test_r2": 0.8234,
    "test_rmse": 13.45
  },
  "xgboost": {
    "train_r2": 0.9234,
    "test_r2": 0.8543,
    "test_rmse": 12.34
  },
  "lightgbm": {
    "train_r2": 0.9156,
    "test_r2": 0.8467,
    "test_rmse": 12.67
  },
  "catboost": {
    "train_r2": 0.9087,
    "test_r2": 0.8401,
    "test_rmse": 12.89
  },
  "random_forest": {
    "train_r2": 0.8756,
    "test_r2": 0.8123,
    "test_rmse": 13.98
  },
  "ensemble": {
    "test_r2": 0.8621,
    "test_rmse": 11.87,
    "confidence": 0.89
  }
}
```

---

## Algorithm Selection Guide

| Use Case | Recommended Algorithm | Rationale |
|----------|----------------------|-----------|
| First-time scheduling | OR-Tools CP-SAT | No training data available |
| Standard operations | Ensemble ML | Fast, accurate predictions |
| Edge cases | OR-Tools CP-SAT | Guaranteed feasibility |
| Real-time updates | Greedy + ML | Sub-second performance |
| Offline planning | Genetic Algorithm | Exploration of solution space |
| Development/Testing | LightGBM | Fastest training iteration |
| Production inference | XGBoost | Best accuracy/speed trade-off |

---

## Future Enhancements

### Planned Improvements

1. **Reinforcement Learning**
   - Q-learning for dynamic scheduling
   - Reward: schedule quality over time
   
2. **Deep Learning**
   - LSTM for time-series prediction
   - Attention mechanisms for trip dependencies

3. **Multi-objective Pareto**
   - Generate Pareto-optimal solution set
   - Allow user to select trade-off point

4. **Transfer Learning**
   - Pre-train on similar metro systems
   - Fine-tune for KMRL specifics

5. **Online Learning**
   - Incremental model updates
   - Adapt to changing patterns without full retraining

---

## References

### Libraries
- **Scikit-learn**: https://scikit-learn.org/
- **XGBoost**: https://xgboost.readthedocs.io/
- **LightGBM**: https://lightgbm.readthedocs.io/
- **CatBoost**: https://catboost.ai/
- **OR-Tools**: https://developers.google.com/optimization

### Papers
1. Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System"
2. Ke, G., et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree"
3. Prokhorenkova, L., et al. (2018). "CatBoost: unbiased boosting with categorical features"

---

**Document Version**: 1.0.0  
**Last Updated**: November 2, 2025  
**Maintained By**: ML-Service Team
