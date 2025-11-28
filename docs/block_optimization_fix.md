# Block Optimization Fix Summary

## The Problem

NSGA-II optimizer was only producing **33-42 blocks** instead of the expected **106 blocks**.

## Root Causes

### 1. Reference vs Copy Issue
When storing best solutions from the Pareto front, we stored references instead of copies:

```python
# WRONG - stores references that get overwritten
best_solutions = [(population[i], objectives[i]) for i in fronts[0]]
best_block_solutions = [block_population[i] for i in fronts[0]]
```

Since `population` and `block_population` are replaced each generation with `offspring`, the stored references pointed to stale/corrupted data.

### 2. Block-Trainset Mismatch
Even with copies, the stored block assignments were created for a *different* trainset selection. When the best solution evolved to have different service trainsets, the old block assignment still mapped to old trainset indices.

Example:
- Generation 50: Best solution has trainsets [0, 2, 5] → blocks assigned to indices 0, 2, 5
- Generation 150: Best solution evolves to trainsets [1, 3, 7] → but block assignment still references 0, 2, 5
- Result: Many blocks map to non-service trainsets → lost blocks

## The Fix

**Always create fresh block assignments for the final best solution:**

```python
# Select best solution from Pareto front
if best_solutions:
    best_idx = min(range(len(best_solutions)), 
                  key=lambda i: self.evaluator.fitness_function(best_solutions[i][0]))
    best_solution, best_objectives = best_solutions[best_idx]
    if self.optimize_blocks:
        # Always create fresh block assignment for the best solution
        # to ensure all 106 blocks are properly assigned
        best_block_sol = self._create_block_assignment(best_solution)
```

The `_create_block_assignment` distributes all blocks evenly across current service trainsets:

```python
def _create_block_assignment(self, trainset_sol: np.ndarray) -> np.ndarray:
    service_indices = np.where(trainset_sol == 0)[0]
    
    if len(service_indices) == 0:
        return np.full(self.n_blocks, -1, dtype=int)
    
    # Distribute blocks evenly across service trains
    block_sol = np.zeros(self.n_blocks, dtype=int)
    for i in range(self.n_blocks):
        block_sol[i] = service_indices[i % len(service_indices)]
    
    return block_sol
```

## Result

| Optimizer | Before Fix | After Fix |
|-----------|-----------|-----------|
| GA        | 106 ✓     | 106 ✓     |
| CMA-ES    | 106 ✓     | 106 ✓     |
| PSO       | 106 ✓     | 106 ✓     |
| SA        | 106 ✓     | 106 ✓     |
| NSGA-II   | 33-42 ✗   | 106 ✓     |

All optimizers now correctly assign all 106 service blocks.
