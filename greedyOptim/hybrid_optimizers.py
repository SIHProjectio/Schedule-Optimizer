"""
Hybrid optimization algorithms combining multiple approaches.
Includes multi-objective, adaptive, and ensemble methods.
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
import random
from concurrent.futures import ThreadPoolExecutor
import time

from .models import OptimizationResult, OptimizationConfig
from .evaluator import TrainsetSchedulingEvaluator
from .genetic_algorithm import GeneticAlgorithmOptimizer
from .advanced_optimizers import CMAESOptimizer, ParticleSwarmOptimizer, SimulatedAnnealingOptimizer


class MultiObjectiveOptimizer:
    """Multi-objective optimization using NSGA-II approach."""
    
    def __init__(self, evaluator: TrainsetSchedulingEvaluator, config: Optional[OptimizationConfig] = None):
        self.evaluator = evaluator
        self.config = config or OptimizationConfig()
        self.n_genes = evaluator.num_trainsets
        self.n_blocks = evaluator.num_blocks
        self.optimize_blocks = self.config.optimize_block_assignment
        
        # Objective weights for dominance comparison
        # Higher weight = more important in determining dominance
        self.objective_weights = {
            'service_availability': 5.0,   # HIGHEST: More trains = better operations
            'mileage_balance': 1.5,        # Medium: Fleet wear balance
            'maintenance_cost': 1.0,       # Medium: Avoid overdue maintenance
            'branding_compliance': 0.2,    # LOW: Nice-to-have
            'constraint_penalty': 10.0     # CRITICAL: Hard constraints
        }
        
    def dominates(self, solution1: Dict[str, float], solution2: Dict[str, float]) -> bool:
        """Check if solution1 dominates solution2 in multi-objective sense.
        
        Uses weighted objectives to prioritize service availability over branding.
        """
        # Convert maximization objectives to minimization (lower is better)
        # Apply weights to emphasize important objectives
        w = self.objective_weights
        obj1 = [
            -solution1['service_availability'] * w['service_availability'],
            -solution1['mileage_balance'] * w['mileage_balance'],
            -solution1['maintenance_cost'] * w['maintenance_cost'],
            -solution1['branding_compliance'] * w['branding_compliance'],
            solution1['constraint_penalty'] * w['constraint_penalty']
        ]
        obj2 = [
            -solution2['service_availability'] * w['service_availability'],
            -solution2['mileage_balance'] * w['mileage_balance'],
            -solution2['maintenance_cost'] * w['maintenance_cost'],
            -solution2['branding_compliance'] * w['branding_compliance'],
            solution2['constraint_penalty'] * w['constraint_penalty']
        ]
        
        # Check if all objectives are better or equal, with at least one strictly better
        all_better_equal = all(o1 <= o2 for o1, o2 in zip(obj1, obj2))
        any_better = any(o1 < o2 for o1, o2 in zip(obj1, obj2))
        
        return all_better_equal and any_better
    
    def fast_non_dominated_sort(self, objectives: List[Dict[str, float]]) -> List[List[int]]:
        """Fast non-dominated sorting for NSGA-II."""
        n = len(objectives)
        dominated_solutions = [[] for _ in range(n)]
        domination_count = [0] * n
        fronts = [[]]
        
        # Find domination relationships
        for i in range(n):
            for j in range(n):
                if i != j:
                    if self.dominates(objectives[i], objectives[j]):
                        dominated_solutions[i].append(j)
                    elif self.dominates(objectives[j], objectives[i]):
                        domination_count[i] += 1
            
            if domination_count[i] == 0:
                fronts[0].append(i)
        
        # Build subsequent fronts
        front_idx = 0
        while fronts[front_idx]:
            next_front = []
            for i in fronts[front_idx]:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            front_idx += 1
            fronts.append(next_front)
        
        return fronts[:-1]  # Remove empty last front
    
    def crowding_distance(self, front: List[int], objectives: List[Dict[str, float]]) -> List[float]:
        """Calculate crowding distance for solutions in a front."""
        if len(front) <= 2:
            return [float('inf')] * len(front)
        
        distances = [0.0] * len(front)
        obj_names = ['service_availability', 'branding_compliance', 'mileage_balance', 
                    'maintenance_cost', 'constraint_penalty']
        
        for obj_name in obj_names:
            # Sort by objective value
            sorted_indices = sorted(range(len(front)), 
                                  key=lambda x: objectives[front[x]][obj_name])
            
            # Boundary solutions get infinite distance
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')
            
            # Calculate crowding distance for intermediate solutions
            obj_range = (objectives[front[sorted_indices[-1]]][obj_name] - 
                        objectives[front[sorted_indices[0]]][obj_name])
            
            if obj_range > 0:
                for i in range(1, len(sorted_indices) - 1):
                    distances[sorted_indices[i]] += (
                        objectives[front[sorted_indices[i+1]]][obj_name] - 
                        objectives[front[sorted_indices[i-1]]][obj_name]
                    ) / obj_range
        
        return distances
    
    def _create_block_assignment(self, trainset_sol: np.ndarray) -> np.ndarray:
        """Create block assignments for a trainset solution."""
        service_indices = np.where(trainset_sol == 0)[0]
        
        if len(service_indices) == 0:
            return np.full(self.n_blocks, -1, dtype=int)
        
        # Distribute blocks evenly across service trains
        block_sol = np.zeros(self.n_blocks, dtype=int)
        for i in range(self.n_blocks):
            block_sol[i] = service_indices[i % len(service_indices)]
        
        return block_sol
    
    def _mutate_block_assignment(self, block_sol: np.ndarray, service_indices: np.ndarray) -> np.ndarray:
        """Mutate block assignment."""
        mutated = block_sol.copy()
        
        if len(service_indices) == 0:
            return mutated
        
        # Randomly reassign some blocks
        num_mutations = max(1, self.n_blocks // 10)
        for _ in range(num_mutations):
            idx = np.random.randint(0, len(mutated))
            mutated[idx] = np.random.choice(service_indices)
        
        return mutated
    
    def _create_smart_initial_solution(self) -> np.ndarray:
        """Create a smart initial solution that respects constraints."""
        solution = np.zeros(self.n_genes, dtype=int)  # Start with all service
        
        standby_count = 0
        for i, ts_id in enumerate(self.evaluator.trainsets):
            valid, _ = self.evaluator.check_hard_constraints(ts_id)
            if not valid:
                solution[i] = 2  # Put constraint-violating trainsets in maintenance
            elif standby_count < self.config.min_standby:
                solution[i] = 1  # Reserve some healthy ones for standby
                standby_count += 1
        
        return solution
    
    def optimize(self) -> OptimizationResult:
        """Run NSGA-II multi-objective optimization."""
        # Initialize population with trainset solutions and block assignments
        # Mix of smart and random solutions for diversity
        population = []
        block_population = []
        
        # First, add some smart solutions (constraint-aware)
        num_smart = min(10, self.config.population_size // 5)
        for _ in range(num_smart):
            solution = self._create_smart_initial_solution()
            # Add some random mutation to create diversity
            for i in range(self.n_genes):
                if np.random.random() < 0.1:  # 10% mutation
                    solution[i] = np.random.choice([0, 1, 2], p=[0.70, 0.20, 0.10])
            population.append(solution)
            if self.optimize_blocks:
                block_sol = self._create_block_assignment(solution)
                block_population.append(block_sol)
        
        # Fill rest with biased random (favor service)
        for _ in range(self.config.population_size - num_smart):
            solution = np.random.choice([0, 1, 2], size=self.n_genes, p=[0.65, 0.20, 0.15])
            population.append(solution)
            if self.optimize_blocks:
                block_sol = self._create_block_assignment(solution)
                block_population.append(block_sol)
        
        best_solutions = []
        best_block_solutions = []
        
        print(f"Starting NSGA-II multi-objective optimization for {self.config.generations} generations")
        if self.optimize_blocks:
            print(f"Optimizing block assignments for {self.n_blocks} service blocks")
        
        for gen in range(self.config.generations):
            try:
                # Evaluate objectives for all solutions
                objectives = []
                for idx, solution in enumerate(population):
                    obj = self.evaluator.calculate_objectives(solution)
                    objectives.append(obj)
                
                # Non-dominated sorting
                fronts = self.fast_non_dominated_sort(objectives)
                
                # Selection for next generation
                new_population = []
                new_block_population = [] if self.optimize_blocks else None
                for front in fronts:
                    if len(new_population) + len(front) <= self.config.population_size:
                        new_population.extend([population[i] for i in front])
                        if self.optimize_blocks:
                            new_block_population.extend([block_population[i] for i in front])
                    else:
                        # Use crowding distance to select from this front
                        distances = self.crowding_distance(front, objectives)
                        sorted_front = sorted(zip(front, distances), 
                                            key=lambda x: x[1], reverse=True)
                        remaining = self.config.population_size - len(new_population)
                        new_population.extend([population[i] for i, _ in sorted_front[:remaining]])
                        if self.optimize_blocks:
                            new_block_population.extend([block_population[i] for i, _ in sorted_front[:remaining]])
                        break
                
                # Store best solutions from first front
                if fronts and len(fronts[0]) > 0:
                    best_solutions = [(population[i].copy(), objectives[i].copy()) for i in fronts[0]]
                    if self.optimize_blocks:
                        best_block_solutions = [block_population[i].copy() for i in fronts[0]]
                
                # Generate offspring through crossover and mutation
                offspring = []
                offspring_blocks = [] if self.optimize_blocks else None
                
                # Ensure block population is synchronized
                if self.optimize_blocks and len(new_block_population) != len(new_population):
                    # Rebuild block population if out of sync
                    new_block_population = [self._create_block_assignment(sol) for sol in new_population]
                
                while len(offspring) < self.config.population_size:
                    idx1 = random.randint(0, len(new_population) - 1)
                    idx2 = random.randint(0, len(new_population) - 1)
                    parent1 = new_population[idx1]
                    parent2 = new_population[idx2]
                    
                    # Simple crossover
                    if random.random() < self.config.crossover_rate:
                        point = random.randint(1, self.n_genes - 1)
                        child = np.concatenate([parent1[:point], parent2[point:]])
                    else:
                        child = parent1.copy()
                    
                    # Mutation with bias towards service (0)
                    for i in range(self.n_genes):
                        if random.random() < self.config.mutation_rate:
                            # 55% chance to mutate to service, 30% depot, 15% maintenance
                            child[i] = np.random.choice([0, 1, 2], p=[0.55, 0.30, 0.15])
                    
                    offspring.append(child)
                    
                    # Handle block crossover and mutation
                    if self.optimize_blocks:
                        block_parent1 = new_block_population[idx1]
                        block_parent2 = new_block_population[idx2]
                        
                        # Block crossover
                        if random.random() < self.config.crossover_rate:
                            block_point = random.randint(1, self.n_blocks - 1)
                            block_child = np.concatenate([block_parent1[:block_point], block_parent2[block_point:]])
                        else:
                            block_child = block_parent1.copy()
                        
                        # Ensure valid block assignments for new child's service trains
                        service_indices = np.where(child == 0)[0]
                        if len(service_indices) > 0:
                            block_child = self._mutate_block_assignment(block_child, service_indices)
                        else:
                            block_child = np.full(self.n_blocks, -1, dtype=int)
                        
                        offspring_blocks.append(block_child)
                
                # ELITISM: Combine parents and offspring, then select best
                combined_population = new_population + offspring
                combined_blocks = (new_block_population + offspring_blocks) if self.optimize_blocks else None
                
                # Evaluate combined population
                combined_objectives = []
                for sol in combined_population:
                    combined_objectives.append(self.evaluator.calculate_objectives(sol))
                
                # Non-dominated sorting on combined population
                combined_fronts = self.fast_non_dominated_sort(combined_objectives)
                
                # Select best individuals for next generation
                population = []
                block_population = [] if self.optimize_blocks else None
                
                for front in combined_fronts:
                    if len(population) + len(front) <= self.config.population_size:
                        population.extend([combined_population[i].copy() for i in front])
                        if self.optimize_blocks:
                            block_population.extend([combined_blocks[i].copy() for i in front])
                    else:
                        # Use crowding distance for this front
                        distances = self.crowding_distance(front, combined_objectives)
                        sorted_front = sorted(zip(front, distances), key=lambda x: x[1], reverse=True)
                        remaining = self.config.population_size - len(population)
                        population.extend([combined_population[i].copy() for i, _ in sorted_front[:remaining]])
                        if self.optimize_blocks:
                            block_population.extend([combined_blocks[i].copy() for i, _ in sorted_front[:remaining]])
                        break
                
                if gen % 50 == 0:
                    best_service = max(obj.get('service_availability', 0) for obj in combined_objectives)
                    min_penalty = min(obj.get('constraint_penalty', 9999) for obj in combined_objectives)
                    print(f"Generation {gen}: {len(combined_fronts)} fronts, best service: {best_service:.1f}, min penalty: {min_penalty:.0f}")
                    
            except Exception as e:
                print(f"Error in NSGA-II generation {gen}: {e}")
                break
        
        # Select best solution from Pareto front - prioritize service availability
        best_block_sol = None
        if best_solutions:
            # First, find solutions with zero constraint penalty
            valid_solutions = [(i, sol, obj) for i, (sol, obj) in enumerate(best_solutions)
                              if obj.get('constraint_penalty', 0) == 0]
            
            if valid_solutions:
                # Among valid solutions, choose the one with highest service_availability
                # (which means more trains in service)
                best_idx = max(valid_solutions, 
                              key=lambda x: x[2].get('service_availability', 0))[0]
            else:
                # Fall back to lowest constraint penalty + highest service
                best_idx = max(range(len(best_solutions)),
                              key=lambda i: (
                                  -best_solutions[i][1].get('constraint_penalty', float('inf')),
                                  best_solutions[i][1].get('service_availability', 0)
                              ))
            
            best_solution, best_objectives = best_solutions[best_idx]
            if self.optimize_blocks:
                # Always create fresh block assignment for the best solution
                # to ensure all 106 blocks are properly assigned
                best_block_sol = self._create_block_assignment(best_solution)
        else:
            # Fallback to first solution
            best_solution = population[0]
            best_objectives = self.evaluator.calculate_objectives(best_solution)
            if self.optimize_blocks:
                best_block_sol = self._create_block_assignment(best_solution)
        
        return self._build_result(best_solution, best_objectives, best_block_sol)
    
    def _build_result(self, solution: np.ndarray, objectives: Dict[str, float],
                      block_solution: Optional[np.ndarray] = None) -> OptimizationResult:
        """Build optimization result."""
        fitness = self.evaluator.fitness_function(solution)
        
        service = [self.evaluator.trainsets[i] for i, v in enumerate(solution) if v == 0]
        standby = [self.evaluator.trainsets[i] for i, v in enumerate(solution) if v == 1]
        maintenance = [self.evaluator.trainsets[i] for i, v in enumerate(solution) if v == 2]
        
        explanations = {}
        for ts_id in service:
            valid, reason = self.evaluator.check_hard_constraints(ts_id)
            explanations[ts_id] = "✓ Fit for service" if valid else f"⚠ {reason}"
        
        # Build block assignments
        block_assignments = {}
        if block_solution is not None and self.optimize_blocks:
            for ts_id in service:
                block_assignments[ts_id] = []
            
            for block_idx, train_idx in enumerate(block_solution):
                if 0 <= train_idx < len(self.evaluator.trainsets):
                    ts_id = self.evaluator.trainsets[int(train_idx)]
                    if ts_id in block_assignments:
                        block_id = self.evaluator.all_blocks[block_idx]['block_id']
                        block_assignments[ts_id].append(block_id)
        
        return OptimizationResult(
            selected_trainsets=service,
            standby_trainsets=standby,
            maintenance_trainsets=maintenance,
            objectives=objectives,
            fitness_score=fitness,
            explanation=explanations,
            service_block_assignments=block_assignments
        )

class AdaptiveOptimizer:
    """Adaptive optimizer that switches between algorithms based on performance."""
    
    def __init__(self, evaluator: TrainsetSchedulingEvaluator, config: Optional[OptimizationConfig] = None):
        self.evaluator = evaluator
        self.config = config or OptimizationConfig()
        
        # Initialize different optimizers
        self.optimizers = {
            'ga': GeneticAlgorithmOptimizer(evaluator, config),
            'cmaes': CMAESOptimizer(evaluator, config),
            'pso': ParticleSwarmOptimizer(evaluator, config),
            'sa': SimulatedAnnealingOptimizer(evaluator, config)
        }
        
        self.performance_history = {name: [] for name in self.optimizers.keys()}
        self.selection_probabilities = {name: 0.25 for name in self.optimizers.keys()}
    
    def update_probabilities(self):
        """Update selection probabilities based on recent performance."""
        # Calculate average improvement for each optimizer
        improvements = {}
        for name, history in self.performance_history.items():
            if len(history) >= 2:
                recent_improvement = history[-2] - history[-1]  # Lower fitness is better
                improvements[name] = max(0, recent_improvement)
            else:
                improvements[name] = 0
        
        # Update probabilities (softmax-like)
        # Calculate raw weights
        weights = {}
        for name in self.optimizers.keys():
            weights[name] = improvements[name] + 0.1
            
        total_weight = sum(weights.values())
        
        # Normalize
        for name in self.optimizers.keys():
            self.selection_probabilities[name] = weights[name] / total_weight
    
    def select_optimizer(self) -> str:
        """Select optimizer based on adaptive probabilities."""
        names = list(self.selection_probabilities.keys())
        probs = list(self.selection_probabilities.values())
        return np.random.choice(names, p=probs)
    
    def optimize(self, max_iterations: int = 5) -> OptimizationResult:
        """Run adaptive optimization with algorithm switching."""
        best_result = None
        best_fitness = float('inf')
        
        print(f"Starting Adaptive Optimization with {max_iterations} iterations")
        
        for iteration in range(max_iterations):
            # Select optimizer
            selected = self.select_optimizer()
            print(f"Iteration {iteration + 1}: Using {selected.upper()}")
            
            try:
                # Run selected optimizer with reduced generations
                reduced_config = OptimizationConfig(
                    required_service_trains=self.config.required_service_trains,
                    min_standby=self.config.min_standby,
                    population_size=max(20, self.config.population_size // 2),
                    generations=max(50, self.config.generations // 4)
                )
                
                if selected == 'ga':
                    optimizer = GeneticAlgorithmOptimizer(self.evaluator, reduced_config)
                    result = optimizer.optimize()
                elif selected == 'cmaes':
                    optimizer = CMAESOptimizer(self.evaluator, reduced_config)
                    result = optimizer.optimize(50)
                elif selected == 'pso':
                    optimizer = ParticleSwarmOptimizer(self.evaluator, reduced_config) 
                    result = optimizer.optimize(50)
                else:  # sa
                    optimizer = SimulatedAnnealingOptimizer(self.evaluator, reduced_config)
                    result = optimizer.optimize(2000)
                
                # Update performance history
                self.performance_history[selected].append(result.fitness_score)
                
                # Track best result
                if result.fitness_score < best_fitness:
                    best_fitness = result.fitness_score
                    best_result = result
                    print(f"  New best fitness: {best_fitness:.2f}")
                
                # Update probabilities
                self.update_probabilities()
                
            except Exception as e:
                print(f"  Error with {selected}: {e}")
                self.performance_history[selected].append(float('inf'))
        
        # Print final probabilities
        print(f"\nFinal algorithm probabilities:")
        for name, prob in self.selection_probabilities.items():
            print(f"  {name.upper()}: {prob:.3f}")
        
        return best_result or OptimizationResult([], [], [], {}, float('inf'), {})


class EnsembleOptimizer:
    """Ensemble optimizer that runs multiple algorithms in parallel."""
    
    def __init__(self, evaluator: TrainsetSchedulingEvaluator, config: Optional[OptimizationConfig] = None):
        self.evaluator = evaluator
        self.config = config or OptimizationConfig()
    
    def run_single_optimizer(self, optimizer_name: str) -> Tuple[str, OptimizationResult]:
        """Run a single optimizer and return result with name."""
        try:
            reduced_config = OptimizationConfig(
                required_service_trains=self.config.required_service_trains,
                min_standby=self.config.min_standby,
                population_size=max(30, self.config.population_size // 2),
                generations=max(100, self.config.generations // 2)
            )
            
            if optimizer_name == 'ga':
                optimizer = GeneticAlgorithmOptimizer(self.evaluator, reduced_config)
                result = optimizer.optimize()
            elif optimizer_name == 'cmaes':
                optimizer = CMAESOptimizer(self.evaluator, reduced_config)
                result = optimizer.optimize(75)
            elif optimizer_name == 'pso':
                optimizer = ParticleSwarmOptimizer(self.evaluator, reduced_config)
                result = optimizer.optimize(100)
            elif optimizer_name == 'sa':
                optimizer = SimulatedAnnealingOptimizer(self.evaluator, reduced_config)  
                result = optimizer.optimize(5000)
            else:
                raise ValueError(f"Unknown optimizer: {optimizer_name}")
            
            return optimizer_name, result
            
        except Exception as e:
            print(f"Error in {optimizer_name}: {e}")
            return optimizer_name, OptimizationResult([], [], [], {}, float('inf'), {})
    
    def optimize(self) -> OptimizationResult:
        """Run ensemble optimization with parallel execution."""
        print("Starting Ensemble Optimization (parallel execution)...")
        
        optimizers = ['ga', 'cmaes', 'pso', 'sa']
        results = {}
        
        start_time = time.time()
        
        # Run optimizers in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(self.run_single_optimizer, opt): opt for opt in optimizers}
            
            for future in futures:
                optimizer_name, result = future.result()
                results[optimizer_name] = result
                print(f"  {optimizer_name.upper()} completed: fitness = {result.fitness_score:.2f}")
        
        elapsed = time.time() - start_time
        print(f"Ensemble completed in {elapsed:.1f} seconds")
        
        # Select best result
        best_name, best_result = min(results.items(), key=lambda x: x[1].fitness_score)
        
        print(f"\n🏆 Best result from {best_name.upper()}: {best_result.fitness_score:.2f}")
        
        # Add ensemble info to explanation
        ensemble_info = f"Ensemble winner: {best_name.upper()}"
        for ts_id in best_result.explanation:
            best_result.explanation[ts_id] += f" ({ensemble_info})"
        
        return best_result


class HyperParameterOptimizer:
    """Optimize hyperparameters for the optimization algorithms."""
    
    def __init__(self, evaluator: TrainsetSchedulingEvaluator):
        self.evaluator = evaluator
    
    def optimize_ga_params(self, trials: int = 10) -> OptimizationConfig:
        """Optimize GA hyperparameters using random search."""
        print(f"Optimizing GA hyperparameters ({trials} trials)...")
        
        best_config = None
        best_fitness = float('inf')
        
        for trial in range(trials):
            # Random hyperparameters
            config = OptimizationConfig(
                population_size=random.choice([30, 50, 80, 100]),
                generations=random.choice([50, 100, 150]),
                mutation_rate=random.uniform(0.05, 0.2),
                crossover_rate=random.uniform(0.6, 0.9),
                elite_size=random.randint(3, 8)
            )
            
            try:
                optimizer = GeneticAlgorithmOptimizer(self.evaluator, config)
                result = optimizer.optimize()
                
                if result.fitness_score < best_fitness:
                    best_fitness = result.fitness_score
                    best_config = config
                    
                print(f"  Trial {trial + 1}: fitness = {result.fitness_score:.2f}")
                
            except Exception as e:
                print(f"  Trial {trial + 1} failed: {e}")
        
        if best_config:
            print(f"Best GA config: pop={best_config.population_size}, "
                  f"gen={best_config.generations}, mut={best_config.mutation_rate:.3f}")
        else:
            print("No valid configuration found, using default")
        
        return best_config or OptimizationConfig()


# Integration function for the new hybrid methods
def optimize_with_hybrid_methods(data: Dict, method: str = 'adaptive') -> OptimizationResult:
    """Run optimization with hybrid methods.
    
    Args:
        data: Metro synthetic data
        method: 'multi-objective', 'adaptive', 'ensemble', or 'auto-tune'
    """
    from .evaluator import TrainsetSchedulingEvaluator
    
    evaluator = TrainsetSchedulingEvaluator(data)
    
    if method == 'multi-objective':
        optimizer = MultiObjectiveOptimizer(evaluator)
        return optimizer.optimize()
    elif method == 'adaptive':
        optimizer = AdaptiveOptimizer(evaluator)
        return optimizer.optimize()
    elif method == 'ensemble':
        optimizer = EnsembleOptimizer(evaluator)
        return optimizer.optimize()
    elif method == 'auto-tune':
        tuner = HyperParameterOptimizer(evaluator)
        best_config = tuner.optimize_ga_params()
        optimizer = GeneticAlgorithmOptimizer(evaluator, best_config)
        return optimizer.optimize()
    else:
        raise ValueError(f"Unknown hybrid method: {method}")


if __name__ == "__main__":
    import json
    
    # Load data
    try:
        with open('metro_enhanced_data.json', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Please generate enhanced data first: python DataService/enhanced_generator.py")
        exit(1)
    
    # Test hybrid methods
    methods = ['adaptive', 'ensemble']
    
    for method in methods:
        print(f"\n{'='*60}")
        print(f"Testing {method.upper()} optimization")
        print(f"{'='*60}")
        
        result = optimize_with_hybrid_methods(data, method)
        print(f"Final result: {len(result.selected_trainsets)} in service, "
              f"fitness = {result.fitness_score:.2f}")