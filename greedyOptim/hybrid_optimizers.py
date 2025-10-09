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
        
    def dominates(self, solution1: Dict[str, float], solution2: Dict[str, float]) -> bool:
        """Check if solution1 dominates solution2 in multi-objective sense."""
        # Convert maximization objectives to minimization (lower is better)
        obj1 = [-solution1['service_availability'], -solution1['branding_compliance'], 
                -solution1['mileage_balance'], -solution1['maintenance_cost'], 
                solution1['constraint_penalty']]
        obj2 = [-solution2['service_availability'], -solution2['branding_compliance'],
                -solution2['mileage_balance'], -solution2['maintenance_cost'],
                solution2['constraint_penalty']]
        
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
    
    def optimize(self) -> OptimizationResult:
        """Run NSGA-II multi-objective optimization."""
        # Initialize population
        population = []
        for _ in range(self.config.population_size):
            solution = np.random.randint(0, 3, self.n_genes)
            population.append(solution)
        
        best_solutions = []
        
        print(f"Starting NSGA-II multi-objective optimization for {self.config.generations} generations")
        
        for gen in range(self.config.generations):
            try:
                # Evaluate objectives for all solutions
                objectives = []
                for solution in population:
                    obj = self.evaluator.calculate_objectives(solution)
                    objectives.append(obj)
                
                # Non-dominated sorting
                fronts = self.fast_non_dominated_sort(objectives)
                
                # Selection for next generation
                new_population = []
                for front in fronts:
                    if len(new_population) + len(front) <= self.config.population_size:
                        new_population.extend([population[i] for i in front])
                    else:
                        # Use crowding distance to select from this front
                        distances = self.crowding_distance(front, objectives)
                        sorted_front = sorted(zip(front, distances), 
                                            key=lambda x: x[1], reverse=True)
                        remaining = self.config.population_size - len(new_population)
                        new_population.extend([population[i] for i, _ in sorted_front[:remaining]])
                        break
                
                # Store best solutions from first front
                if fronts and len(fronts[0]) > 0:
                    best_solutions = [(population[i], objectives[i]) for i in fronts[0]]
                
                # Generate offspring through crossover and mutation
                offspring = []
                while len(offspring) < self.config.population_size:
                    parent1 = random.choice(new_population)
                    parent2 = random.choice(new_population)
                    
                    # Simple crossover
                    if random.random() < self.config.crossover_rate:
                        point = random.randint(1, self.n_genes - 1)
                        child = np.concatenate([parent1[:point], parent2[point:]])
                    else:
                        child = parent1.copy()
                    
                    # Mutation
                    for i in range(self.n_genes):
                        if random.random() < self.config.mutation_rate:
                            child[i] = random.randint(0, 2)
                    
                    offspring.append(child)
                
                population = offspring
                
                if gen % 50 == 0:
                    print(f"Generation {gen}: {len(fronts)} fronts, best front size: {len(fronts[0]) if fronts else 0}")
                    
            except Exception as e:
                print(f"Error in NSGA-II generation {gen}: {e}")
                break
        
        # Select best solution from Pareto front
        if best_solutions:
            # Choose solution with best overall fitness
            best_solution, best_objectives = min(best_solutions, 
                                               key=lambda x: self.evaluator.fitness_function(x[0]))
        else:
            # Fallback to first solution
            best_solution = population[0]
            best_objectives = self.evaluator.calculate_objectives(best_solution)
        
        return self._build_result(best_solution, best_objectives)
    
    def _build_result(self, solution: np.ndarray, objectives: Dict[str, float]) -> OptimizationResult:
        """Build optimization result."""
        fitness = self.evaluator.fitness_function(solution)
        
        service = [self.evaluator.trainsets[i] for i, v in enumerate(solution) if v == 0]
        standby = [self.evaluator.trainsets[i] for i, v in enumerate(solution) if v == 1]
        maintenance = [self.evaluator.trainsets[i] for i, v in enumerate(solution) if v == 2]
        
        explanations = {}
        for ts_id in service:
            valid, reason = self.evaluator.check_hard_constraints(ts_id)
            explanations[ts_id] = "✓ Fit for service" if valid else f"⚠ {reason}"
        
        return OptimizationResult(
            selected_trainsets=service,
            standby_trainsets=standby,
            maintenance_trainsets=maintenance,
            objectives=objectives,
            fitness_score=fitness,
            explanation=explanations
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
        total_improvement = sum(improvements.values()) + 1e-6
        for name in self.optimizers.keys():
            self.selection_probabilities[name] = (improvements[name] + 0.1) / (total_improvement + 0.4)
    
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