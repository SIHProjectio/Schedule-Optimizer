"""
Genetic Algorithm optimizer for trainset scheduling.
"""
import numpy as np
from typing import Tuple, Optional

from .models import OptimizationResult, OptimizationConfig
from .evaluator import TrainsetSchedulingEvaluator


class GeneticAlgorithmOptimizer:
    """Genetic Algorithm implementation for trainset scheduling optimization."""
    
    def __init__(self, evaluator: TrainsetSchedulingEvaluator, config: Optional[OptimizationConfig] = None):
        self.evaluator = evaluator
        self.config = config or OptimizationConfig()
        self.n_genes = evaluator.num_trainsets
        
    def initialize_population(self) -> np.ndarray:
        """Initialize random population with smart seeding."""
        population = []
        
        # Seeded solutions (50% of population)
        for _ in range(self.config.population_size // 2):
            solution = np.zeros(self.n_genes, dtype=int)
            
            # Randomly assign required number to service, min to standby, rest to maintenance
            indices = np.random.permutation(self.n_genes)
            service_count = self.config.required_service_trains
            standby_count = self.config.min_standby
            
            solution[indices[:service_count]] = 0  # Service
            solution[indices[service_count:service_count + standby_count]] = 1  # Standby
            solution[indices[service_count + standby_count:]] = 2  # Maintenance
            
            population.append(solution)
        
        # Random solutions (50% of population)
        for _ in range(self.config.population_size // 2):
            solution = np.random.randint(0, 3, self.n_genes)
            population.append(solution)
        
        return np.array(population)
    
    def tournament_selection(self, population: np.ndarray, 
                            fitness: np.ndarray, 
                            tournament_size: int = 5) -> np.ndarray:
        """Tournament selection for parent selection."""
        indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitness = fitness[indices]
        winner_idx = indices[np.argmin(tournament_fitness)]
        return population[winner_idx].copy()
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Two-point crossover with repair mechanism."""
        if np.random.random() > self.config.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        point1, point2 = sorted(np.random.choice(self.n_genes, 2, replace=False))
        
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        child1[point1:point2] = parent2[point1:point2]
        child2[point1:point2] = parent1[point1:point2]
        
        return child1, child2
    
    def mutate(self, solution: np.ndarray) -> np.ndarray:
        """Random mutation with constraint awareness."""
        mutated = solution.copy()
        for i in range(self.n_genes):
            if np.random.random() < self.config.mutation_rate:
                mutated[i] = np.random.randint(0, 3)
        return mutated
    
    def repair_solution(self, solution: np.ndarray) -> np.ndarray:
        """Repair solution to meet basic constraints."""
        repaired = solution.copy()
        
        # Count current assignments
        service_count = np.sum(repaired == 0)
        standby_count = np.sum(repaired == 1)
        
        # If too few in service, convert some from maintenance/standby
        if service_count < self.config.required_service_trains:
            needed = self.config.required_service_trains - service_count
            candidates = np.where((repaired == 1) | (repaired == 2))[0]
            if len(candidates) >= needed:
                selected = np.random.choice(candidates, needed, replace=False)
                repaired[selected] = 0
        
        # If too few in standby, convert some from maintenance
        standby_count = np.sum(repaired == 1)
        if standby_count < self.config.min_standby:
            needed = self.config.min_standby - standby_count
            candidates = np.where(repaired == 2)[0]
            if len(candidates) >= needed:
                selected = np.random.choice(candidates, min(needed, len(candidates)), replace=False)
                repaired[selected] = 1
        
        return repaired
    
    def optimize(self) -> OptimizationResult:
        """Run genetic algorithm optimization."""
        population = self.initialize_population()
        best_fitness = float('inf')
        best_solution: np.ndarray = population[0].copy()
        
        print(f"Starting GA optimization with {self.config.population_size} individuals for {self.config.generations} generations")
        
        for gen in range(self.config.generations):
            try:
                # Evaluate fitness
                fitness = np.array([self.evaluator.fitness_function(ind) for ind in population])
                
                # Track best solution
                gen_best_idx = np.argmin(fitness)
                if fitness[gen_best_idx] < best_fitness:
                    best_fitness = fitness[gen_best_idx]
                    best_solution = population[gen_best_idx].copy()
                
                if gen % 50 == 0:
                    print(f"Generation {gen}: Best Fitness = {best_fitness:.2f}")
                
                # Create new population
                new_population = []
                
                # Elitism - keep best solutions
                elite_indices = np.argsort(fitness)[:self.config.elite_size]
                for idx in elite_indices:
                    new_population.append(population[idx].copy())
                
                # Generate offspring through selection, crossover, and mutation
                while len(new_population) < self.config.population_size:
                    parent1 = self.tournament_selection(population, fitness)
                    parent2 = self.tournament_selection(population, fitness)
                    
                    child1, child2 = self.crossover(parent1, parent2)
                    child1 = self.mutate(child1)
                    child2 = self.mutate(child2)
                    
                    # Repair solutions to meet basic constraints
                    child1 = self.repair_solution(child1)
                    child2 = self.repair_solution(child2)
                    
                    new_population.append(child1)
                    if len(new_population) < self.config.population_size:
                        new_population.append(child2)
                
                population = np.array(new_population)
                
            except Exception as e:
                print(f"Error in generation {gen}: {e}")
                break
        
        # Build result
        return self._build_result(best_solution, best_fitness)
    
    def _build_result(self, solution: np.ndarray, fitness: float) -> OptimizationResult:
        """Build optimization result from solution."""
        objectives = self.evaluator.calculate_objectives(solution)
        
        # Decode solution
        service = [self.evaluator.trainsets[i] for i, v in enumerate(solution) if v == 0]
        standby = [self.evaluator.trainsets[i] for i, v in enumerate(solution) if v == 1]
        maintenance = [self.evaluator.trainsets[i] for i, v in enumerate(solution) if v == 2]
        
        # Generate explanations
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