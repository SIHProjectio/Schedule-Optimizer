"""
Genetic Algorithm optimizer for trainset scheduling.
"""
import numpy as np
from typing import Tuple, Optional

from greedyOptim.core.models import OptimizationResult, OptimizationConfig
from greedyOptim.scheduling.evaluator import TrainsetSchedulingEvaluator
from .base_optimizer import BaseOptimizer


class GeneticAlgorithmOptimizer(BaseOptimizer):
    """Genetic Algorithm implementation for trainset scheduling optimization."""
    
    def __init__(self, evaluator: TrainsetSchedulingEvaluator, config: Optional[OptimizationConfig] = None):
        super().__init__(evaluator, config)
        self.n_genes = evaluator.num_trainsets
        
    def initialize_population(self) -> np.ndarray:
        """Initialize random population with smart seeding."""
        population = []
        
        # Seeded solutions (50% of population)
        for _ in range(self.config.population_size // 2):
            solution = np.zeros(self.n_genes, dtype=int)
            
            # Randomly assign required number to service, min to standby, rest to maintenance
            indices = np.random.permutation(self.n_genes)
            service_count = min(self.config.required_service_trains, self.n_genes)
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
    
    def initialize_block_population(self, trainset_population: np.ndarray) -> np.ndarray:
        """Initialize block assignment population."""
        block_population = []
        
        for trainset_sol in trainset_population:
            block_sol = self.create_block_assignment(trainset_sol)
            block_population.append(block_sol)
        
        return np.array(block_population)
    
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
        return self.repair_trainset_solution(solution)
    
    def repair_block_solution(self, block_sol: np.ndarray, trainset_sol: np.ndarray) -> np.ndarray:
        """Repair block assignments to only assign to service trains."""
        from greedyOptim.core.utils import repair_block_assignment
        return repair_block_assignment(block_sol, trainset_sol)
    
    def mutate_block_solution(self, block_sol: np.ndarray, service_indices: np.ndarray) -> np.ndarray:
        """Mutate block assignments."""
        return self.mutate_block_assignment(block_sol, service_indices)
    
    def optimize(self, **kwargs) -> OptimizationResult:
        """Run genetic algorithm optimization."""
        population = self.initialize_population()
        
        # Initialize block population if optimizing blocks
        block_population = None
        if self.optimize_blocks:
            block_population = self.initialize_block_population(population)
        
        best_fitness = float('inf')
        best_solution: np.ndarray = population[0].copy()
        best_block_solution: Optional[np.ndarray] = None
        
        print(f"Starting GA optimization with {self.config.population_size} individuals for {self.config.generations} generations")
        if self.optimize_blocks:
            print(f"Optimizing block assignments for {self.n_blocks} service blocks")
        
        for gen in range(self.config.generations):
            try:
                # Evaluate fitness
                if self.optimize_blocks and block_population is not None:
                    fitness = np.array([
                        self.evaluator.schedule_fitness_function(population[i], block_population[i])
                        for i in range(len(population))
                    ])
                else:
                    fitness = np.array([self.evaluator.fitness_function(ind) for ind in population])
                
                # Track best solution
                gen_best_idx = np.argmin(fitness)
                if fitness[gen_best_idx] < best_fitness:
                    best_fitness = fitness[gen_best_idx]
                    best_solution = population[gen_best_idx].copy()
                    if self.optimize_blocks and block_population is not None:
                        best_block_solution = block_population[gen_best_idx].copy()
                
                if gen % 50 == 0:
                    print(f"Generation {gen}: Best Fitness = {best_fitness:.2f}")
                
                # Create new population
                new_population = []
                new_block_population = [] if self.optimize_blocks else None
                
                # Elitism - keep best solutions
                elite_indices = np.argsort(fitness)[:self.config.elite_size]
                for idx in elite_indices:
                    new_population.append(population[idx].copy())
                    if self.optimize_blocks and block_population is not None:
                        new_block_population.append(block_population[idx].copy())
                
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
                    
                    # Handle block solutions
                    if self.optimize_blocks and new_block_population is not None:
                        service_indices_1 = np.where(child1 == 0)[0]
                        service_indices_2 = np.where(child2 == 0)[0]
                        
                        # Create new block assignments for children
                        block_child1 = self.create_block_assignment(child1)
                        block_child1 = self.mutate_block_solution(block_child1, service_indices_1)
                        
                        new_block_population.append(block_child1)
                        
                        if len(new_block_population) < self.config.population_size:
                            block_child2 = self.create_block_assignment(child2)
                            block_child2 = self.mutate_block_solution(block_child2, service_indices_2)
                            new_block_population.append(block_child2)
                
                population = np.array(new_population)
                if self.optimize_blocks:
                    block_population = np.array(new_block_population)
                
            except Exception as e:
                print(f"Error in generation {gen}: {e}")
                break
        
        return self.build_result(best_solution, best_fitness, best_block_solution)
    
