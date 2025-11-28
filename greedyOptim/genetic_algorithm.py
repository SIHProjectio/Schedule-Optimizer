"""
Genetic Algorithm optimizer for trainset scheduling.
"""
import numpy as np
from typing import Tuple, Optional, Dict, List

from .models import OptimizationResult, OptimizationConfig
from .evaluator import TrainsetSchedulingEvaluator


class GeneticAlgorithmOptimizer:
    """Genetic Algorithm implementation for trainset scheduling optimization."""
    
    def __init__(self, evaluator: TrainsetSchedulingEvaluator, config: Optional[OptimizationConfig] = None):
        self.evaluator = evaluator
        self.config = config or OptimizationConfig()
        self.n_genes = evaluator.num_trainsets
        self.n_blocks = evaluator.num_blocks
        self.optimize_blocks = self.config.optimize_block_assignment
        
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
        """Initialize block assignment population.
        
        For each trainset solution, initialize a block assignment solution.
        block_solution[i] = index of trainset assigned to block i, or -1 if unassigned.
        """
        block_population = []
        
        for trainset_sol in trainset_population:
            # Get service train indices
            service_indices = np.where(trainset_sol == 0)[0]
            
            if len(service_indices) == 0:
                # No service trains, all blocks unassigned
                block_sol = np.full(self.n_blocks, -1, dtype=int)
            else:
                # Distribute blocks evenly among service trains
                block_sol = np.zeros(self.n_blocks, dtype=int)
                for i in range(self.n_blocks):
                    # Assign to a random service train
                    block_sol[i] = service_indices[i % len(service_indices)]
            
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
        repaired = solution.copy()
        
        # Count current assignments
        service_count = np.sum(repaired == 0)
        standby_count = np.sum(repaired == 1)
        
        # If too few in service, convert some from maintenance/standby
        target_service = min(self.config.required_service_trains, self.n_genes - self.config.min_standby)
        if service_count < target_service:
            needed = target_service - service_count
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
    
    def repair_block_solution(self, block_sol: np.ndarray, trainset_sol: np.ndarray) -> np.ndarray:
        """Repair block assignments to only assign to service trains."""
        repaired = block_sol.copy()
        service_indices = np.where(trainset_sol == 0)[0]
        
        if len(service_indices) == 0:
            return np.full(self.n_blocks, -1, dtype=int)
        
        for i in range(len(repaired)):
            if repaired[i] not in service_indices:
                # Reassign to a random service train
                repaired[i] = np.random.choice(service_indices)
        
        return repaired
    
    def mutate_block_solution(self, block_sol: np.ndarray, service_indices: np.ndarray) -> np.ndarray:
        """Mutate block assignments."""
        mutated = block_sol.copy()
        
        if len(service_indices) == 0:
            return mutated
        
        for i in range(len(mutated)):
            if np.random.random() < self.config.mutation_rate:
                mutated[i] = np.random.choice(service_indices)
        
        return mutated
    
    def optimize(self) -> OptimizationResult:
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
                        block_child1 = self._create_block_for_trainset(child1)
                        block_child1 = self.mutate_block_solution(block_child1, service_indices_1)
                        
                        new_block_population.append(block_child1)
                        
                        if len(new_block_population) < self.config.population_size:
                            block_child2 = self._create_block_for_trainset(child2)
                            block_child2 = self.mutate_block_solution(block_child2, service_indices_2)
                            new_block_population.append(block_child2)
                
                population = np.array(new_population)
                if self.optimize_blocks:
                    block_population = np.array(new_block_population)
                
            except Exception as e:
                print(f"Error in generation {gen}: {e}")
                break
        
        # Build result
        return self._build_result(best_solution, best_fitness, best_block_solution)
    
    def _create_block_for_trainset(self, trainset_sol: np.ndarray) -> np.ndarray:
        """Create block assignments for a trainset solution."""
        service_indices = np.where(trainset_sol == 0)[0]
        
        if len(service_indices) == 0:
            return np.full(self.n_blocks, -1, dtype=int)
        
        block_sol = np.zeros(self.n_blocks, dtype=int)
        for i in range(self.n_blocks):
            block_sol[i] = service_indices[i % len(service_indices)]
        
        return block_sol
    
    def _build_result(self, solution: np.ndarray, fitness: float, 
                      block_solution: Optional[np.ndarray] = None) -> OptimizationResult:
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
        
        # Build block assignments
        block_assignments = {}
        if block_solution is not None and self.optimize_blocks:
            for ts_id in service:
                block_assignments[ts_id] = []
            
            for block_idx, train_idx in enumerate(block_solution):
                if 0 <= train_idx < len(self.evaluator.trainsets):
                    ts_id = self.evaluator.trainsets[train_idx]
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