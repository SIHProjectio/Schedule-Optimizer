"""
Advanced optimization algorithms for trainset scheduling.
Includes CMA-ES, Particle Swarm Optimization, and Simulated Annealing.
"""
import numpy as np
from typing import Optional
import math

from greedyOptim.core.models import OptimizationResult, OptimizationConfig
from greedyOptim.scheduling.evaluator import TrainsetSchedulingEvaluator
from .base_optimizer import BaseOptimizer


class CMAESOptimizer(BaseOptimizer):
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimizer."""
    
    def __init__(self, evaluator: TrainsetSchedulingEvaluator, config: Optional[OptimizationConfig] = None):
        self.n = evaluator.num_trainsets
        self.lam = self.config.population_size
        self.mu = self.config.population_size // 2
        
        self._initialize_cmaes()
        
    def _initialize_cmaes(self):
        """Initialize CMA-ES strategy parameters."""
        self.mean = np.random.rand(self.n) * 3
        self.sigma = 1.0
        self.C = np.eye(self.n)
        
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= self.weights.sum()
        self.mu_eff = 1.0 / (self.weights ** 2).sum()
        
        self.cc = 4.0 / (self.n + 4.0)
        self.cs = (self.mu_eff + 2.0) / (self.n + self.mu_eff + 5.0)
        self.c1 = 2.0 / ((self.n + 1.3) ** 2 + self.mu_eff)
        self.cmu = min(1 - self.c1, 2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((self.n + 2) ** 2 + self.mu_eff))
        self.damps = 1.0 + 2.0 * max(0, np.sqrt((self.mu_eff - 1) / (self.n + 1)) - 1) + self.cs
        
        self.pc = np.zeros(self.n)
        self.ps = np.zeros(self.n)
        
    def _decode(self, x: np.ndarray) -> np.ndarray:
        """Decode continuous values to discrete actions."""
        return self.decode(x)
    
    def _create_block_assignment(self, trainset_sol: np.ndarray) -> np.ndarray:
        """Create optimized block assignments for a trainset solution."""
        return self.create_block_assignment(trainset_sol, randomize=True)
    
    def optimize(self, generations: Optional[int] = None, **kwargs) -> OptimizationResult:
        """Run CMA-ES optimization."""
        if generations is None:
            generations = self.config.iterations * 15
        
        best_fitness = float('inf')
        best_solution: Optional[np.ndarray] = None
        best_block_solution: Optional[np.ndarray] = None
        
        print(f"Starting CMA-ES optimization for {generations} generations")
        if self.optimize_blocks:
            print(f"Optimizing block assignments for {self.n_blocks} service blocks")
        
        for gen in range(generations):
            try:
                population = []
                for _ in range(self.lam):
                    z = np.random.randn(self.n)
                    y = self.mean + self.sigma * (self.C @ z)
                    population.append(y)
                
                population = np.array(population)
                
                fitness = []
                decoded_pop = []
                block_pop = []
                
                for ind in population:
                    decoded = self._decode(ind)
                    decoded_pop.append(decoded)
                    
                    if self.optimize_blocks:
                        block_sol = self._create_block_assignment(decoded)
                        block_pop.append(block_sol)
                        fit = self.evaluator.schedule_fitness_function(decoded, block_sol)
                    else:
                        fit = self.evaluator.fitness_function(decoded)
                    
                    fitness.append(fit)
                
                fitness = np.array(fitness)
                decoded_pop = np.array(decoded_pop)
                
                gen_best_idx = np.argmin(fitness)
                if fitness[gen_best_idx] < best_fitness:
                    best_fitness = fitness[gen_best_idx]
                    best_solution = decoded_pop[gen_best_idx].copy()
                    if self.optimize_blocks and block_pop:
                        best_block_solution = block_pop[gen_best_idx].copy()
                
                if gen % 30 == 0:
                    print(f"Generation {gen}: Best Fitness = {best_fitness:.2f}")
                
                sorted_indices = np.argsort(fitness)[:self.mu]
                selected = population[sorted_indices]
                
                old_mean = self.mean.copy()
                self.mean = selected.T @ self.weights
                
                self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mu_eff) * (self.mean - old_mean) / self.sigma
                
                hsig = (np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * (gen + 1))) 
                       < (1.4 + 2 / (self.n + 1)) * np.sqrt(self.n))
                
                self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mu_eff) * (self.mean - old_mean) / self.sigma
                
                artmp = (selected - old_mean) / self.sigma
                self.C = ((1 - self.c1 - self.cmu) * self.C 
                         + self.c1 * np.outer(self.pc, self.pc)
                         + self.cmu * (artmp.T @ np.diag(self.weights) @ artmp))
                
                self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / np.sqrt(self.n) - 1))
                
            except Exception as e:
                print(f"Error in CMA-ES generation {gen}: {e}")
                break
        
        if best_solution is None:
            raise RuntimeError("No valid solution found during CMA-ES optimization")
        
        return self.build_result(best_solution, best_fitness, best_block_solution)


class ParticleSwarmOptimizer(BaseOptimizer):
    """Particle Swarm Optimization for trainset scheduling."""
    
    def __init__(self, evaluator: TrainsetSchedulingEvaluator, config: Optional[OptimizationConfig] = None):
        super().__init__(evaluator, config)
        self.n_particles = self.config.population_size
        self.n_dimensions = evaluator.num_trainsets
        
        # PSO parameters
        self.w = 0.7  # Inertia weight
        self.c1 = 1.5  # Cognitive parameter
        self.c2 = 1.5  # Social parameter
        self.v_max = 2.0  # Maximum velocity (for clamping)
        
    def _decode(self, x: np.ndarray) -> np.ndarray:
        """Decode continuous values to discrete actions."""
        return self.decode(x)
    
    def _create_block_assignment(self, trainset_sol: np.ndarray) -> np.ndarray:
        """Create block assignments for a trainset solution."""
        return self.create_block_assignment(trainset_sol)
    
    def optimize(self, generations: Optional[int] = None, **kwargs) -> OptimizationResult:
        """Run PSO optimization."""
        if generations is None:
            generations = self.config.iterations * 20
        
        positions = np.random.uniform(0, 3, (self.n_particles, self.n_dimensions))
        velocities = np.random.uniform(-1, 1, (self.n_particles, self.n_dimensions))
        
        p_best_positions = positions.copy()
        p_best_fitness = np.array([float('inf')] * self.n_particles)
        p_best_blocks = [None] * self.n_particles
        
        g_best_position = np.zeros(self.n_dimensions)
        g_best_fitness = float('inf')
        g_best_block = None
        
        print(f"Starting PSO optimization with {self.n_particles} particles for {generations} generations")
        if self.optimize_blocks:
            print(f"Optimizing block assignments for {self.n_blocks} service blocks")
        
        for gen in range(generations):
            try:
                for i in range(self.n_particles):
                    # Evaluate particle
                    decoded = self._decode(positions[i])
                    
                    if self.optimize_blocks:
                        block_sol = self._create_block_assignment(decoded)
                        fitness = self.evaluator.schedule_fitness_function(decoded, block_sol)
                    else:
                        block_sol = None
                        fitness = self.evaluator.fitness_function(decoded)
                    
                    if fitness < p_best_fitness[i]:
                        p_best_fitness[i] = fitness
                        p_best_positions[i] = positions[i].copy()
                        p_best_blocks[i] = block_sol.copy() if block_sol is not None else None
                        
                        if fitness < g_best_fitness:
                            g_best_fitness = fitness
                            g_best_position = positions[i].copy()
                            g_best_block = block_sol.copy() if block_sol is not None else None
                
                for i in range(self.n_particles):
                    r1, r2 = np.random.random(2)
                    velocities[i] = (self.w * velocities[i] + 
                                   self.c1 * r1 * (p_best_positions[i] - positions[i]) +
                                   self.c2 * r2 * (g_best_position - positions[i]))
                    
                    velocities[i] = np.clip(velocities[i], -self.v_max, self.v_max)
                    
                    positions[i] += velocities[i]
                    
                    positions[i] = np.clip(positions[i], 0, 3)
                
                if gen % 50 == 0:
                    print(f"Generation {gen}: Best Fitness = {g_best_fitness:.2f}")
                    
            except Exception as e:
                print(f"Error in PSO generation {gen}: {e}")
                break
        
        best_solution = self._decode(g_best_position)
        return self.build_result(best_solution, g_best_fitness, g_best_block)


class SimulatedAnnealingOptimizer(BaseOptimizer):
    """Simulated Annealing optimizer for trainset scheduling."""
    
    def __init__(self, evaluator: TrainsetSchedulingEvaluator, config: Optional[OptimizationConfig] = None):
        super().__init__(evaluator, config)
        self.n_dimensions = evaluator.num_trainsets
        
    def _get_neighbor(self, solution: np.ndarray) -> np.ndarray:
        """Generate a neighbor solution by randomly changing one gene."""
        neighbor = solution.copy()
        idx = np.random.randint(0, len(solution))
        neighbor[idx] = np.random.randint(0, 3)
        return neighbor
    
    def _get_block_neighbor(self, block_sol: np.ndarray, service_indices: np.ndarray) -> np.ndarray:
        """Generate a neighbor block assignment."""
        return self.mutate_block_assignment(block_sol, service_indices)
    
    def _create_block_assignment(self, trainset_sol: np.ndarray) -> np.ndarray:
        """Create block assignments for a trainset solution."""
        return self.create_block_assignment(trainset_sol)
    
    def _temperature(self, iteration: int, max_iterations: int) -> float:
        """Calculate temperature using exponential cooling."""
        return 100.0 * (0.95 ** iteration)
    
    def optimize(self, max_iterations: Optional[int] = None, **kwargs) -> OptimizationResult:
        """Run Simulated Annealing optimization."""
        if max_iterations is None:
            max_iterations = self.config.iterations * 1000
        
        current_solution = np.random.randint(0, 3, self.n_dimensions)
        current_block_sol = self._create_block_assignment(current_solution) if self.optimize_blocks else None
        
        if self.optimize_blocks and current_block_sol is not None:
            current_fitness = self.evaluator.schedule_fitness_function(current_solution, current_block_sol)
        else:
            current_fitness = self.evaluator.fitness_function(current_solution)
        
        best_solution = current_solution.copy()
        best_block_sol = current_block_sol.copy() if current_block_sol is not None else None
        best_fitness = current_fitness
        
        print(f"Starting Simulated Annealing optimization for {max_iterations} iterations")
        if self.optimize_blocks:
            print(f"Optimizing block assignments for {self.n_blocks} service blocks")
        
        for iteration in range(max_iterations):
            try:
                neighbor = self._get_neighbor(current_solution)
                
                if self.optimize_blocks:
                    service_indices = np.where(neighbor == 0)[0]
                    if np.random.random() < 0.3:
                        neighbor_block_sol = self._create_block_assignment(neighbor)
                    else:
                        neighbor_block_sol = self._get_block_neighbor(
                            current_block_sol if current_block_sol is not None else self._create_block_assignment(neighbor),
                            service_indices
                        )
                    neighbor_fitness = self.evaluator.schedule_fitness_function(neighbor, neighbor_block_sol)
                else:
                    neighbor_block_sol = None
                    neighbor_fitness = self.evaluator.fitness_function(neighbor)
                
                temperature = self._temperature(iteration, max_iterations)
                
                if neighbor_fitness < current_fitness:
                    current_solution = neighbor
                    current_fitness = neighbor_fitness
                    current_block_sol = neighbor_block_sol
                elif temperature > 0:
                    delta = neighbor_fitness - current_fitness
                    probability = math.exp(-delta / temperature)
                    if np.random.random() < probability:
                        current_solution = neighbor
                        current_fitness = neighbor_fitness
                        current_block_sol = neighbor_block_sol
                
                if current_fitness < best_fitness:
                    best_solution = current_solution.copy()
                    best_fitness = current_fitness
                    best_block_sol = current_block_sol.copy() if current_block_sol is not None else None
                
                if iteration % 1000 == 0:
                    print(f"Iteration {iteration}: Best Fitness = {best_fitness:.2f}, Temperature = {temperature:.2f}")
                    
            except Exception as e:
                print(f"Error in SA iteration {iteration}: {e}")
                break
        
        return self._build_result(best_solution, best_fitness, best_block_sol)
    
    def _build_result(self, solution: np.ndarray, fitness: float,
                      block_solution: Optional[np.ndarray] = None) -> OptimizationResult:
        """Build optimization result from solution."""
        objectives = self.evaluator.calculate_objectives(solution)
        
        service = [self.evaluator.trainsets[i] for i, v in enumerate(solution) if v == 0]
        standby = [self.evaluator.trainsets[i] for i, v in enumerate(solution) if v == 1]
        maintenance = [self.evaluator.trainsets[i] for i, v in enumerate(solution) if v == 2]
        
        explanations = {}
        for ts_id in service:
            valid, reason = self.evaluator.check_hard_constraints(ts_id)
            explanations[ts_id] = "✓ Fit for service" if valid else f"⚠ {reason}"
        
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