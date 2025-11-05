"""
Advanced optimization algorithms for trainset scheduling.
Includes CMA-ES, Particle Swarm Optimization, and Simulated Annealing.
"""
import numpy as np
from typing import Optional
import math

from .models import OptimizationResult, OptimizationConfig
from .evaluator import TrainsetSchedulingEvaluator


class CMAESOptimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimizer."""
    
    def __init__(self, evaluator: TrainsetSchedulingEvaluator, config: Optional[OptimizationConfig] = None):
        self.evaluator = evaluator
        self.config = config or OptimizationConfig()
        self.n = evaluator.num_trainsets
        self.lam = self.config.population_size  # Population size
        self.mu = self.config.population_size // 2  # Number of parents
        
        # Initialize CMA-ES parameters
        self._initialize_cmaes()
        
    def _initialize_cmaes(self):
        """Initialize CMA-ES strategy parameters."""
        self.mean = np.random.rand(self.n) * 3  # Initial mean in [0, 3)
        self.sigma = 1.0  # Step size
        self.C = np.eye(self.n)  # Covariance matrix
        
        # Strategy parameters
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
        return np.clip(np.round(x), 0, 2).astype(int)
    
    def optimize(self, generations: int = 150) -> OptimizationResult:
        """Run CMA-ES optimization."""
        best_fitness = float('inf')
        best_solution: Optional[np.ndarray] = None
        
        print(f"Starting CMA-ES optimization for {generations} generations")
        
        for gen in range(generations):
            try:
                # Sample population
                population = []
                for _ in range(self.lam):
                    z = np.random.randn(self.n)
                    y = self.mean + self.sigma * (self.C @ z)
                    population.append(y)
                
                population = np.array(population)
                
                # Evaluate
                fitness = []
                decoded_pop = []
                for ind in population:
                    decoded = self._decode(ind)
                    decoded_pop.append(decoded)
                    fitness.append(self.evaluator.fitness_function(decoded))
                
                fitness = np.array(fitness)
                decoded_pop = np.array(decoded_pop)
                
                # Track best
                gen_best_idx = np.argmin(fitness)
                if fitness[gen_best_idx] < best_fitness:
                    best_fitness = fitness[gen_best_idx]
                    best_solution = decoded_pop[gen_best_idx].copy()
                
                if gen % 30 == 0:
                    print(f"Generation {gen}: Best Fitness = {best_fitness:.2f}")
                
                # Selection and recombination
                sorted_indices = np.argsort(fitness)[:self.mu]
                selected = population[sorted_indices]
                
                old_mean = self.mean.copy()
                self.mean = selected.T @ self.weights
                
                # Update evolution paths
                self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mu_eff) * (self.mean - old_mean) / self.sigma
                
                # Fix: Use (gen + 1) to avoid division by zero in first iteration
                hsig = (np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * (gen + 1))) 
                       < (1.4 + 2 / (self.n + 1)) * np.sqrt(self.n))
                
                self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mu_eff) * (self.mean - old_mean) / self.sigma
                
                # Update covariance matrix
                artmp = (selected - old_mean) / self.sigma
                self.C = ((1 - self.c1 - self.cmu) * self.C 
                         + self.c1 * np.outer(self.pc, self.pc)
                         + self.cmu * (artmp.T @ np.diag(self.weights) @ artmp))
                
                # Update step size
                self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / np.sqrt(self.n) - 1))
                
            except Exception as e:
                print(f"Error in CMA-ES generation {gen}: {e}")
                break
        
        if best_solution is None:
            raise RuntimeError("No valid solution found during CMA-ES optimization")
        
        return self._build_result(best_solution, best_fitness)
    
    def _build_result(self, solution: np.ndarray, fitness: float) -> OptimizationResult:
        """Build optimization result from solution."""
        objectives = self.evaluator.calculate_objectives(solution)
        
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


class ParticleSwarmOptimizer:
    """Particle Swarm Optimization for trainset scheduling."""
    
    def __init__(self, evaluator: TrainsetSchedulingEvaluator, config: Optional[OptimizationConfig] = None):
        self.evaluator = evaluator
        self.config = config or OptimizationConfig()
        self.n_particles = self.config.population_size
        self.n_dimensions = evaluator.num_trainsets
        
        # PSO parameters
        self.w = 0.7  # Inertia weight
        self.c1 = 1.5  # Cognitive parameter
        self.c2 = 1.5  # Social parameter
        self.v_max = 2.0  # Maximum velocity (for clamping)
        
    def _decode(self, x: np.ndarray) -> np.ndarray:
        """Decode continuous values to discrete actions."""
        return np.clip(np.round(x), 0, 2).astype(int)
    
    def optimize(self, generations: int = 200) -> OptimizationResult:
        """Run PSO optimization."""
        # Initialize particles
        positions = np.random.uniform(0, 3, (self.n_particles, self.n_dimensions))
        velocities = np.random.uniform(-1, 1, (self.n_particles, self.n_dimensions))
        
        # Personal best positions and fitness
        p_best_positions = positions.copy()
        p_best_fitness = np.array([float('inf')] * self.n_particles)
        
        # Global best
        g_best_position = np.zeros(self.n_dimensions)
        g_best_fitness = float('inf')
        
        print(f"Starting PSO optimization with {self.n_particles} particles for {generations} generations")
        
        for gen in range(generations):
            try:
                for i in range(self.n_particles):
                    # Evaluate particle
                    decoded = self._decode(positions[i])
                    fitness = self.evaluator.fitness_function(decoded)
                    
                    # Update personal best
                    if fitness < p_best_fitness[i]:
                        p_best_fitness[i] = fitness
                        p_best_positions[i] = positions[i].copy()
                        
                        # Update global best
                        if fitness < g_best_fitness:
                            g_best_fitness = fitness
                            g_best_position = positions[i].copy()
                
                # Update velocities and positions
                for i in range(self.n_particles):
                    r1, r2 = np.random.random(2)
                    velocities[i] = (self.w * velocities[i] + 
                                   self.c1 * r1 * (p_best_positions[i] - positions[i]) +
                                   self.c2 * r2 * (g_best_position - positions[i]))
                    
                    # Clamp velocity to prevent explosion
                    velocities[i] = np.clip(velocities[i], -self.v_max, self.v_max)
                    
                    positions[i] += velocities[i]
                    
                    # Bound positions
                    positions[i] = np.clip(positions[i], 0, 3)
                
                if gen % 50 == 0:
                    print(f"Generation {gen}: Best Fitness = {g_best_fitness:.2f}")
                    
            except Exception as e:
                print(f"Error in PSO generation {gen}: {e}")
                break
        
        best_solution = self._decode(g_best_position)
        return self._build_result(best_solution, g_best_fitness)
    
    def _build_result(self, solution: np.ndarray, fitness: float) -> OptimizationResult:
        """Build optimization result from solution."""
        objectives = self.evaluator.calculate_objectives(solution)
        
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


class SimulatedAnnealingOptimizer:
    """Simulated Annealing optimizer for trainset scheduling."""
    
    def __init__(self, evaluator: TrainsetSchedulingEvaluator, config: Optional[OptimizationConfig] = None):
        self.evaluator = evaluator
        self.config = config or OptimizationConfig()
        self.n_dimensions = evaluator.num_trainsets
        
    def _get_neighbor(self, solution: np.ndarray) -> np.ndarray:
        """Generate a neighbor solution by randomly changing one gene."""
        neighbor = solution.copy()
        idx = np.random.randint(0, len(solution))
        neighbor[idx] = np.random.randint(0, 3)
        return neighbor
    
    def _temperature(self, iteration: int, max_iterations: int) -> float:
        """Calculate temperature using exponential cooling."""
        return 100.0 * (0.95 ** iteration)
    
    def optimize(self, max_iterations: int = 10000) -> OptimizationResult:
        """Run Simulated Annealing optimization."""
        # Initialize with a random solution
        current_solution = np.random.randint(0, 3, self.n_dimensions)
        current_fitness = self.evaluator.fitness_function(current_solution)
        
        best_solution = current_solution.copy()
        best_fitness = current_fitness
        
        print(f"Starting Simulated Annealing optimization for {max_iterations} iterations")
        
        for iteration in range(max_iterations):
            try:
                # Generate neighbor
                neighbor = self._get_neighbor(current_solution)
                neighbor_fitness = self.evaluator.fitness_function(neighbor)
                
                # Calculate acceptance probability
                temperature = self._temperature(iteration, max_iterations)
                
                if neighbor_fitness < current_fitness:
                    # Accept better solution
                    current_solution = neighbor
                    current_fitness = neighbor_fitness
                elif temperature > 0:
                    # Accept worse solution with probability
                    delta = neighbor_fitness - current_fitness
                    probability = math.exp(-delta / temperature)
                    if np.random.random() < probability:
                        current_solution = neighbor
                        current_fitness = neighbor_fitness
                
                # Update best solution
                if current_fitness < best_fitness:
                    best_solution = current_solution.copy()
                    best_fitness = current_fitness
                
                if iteration % 1000 == 0:
                    print(f"Iteration {iteration}: Best Fitness = {best_fitness:.2f}, Temperature = {temperature:.2f}")
                    
            except Exception as e:
                print(f"Error in SA iteration {iteration}: {e}")
                break
        
        return self._build_result(best_solution, best_fitness)
    
    def _build_result(self, solution: np.ndarray, fitness: float) -> OptimizationResult:
        """Build optimization result from solution."""
        objectives = self.evaluator.calculate_objectives(solution)
        
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