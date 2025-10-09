"""
Refactored base.py - now uses modular structure for better maintainability.

This module has been restructured into separate files:
- models.py: Data classes and configurations
- evaluator.py: Constraint checking and objective evaluation  
- genetic_algorithm.py: Genetic Algorithm implementation
- advanced_optimizers.py: CMA-ES, PSO, and Simulated Annealing
- scheduler.py: Main interface and comparison tools

For backward compatibility, the main functions are still available here.
"""
import json
from typing import Dict, Tuple, Optional
import numpy as np
from datetime import datetime

# Import from new modular structure
from .scheduler import optimize_trainset_schedule, compare_optimization_methods
from .models import OptimizationResult, OptimizationConfig

# For backward compatibility, expose the main function with original signature
def optimize_trainset_schedule_main(data: Dict, method: str = 'ga') -> OptimizationResult:
    """Multi-objective optimizer for trainset scheduling using genetic algorithm.
    
    This is a backward compatibility wrapper around the new modular system.
    """
    # Use the new modular system
    config = OptimizationConfig()
    return optimize_trainset_schedule(data, method, config)


# Usage example
if __name__ == "__main__":
    # Load synthetic data
    try:
        with open('metro_synthetic_data.json', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Please generate synthetic data first")
        exit(1)
    
    # Run optimization with Genetic Algorithm (backward compatibility)
    result_ga = optimize_trainset_schedule_main(data, method='ga')
    
    print(f"\nOptimization completed!")
    print(f"Service trainsets: {len(result_ga.selected_trainsets)}")
    print(f"Standby trainsets: {len(result_ga.standby_trainsets)}")
    print(f"Maintenance trainsets: {len(result_ga.maintenance_trainsets)}")
    print(f"Final fitness score: {result_ga.fitness_score:.2f}")
    
    # You can also use the new modular interface
    print(f"\nFor more advanced features, use:")
    print(f"from greedyOptim import optimize_trainset_schedule, compare_optimization_methods")
        
        # Fitness certificates by trainset and department
        self.fitness_map = {}
        for cert in self.data['fitness_certificates']:
            ts_id = cert['trainset_id']
            if ts_id not in self.fitness_map:
                self.fitness_map[ts_id] = {}
            self.fitness_map[ts_id][cert['department']] = cert
        
        # Job cards by trainset
        self.job_map = {}
        for job in self.data['job_cards']:
            ts_id = job['trainset_id']
            if ts_id not in self.job_map:
                self.job_map[ts_id] = []
            self.job_map[ts_id].append(job)
        
        # Component health by trainset
        self.health_map = {}
        for health in self.data['component_health']:
            ts_id = health['trainset_id']
            if ts_id not in self.health_map:
                self.health_map[ts_id] = []
            self.health_map[ts_id].append(health)
        
        # Branding contracts
        self.brand_map = {b['trainset_id']: b for b in self.data['branding_contracts']}
        
        # Maintenance schedule
        self.maint_map = {m['trainset_id']: m for m in self.data['maintenance_schedule']}
    
    def check_hard_constraints(self, trainset_id: str) -> Tuple[bool, str]:
        """Check if trainset passes hard constraints"""
        
        # Check fitness certificates
        if trainset_id in self.fitness_map:
            for dept, cert in self.fitness_map[trainset_id].items():
                if cert['status'] in ['Expired']:
                    return False, f"Expired {dept} certificate"
                expiry = datetime.fromisoformat(cert['expiry_date'])
                if expiry < datetime.now():
                    return False, f"{dept} certificate expired"
        else:
            return False, "Missing fitness certificates"
        
        # Check open critical job cards
        if trainset_id in self.job_map:
            for job in self.job_map[trainset_id]:
                if job['status'] == 'Open' and job['priority'] == 'Critical':
                    return False, f"Critical job card {job['job_card_id']} open"
        
        # Check component health
        if trainset_id in self.health_map:
            for health in self.health_map[trainset_id]:
                if health['status'] == 'Warning' and health['wear_level'] > 90:
                    return False, f"{health['component']} critical wear"
        
        return True, "Pass"
    
    def calculate_objectives(self, solution: np.ndarray) -> Dict[str, float]:
        """Calculate multiple objectives for a solution
        
        Solution encoding: 0=Service, 1=Standby, 2=Maintenance
        """
        objectives = {
            'service_availability': 0.0,
            'maintenance_cost': 0.0,
            'branding_compliance': 0.0,
            'mileage_balance': 0.0,
            'constraint_penalty': 0.0
        }
        
        service_trains = []
        standby_trains = []
        maint_trains = []
        
        for idx, action in enumerate(solution):
            ts_id = self.trainsets[idx]
            if action == 0:
                service_trains.append(ts_id)
            elif action == 1:
                standby_trains.append(ts_id)
            else:
                maint_trains.append(ts_id)
        
        # Objective 1: Service Availability (maximize)
        # Penalty if not meeting required count
        availability = len(service_trains) / self.required_service_trains
        if len(service_trains) < self.required_service_trains:
            objectives['constraint_penalty'] += (self.required_service_trains - len(service_trains)) * 100
        objectives['service_availability'] = min(availability, 1.0) * 100
        
        # Objective 2: Maintenance Cost (minimize via mileage balancing)
        mileages = [self.status_map[ts]['total_mileage_km'] for ts in service_trains]
        if mileages:
            std_dev = float(np.std(mileages))
            objectives['mileage_balance'] = 100.0 - min(std_dev / 1000.0, 100.0)  # Lower std = better balance
        
        # Objective 3: Branding Compliance (maximize)
        brand_scores = []
        for ts_id in service_trains:
            if ts_id in self.brand_map:
                contract = self.brand_map[ts_id]
                target = contract['daily_target_hours']
                actual = contract['actual_exposure_hours'] / 30  # Daily average
                compliance = min(actual / target, 1.0) if target > 0 else 1.0
                brand_scores.append(compliance)
        
        objectives['branding_compliance'] = float(np.mean(brand_scores)) * 100.0 if brand_scores else 100.0
        
        # Objective 4: Maintenance Cost (minimize)
        # Trains needing maintenance but not scheduled increase cost
        maint_cost = 0
        for ts_id in service_trains:
            if ts_id in self.maint_map:
                if self.maint_map[ts_id]['status'] == 'Overdue':
                    maint_cost += 50  # Penalty for overdue maintenance
        objectives['maintenance_cost'] = 100 - min(maint_cost, 100)
        
        # Hard constraint violations
        for ts_id in service_trains:
            valid, reason = self.check_hard_constraints(ts_id)
            if not valid:
                objectives['constraint_penalty'] += 200
        
        # Standby constraint
        if len(standby_trains) < self.min_standby:
            objectives['constraint_penalty'] += (self.min_standby - len(standby_trains)) * 50
        
        return objectives
    
    def fitness_function(self, solution: np.ndarray) -> float:
        """Aggregate fitness function (minimize)"""
        obj = self.calculate_objectives(solution)
        
        # Weighted sum (convert to minimization)
        fitness = (
            -obj['service_availability'] * 2.0 +      # Maximize (negative weight)
            -obj['branding_compliance'] * 1.5 +        # Maximize
            -obj['mileage_balance'] * 1.0 +            # Maximize
            -obj['maintenance_cost'] * 1.0 +           # Maximize
            obj['constraint_penalty'] * 5.0            # Minimize (positive weight)
        )
        
        return fitness


class GeneticAlgorithmOptimizer:
    """Custom Genetic Algorithm implementation"""
    
    def __init__(self, evaluator: TrainsetSchedulingOptimizer, 
                 population_size: int = 100, 
                 generations: int = 200):
        self.evaluator = evaluator
        self.pop_size = population_size
        self.generations = generations
        self.n_genes = evaluator.num_trainsets
        
        # GA parameters
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        self.elite_size = 5
        
    def initialize_population(self) -> np.ndarray:
        """Initialize random population"""
        population = []
        
        # Seeded solutions
        for _ in range(self.pop_size // 2):
            solution = np.zeros(self.n_genes, dtype=int)
            
            # Randomly assign 20 to service, 2+ to standby, rest to maintenance
            indices = np.random.permutation(self.n_genes)
            solution[indices[:20]] = 0  # Service
            solution[indices[20:23]] = 1  # Standby
            solution[indices[23:]] = 2  # Maintenance
            
            population.append(solution)
        
        # Random solutions
        for _ in range(self.pop_size // 2):
            solution = np.random.randint(0, 3, self.n_genes)
            population.append(solution)
        
        return np.array(population)
    
    def tournament_selection(self, population: np.ndarray, 
                            fitness: np.ndarray, 
                            tournament_size: int = 5) -> np.ndarray:
        """Tournament selection"""
        indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitness = fitness[indices]
        winner_idx = indices[np.argmin(tournament_fitness)]
        return population[winner_idx].copy()
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Two-point crossover"""
        if np.random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        point1, point2 = sorted(np.random.choice(self.n_genes, 2, replace=False))
        
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        child1[point1:point2] = parent2[point1:point2]
        child2[point1:point2] = parent1[point1:point2]
        
        return child1, child2
    
    def mutate(self, solution: np.ndarray) -> np.ndarray:
        """Random mutation"""
        mutated = solution.copy()
        for i in range(self.n_genes):
            if np.random.random() < self.mutation_rate:
                mutated[i] = np.random.randint(0, 3)
        return mutated
    
    def optimize(self) -> OptimizationResult:
        """Run genetic algorithm"""
        population = self.initialize_population()
        best_fitness = float('inf')
        best_solution: Optional[np.ndarray] = None
        
        for gen in range(self.generations):
            # Evaluate fitness
            fitness = np.array([self.evaluator.fitness_function(ind) for ind in population])
            
            # Track best
            gen_best_idx = np.argmin(fitness)
            if fitness[gen_best_idx] < best_fitness:
                best_fitness = fitness[gen_best_idx]
                best_solution = population[gen_best_idx].copy()
            
            if gen % 50 == 0:
                print(f"Generation {gen}: Best Fitness = {best_fitness:.2f}")
            
            # Selection and reproduction
            new_population = []
            
            # Elitism
            elite_indices = np.argsort(fitness)[:self.elite_size]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
            
            # Generate offspring
            while len(new_population) < self.pop_size:
                parent1 = self.tournament_selection(population, fitness)
                parent2 = self.tournament_selection(population, fitness)
                
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.append(child1)
                if len(new_population) < self.pop_size:
                    new_population.append(child2)
            
            population = np.array(new_population)
        
        # Final evaluation
        if best_solution is None:
            raise RuntimeError("No valid solution found during optimization")
            
        objectives = self.evaluator.calculate_objectives(best_solution)
        
        # Decode solution
        service = [self.evaluator.trainsets[i] for i, v in enumerate(best_solution) if v == 0]
        standby = [self.evaluator.trainsets[i] for i, v in enumerate(best_solution) if v == 1]
        maintenance = [self.evaluator.trainsets[i] for i, v in enumerate(best_solution) if v == 2]
        
        # Generate explanations
        explanations = {}
        for ts_id in service:
            valid, reason = self.evaluator.check_hard_constraints(ts_id)
            status = "✓ Fit for service" if valid else f"⚠ {reason}"
            explanations[ts_id] = status
        
        return OptimizationResult(
            selected_trainsets=service,
            standby_trainsets=standby,
            maintenance_trainsets=maintenance,
            objectives=objectives,
            fitness_score=best_fitness,
            explanation=explanations
        )


class CMAESOptimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy)"""
    
    def __init__(self, evaluator: TrainsetSchedulingOptimizer, population_size: int = 50):
        self.evaluator = evaluator
        self.n = evaluator.num_trainsets
        self.lam = population_size  # Population size
        self.mu = population_size // 2  # Number of parents
        
        # Initialize
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
        """Decode continuous values to discrete actions"""
        return np.clip(np.round(x), 0, 2).astype(int)
    
    def optimize(self, generations: int = 150) -> OptimizationResult:
        """Run CMA-ES optimization"""
        best_fitness = float('inf')
        best_solution: Optional[np.ndarray] = None
        
        for gen in range(generations):
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
            
            hsig = (np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * gen)) 
                   < (1.4 + 2 / (self.n + 1)) * np.sqrt(self.n))
            
            self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mu_eff) * (self.mean - old_mean) / self.sigma
            
            # Update covariance matrix
            artmp = (selected - old_mean) / self.sigma
            self.C = ((1 - self.c1 - self.cmu) * self.C 
                     + self.c1 * np.outer(self.pc, self.pc)
                     + self.cmu * (artmp.T @ np.diag(self.weights) @ artmp))
            
            # Update step size
            self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / np.sqrt(self.n) - 1))
        
        if best_solution is None:
            raise RuntimeError("No valid solution found during CMA-ES optimization")
            
        objectives = self.evaluator.calculate_objectives(best_solution)
        
        service = [self.evaluator.trainsets[i] for i, v in enumerate(best_solution) if v == 0]
        standby = [self.evaluator.trainsets[i] for i, v in enumerate(best_solution) if v == 1]
        maintenance = [self.evaluator.trainsets[i] for i, v in enumerate(best_solution) if v == 2]
        
        explanations = {}
        for ts_id in service:
            valid, reason = self.evaluator.check_hard_constraints(ts_id)
            explanations[ts_id] = "✓ Fit for service" if valid else f"⚠ {reason}"
        
        return OptimizationResult(
            selected_trainsets=service,
            standby_trainsets=standby,
            maintenance_trainsets=maintenance,
            objectives=objectives,
            fitness_score=best_fitness,
            explanation=explanations
        )


def optimize_trainset_schedule(data: Dict, method: str = 'ga') -> OptimizationResult:
    """Main optimization function
    
    Args:
        data: Synthetic metro data dictionary
        method: 'ga' for Genetic Algorithm, 'cmaes' for CMA-ES
    """
    evaluator = TrainsetSchedulingOptimizer(data)
    
    print(f"\nOptimizing with {method.upper()}...")
    print(f"Trainsets: {evaluator.num_trainsets}")
    print(f"Required in service: {evaluator.required_service_trains}")
    print(f"Minimum standby: {evaluator.min_standby}\n")
    
    if method == 'ga':
        optimizer = GeneticAlgorithmOptimizer(evaluator, population_size=100, generations=200)
    elif method == 'cmaes':
        optimizer = CMAESOptimizer(evaluator, population_size=50)
        return optimizer.optimize(generations=150)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    result = optimizer.optimize()
    
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)
    print(f"\nService Trainsets ({len(result.selected_trainsets)}):")
    for ts in result.selected_trainsets[:5]:
        print(f"  {ts}: {result.explanation.get(ts, 'N/A')}")
    if len(result.selected_trainsets) > 5:
        print(f"  ... and {len(result.selected_trainsets) - 5} more")
    
    print(f"\nStandby Trainsets ({len(result.standby_trainsets)}):")
    for ts in result.standby_trainsets:
        print(f"  {ts}")
    
    print(f"\nMaintenance Trainsets ({len(result.maintenance_trainsets)}):")
    for ts in result.maintenance_trainsets:
        print(f"  {ts}")
    
    print(f"\nObjectives:")
    for key, value in result.objectives.items():
        print(f"  {key}: {value:.2f}")
    
    print(f"\nFitness Score: {result.fitness_score:.2f}")
    
    return result


# Usage example
if __name__ == "__main__":
    # Load synthetic data
    with open('metro_synthetic_data.json', 'r') as f:
        data = json.load(f)
    
    # Run optimization with Genetic Algorithm
    result_ga = optimize_trainset_schedule(data, method='ga')
    
    # Run optimization with CMA-ES
    result_cmaes = optimize_trainset_schedule(data, method='cmaes')