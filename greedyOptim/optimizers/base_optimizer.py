"""
Base optimizer class providing common functionality for all optimization algorithms.
"""
import numpy as np
from typing import Dict, List, Optional
from abc import ABC, abstractmethod

from greedyOptim.core.models import OptimizationResult, OptimizationConfig
from greedyOptim.scheduling.evaluator import TrainsetSchedulingEvaluator
from greedyOptim.core.utils import (
    decode_solution, create_block_assignment, extract_solution_groups,
    build_block_assignments_dict, mutate_block_assignment
)


class BaseOptimizer(ABC):
    """Base class for all optimization algorithms.
    
    Provides common functionality for:
    - Solution decoding
    - Block assignment management
    - Result building
    """
    
    def __init__(self, evaluator: TrainsetSchedulingEvaluator, config: Optional[OptimizationConfig] = None):
        """Initialize base optimizer.
        
        Args:
            evaluator: TrainsetSchedulingEvaluator instance
            config: Optional configuration parameters
        """
        self.evaluator = evaluator
        self.config = config or OptimizationConfig()
        self.n_trainsets = evaluator.num_trainsets
        self.n_blocks = evaluator.num_blocks
        self.optimize_blocks = self.config.optimize_block_assignment
    
    @abstractmethod
    def optimize(self, **kwargs) -> OptimizationResult:
        """Run optimization algorithm.
        
        Returns:
            OptimizationResult containing the solution
        """
        pass
    
    def decode(self, x: np.ndarray) -> np.ndarray:
        """Decode continuous values to discrete actions."""
        return decode_solution(x)
    
    def create_block_assignment(self, trainset_solution: np.ndarray, randomize: bool = False) -> np.ndarray:
        """Create block assignments for a trainset solution."""
        return create_block_assignment(trainset_solution, self.n_blocks, randomize)
    
    def mutate_block_assignment(self, block_solution: np.ndarray, 
                                  service_indices: np.ndarray) -> np.ndarray:
        """Mutate block assignment."""
        return mutate_block_assignment(block_solution, service_indices, self.config.mutation_rate)
    
    def build_result(
        self,
        solution: np.ndarray,
        fitness: float,
        block_solution: Optional[np.ndarray] = None
    ) -> OptimizationResult:
        """Build optimization result from solution.
        
        Args:
            solution: Trainset assignment array
            fitness: Fitness score
            block_solution: Optional block assignment array
            
        Returns:
            OptimizationResult instance
        """
        objectives = self.evaluator.calculate_objectives(solution)
        
        # Extract groups
        service, standby, maintenance = extract_solution_groups(
            solution, self.evaluator.trainsets
        )
        
        # Generate explanations
        explanations = {}
        for ts_id in service:
            valid, reason = self.evaluator.check_hard_constraints(ts_id)
            explanations[ts_id] = "✓ Fit for service" if valid else f"⚠ {reason}"
        
        # Build block assignments
        block_assignments = {}
        if block_solution is not None and self.optimize_blocks:
            block_assignments = build_block_assignments_dict(
                block_solution, service, self.evaluator.trainsets, self.evaluator.all_blocks
            )
        
        return OptimizationResult(
            selected_trainsets=service,
            standby_trainsets=standby,
            maintenance_trainsets=maintenance,
            objectives=objectives,
            fitness_score=fitness,
            explanation=explanations,
            service_block_assignments=block_assignments
        )
    
    def repair_trainset_solution(self, solution: np.ndarray) -> np.ndarray:
        """Repair solution to meet basic constraints.
        
        Args:
            solution: Current solution
            
        Returns:
            Repaired solution
        """
        repaired = solution.copy()
        
        # Count current assignments
        service_count = np.sum(repaired == 0)
        
        # If too few in service, convert some from maintenance/standby
        target_service = min(
            self.config.required_service_trains,
            self.n_trainsets - self.config.min_standby
        )
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
    
    def evaluate_fitness(self, solution: np.ndarray, 
                          block_solution: Optional[np.ndarray] = None) -> float:
        """Evaluate fitness of a solution.
        
        Args:
            solution: Trainset assignment array
            block_solution: Optional block assignment array
            
        Returns:
            Fitness score
        """
        if self.optimize_blocks and block_solution is not None:
            return self.evaluator.schedule_fitness_function(solution, block_solution)
        return self.evaluator.fitness_function(solution)
