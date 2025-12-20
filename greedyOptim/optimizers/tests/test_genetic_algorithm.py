"""
Unit tests for greedyOptim.optimizers module.
"""
import pytest
import numpy as np
from greedyOptim.optimizers import BaseOptimizer, GeneticAlgorithmOptimizer
from greedyOptim.core.models import OptimizationConfig


class TestGeneticAlgorithmOptimizer:
    """Tests for GeneticAlgorithmOptimizer class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return {
            'trainset_status': [
                {'trainset_id': f'TS-{i+1:03d}', 'operational_status': 'Available', 'total_mileage_km': 50000 + i * 1000}
                for i in range(10)
            ],
            'fitness_certificates': [
                {'trainset_id': f'TS-{i+1:03d}', 'department': 'Rolling Stock', 'status': 'Valid', 'expiry_date': '2025-12-31'}
                for i in range(10)
            ],
            'job_cards': [],
            'component_health': [
                {'trainset_id': f'TS-{i+1:03d}', 'component': 'Bogie', 'status': 'Good', 'wear_level': 30}
                for i in range(10)
            ],
            'branding_contracts': []
        }
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return OptimizationConfig(
            required_service_trains=6,
            min_standby=2,
            population_size=10,
            generations=5
        )
    
    def test_population_initialization(self, sample_data, config):
        """Test population initialization."""
        from greedyOptim.scheduling.evaluator import TrainsetSchedulingEvaluator
        
        evaluator = TrainsetSchedulingEvaluator(sample_data, config)
        optimizer = GeneticAlgorithmOptimizer(evaluator, config)
        
        population = optimizer.initialize_population()
        
        assert len(population) == config.population_size
        assert population.shape[1] == evaluator.num_trainsets
    
    def test_tournament_selection(self, sample_data, config):
        """Test tournament selection."""
        from greedyOptim.scheduling.evaluator import TrainsetSchedulingEvaluator
        
        evaluator = TrainsetSchedulingEvaluator(sample_data, config)
        optimizer = GeneticAlgorithmOptimizer(evaluator, config)
        
        population = optimizer.initialize_population()
        fitness = np.random.rand(len(population)) * 100
        
        selected = optimizer.tournament_selection(population, fitness, tournament_size=3)
        
        assert len(selected) == evaluator.num_trainsets
    
    def test_crossover(self, sample_data, config):
        """Test crossover operation."""
        from greedyOptim.scheduling.evaluator import TrainsetSchedulingEvaluator
        
        evaluator = TrainsetSchedulingEvaluator(sample_data, config)
        optimizer = GeneticAlgorithmOptimizer(evaluator, config)
        
        parent1 = np.zeros(10, dtype=int)
        parent2 = np.ones(10, dtype=int) * 2
        
        child1, child2 = optimizer.crossover(parent1, parent2)
        
        assert len(child1) == 10
        assert len(child2) == 10
    
    def test_mutation(self, sample_data, config):
        """Test mutation operation."""
        from greedyOptim.scheduling.evaluator import TrainsetSchedulingEvaluator
        
        evaluator = TrainsetSchedulingEvaluator(sample_data, config)
        optimizer = GeneticAlgorithmOptimizer(evaluator, config)
        
        solution = np.zeros(10, dtype=int)
        np.random.seed(42)
        
        mutated = optimizer.mutate(solution)
        
        assert len(mutated) == 10
        # With high probability, at least one should be mutated
        # (though this isn't guaranteed)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
