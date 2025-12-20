"""
Unit tests for greedyOptim.core.models module.
"""
import pytest
from greedyOptim.core.models import (
    OptimizationResult, OptimizationConfig, TrainsetConstraints,
    ScheduleResult, ScheduleTrainset, ServiceBlock, FleetSummary,
    OptimizationMetrics, ScheduleAlert, TrainStatus, MaintenanceType, AlertSeverity,
    StationStop, Trip
)


class TestOptimizationConfig:
    """Tests for OptimizationConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = OptimizationConfig()
        assert config.required_service_trains == 20
        assert config.min_standby == 2
        assert config.population_size == 100  # Updated to match actual default
        assert config.generations == 200      # Updated to match actual default
        assert config.mutation_rate == 0.1
        assert config.crossover_rate == 0.8
        assert config.elite_size == 5
        assert config.iterations == 15        # Updated to match actual default
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = OptimizationConfig(
            required_service_trains=15,
            min_standby=3,
            population_size=100,
            generations=200
        )
        assert config.required_service_trains == 15
        assert config.min_standby == 3
        assert config.population_size == 100
        assert config.generations == 200


class TestOptimizationResult:
    """Tests for OptimizationResult dataclass."""
    
    def test_creation(self):
        """Test OptimizationResult creation."""
        result = OptimizationResult(
            selected_trainsets=['TS-001', 'TS-002'],
            standby_trainsets=['TS-003'],
            maintenance_trainsets=['TS-004'],
            objectives={'service_availability': 100.0},
            fitness_score=50.0,
            explanation={'TS-001': 'Fit for service'}
        )
        assert len(result.selected_trainsets) == 2
        assert len(result.standby_trainsets) == 1
        assert result.fitness_score == 50.0


class TestTrainsetConstraints:
    """Tests for TrainsetConstraints dataclass."""
    
    def test_creation(self):
        """Test TrainsetConstraints creation with required fields."""
        constraints = TrainsetConstraints(
            has_valid_certificates=True,
            has_critical_jobs=False,
            component_warnings=[],
            maintenance_due=False,
            mileage=50000,
            last_service_days=5
        )
        assert constraints.has_valid_certificates is True
        assert constraints.has_critical_jobs is False
        assert constraints.component_warnings == []
        assert constraints.maintenance_due is False
        assert constraints.mileage == 50000
        assert constraints.last_service_days == 5


class TestEnums:
    """Tests for enum types."""
    
    def test_train_status_values(self):
        """Test TrainStatus enum values."""
        assert TrainStatus.REVENUE_SERVICE == "REVENUE_SERVICE"
        assert TrainStatus.STANDBY == "STANDBY"
        assert TrainStatus.MAINTENANCE == "MAINTENANCE"
    
    def test_alert_severity_values(self):
        """Test AlertSeverity enum values."""
        assert AlertSeverity.LOW == "LOW"
        assert AlertSeverity.MEDIUM == "MEDIUM"
        assert AlertSeverity.HIGH == "HIGH"


class TestScheduleTrainset:
    """Tests for ScheduleTrainset dataclass."""
    
    def test_creation(self):
        """Test ScheduleTrainset creation."""
        trainset = ScheduleTrainset(
            trainset_id='TS-001',
            status=TrainStatus.REVENUE_SERVICE,
            readiness_score=0.95,
            daily_km_allocation=200,
            cumulative_km=50000
        )
        assert trainset.trainset_id == 'TS-001'
        assert trainset.status == TrainStatus.REVENUE_SERVICE
        assert trainset.readiness_score == 0.95


class TestServiceBlock:
    """Tests for ServiceBlock dataclass."""
    
    def test_creation(self):
        """Test ServiceBlock creation."""
        block = ServiceBlock(
            block_id='BLK-001',
            departure_time='07:00',
            origin='Aluva',
            destination='Pettah',
            trip_count=3,
            estimated_km=150
        )
        assert block.block_id == 'BLK-001'
        assert block.trip_count == 3
        assert block.estimated_km == 150


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
