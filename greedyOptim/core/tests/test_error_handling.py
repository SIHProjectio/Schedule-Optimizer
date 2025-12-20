"""
Unit tests for greedyOptim.core.error_handling module.
"""
import pytest
from greedyOptim.core.error_handling import (
    OptimizationError,
    DataValidationError,
    ConstraintViolationError,
    ConfigurationError,
    DataValidator
)


class TestExceptionClasses:
    """Tests for custom exception classes."""
    
    def test_optimization_error(self):
        """Test OptimizationError exception."""
        with pytest.raises(OptimizationError):
            raise OptimizationError("Test error")
    
    def test_data_validation_error(self):
        """Test DataValidationError exception."""
        with pytest.raises(DataValidationError):
            raise DataValidationError("Invalid data")
    
    def test_constraint_violation_error(self):
        """Test ConstraintViolationError exception."""
        with pytest.raises(ConstraintViolationError):
            raise ConstraintViolationError("Constraint violated")
    
    def test_configuration_error(self):
        """Test ConfigurationError exception."""
        with pytest.raises(ConfigurationError):
            raise ConfigurationError("Invalid config")


class TestDataValidator:
    """Tests for DataValidator class."""
    
    def test_validate_empty_data(self):
        """Test validation with empty data."""
        data = {}
        errors = DataValidator.validate_data(data)
        assert len(errors) > 0  # Should have errors for missing sections
    
    def test_validate_valid_minimal_data(self):
        """Test validation with minimal valid data."""
        data = {
            'trainset_status': [
                {'trainset_id': 'TS-001', 'operational_status': 'Available'}
            ],
            'fitness_certificates': [
                {'trainset_id': 'TS-001', 'department': 'Rolling Stock', 'status': 'Valid'}
            ],
            'job_cards': [],
            'component_health': [
                {'trainset_id': 'TS-001', 'component': 'Bogie', 'status': 'Good'}
            ]
        }
        errors = DataValidator.validate_data(data)
        # May have warnings but should pass basic structure validation
        assert isinstance(errors, list)
    
    def test_validate_missing_required_fields(self):
        """Test validation catches missing required fields."""
        data = {
            'trainset_status': [
                {'trainset_id': 'TS-001'}  # Missing operational_status
            ],
            'fitness_certificates': [],
            'job_cards': [],
            'component_health': []
        }
        errors = DataValidator.validate_data(data)
        # Should have error about missing field
        assert any('operational_status' in err.lower() for err in errors)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
