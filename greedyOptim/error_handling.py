"""
Error handling and validation utilities for the optimization system.
"""
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json


class OptimizationError(Exception):
    """Base exception for optimization errors."""
    pass


class DataValidationError(OptimizationError):
    """Raised when input data is invalid or malformed."""
    pass


class ConstraintViolationError(OptimizationError):
    """Raised when optimization constraints cannot be satisfied."""
    pass


class ConfigurationError(OptimizationError):
    """Raised when optimization configuration is invalid."""
    pass


class DataValidator:
    """Validates input data for optimization."""
    
    REQUIRED_FIELDS = {
        'trainset_status': ['trainset_id', 'operational_status'],
        'fitness_certificates': ['trainset_id', 'department', 'status'],
        'job_cards': ['trainset_id', 'priority', 'status'],
        'component_health': ['trainset_id', 'component', 'status']
    }
    
    # Accept both legacy and new backend formats
    VALID_STATUSES = {
        'operational': [
            # Legacy format
            'Available', 'In-Service', 'Maintenance', 'Standby', 'Out-of-Order',
            # New backend format
            'IN_SERVICE', 'STANDBY', 'MAINTENANCE', 'OUT_OF_SERVICE', 'TESTING'
        ],
        'certificate': [
            # Legacy format
            'Valid', 'Expired', 'Expiring-Soon', 'Suspended',
            # New backend format
            'PENDING', 'IN_PROGRESS', 'ISSUED', 'EXPIRED', 'SUSPENDED', 
            'REVOKED', 'RENEWED', 'CANCELLED'
        ],
        'job': ['Open', 'In-Progress', 'Closed', 'Pending-Parts'],
        'component': [
            # Legacy format
            'Good', 'Fair', 'Warning', 'Critical',
            # New backend format
            'EXCELLENT', 'GOOD', 'FAIR', 'POOR', 'CRITICAL', 'FAILED'
        ]
    }
    
    # Mapping from backend format to internal format for optimization logic
    STATUS_MAPPINGS = {
        'operational': {
            'IN_SERVICE': 'In-Service',
            'STANDBY': 'Standby',
            'MAINTENANCE': 'Maintenance',
            'OUT_OF_SERVICE': 'Out-of-Order',
            'TESTING': 'Maintenance',  # Treat testing as maintenance for optimization
        },
        'certificate': {
            'PENDING': 'Expiring-Soon',
            'IN_PROGRESS': 'Expiring-Soon',
            'ISSUED': 'Valid',
            'EXPIRED': 'Expired',
            'SUSPENDED': 'Suspended',
            'REVOKED': 'Expired',
            'RENEWED': 'Valid',
            'CANCELLED': 'Expired',
        },
        'component': {
            'EXCELLENT': 'Good',
            'GOOD': 'Good',
            'FAIR': 'Fair',
            'POOR': 'Warning',
            'CRITICAL': 'Critical',
            'FAILED': 'Critical',
        }
    }
    
    @classmethod
    def validate_data(cls, data: Dict) -> List[str]:
        """Validate input data structure and content.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        try:
            # Check required top-level keys (job_cards is now optional)
            required_keys = ['trainset_status', 'fitness_certificates', 'component_health']
            optional_keys = ['job_cards']
            
            for key in required_keys:
                if key not in data:
                    errors.append(f"Missing required data section: {key}")
                    continue
                
                if not isinstance(data[key], list):
                    errors.append(f"Data section {key} must be a list")
                    continue
                
                # Validate individual records
                section_errors = cls._validate_section(data[key], key)
                errors.extend(section_errors)
            
            # Validate optional keys if present
            for key in optional_keys:
                if key in data and data[key]:
                    if not isinstance(data[key], list):
                        errors.append(f"Data section {key} must be a list")
                        continue
                    section_errors = cls._validate_section(data[key], key)
                    errors.extend(section_errors)
            
            # Cross-validation
            if not errors:  # Only if basic structure is valid
                cross_errors = cls._cross_validate(data)
                errors.extend(cross_errors)
                
        except Exception as e:
            errors.append(f"Unexpected error during validation: {str(e)}")
        
        return errors
    
    @classmethod
    def _validate_section(cls, section_data: List[Dict], section_name: str) -> List[str]:
        """Validate a specific data section."""
        errors = []
        required_fields = cls.REQUIRED_FIELDS.get(section_name, [])
        
        for i, record in enumerate(section_data):
            if not isinstance(record, dict):
                errors.append(f"{section_name}[{i}]: Record must be a dictionary")
                continue
            
            # Check required fields
            for field in required_fields:
                if field not in record:
                    errors.append(f"{section_name}[{i}]: Missing required field '{field}'")
                elif record[field] is None or record[field] == "":
                    errors.append(f"{section_name}[{i}]: Field '{field}' cannot be empty")
            
            # Validate specific fields
            validation_errors = cls._validate_record_fields(record, section_name, i)
            errors.extend(validation_errors)
        
        return errors
    
    @classmethod
    def _validate_record_fields(cls, record: Dict, section_name: str, index: int) -> List[str]:
        """Validate specific fields in a record."""
        errors = []
        
        try:
            if section_name == 'trainset_status':
                if 'operational_status' in record:
                    if record['operational_status'] not in cls.VALID_STATUSES['operational']:
                        errors.append(f"{section_name}[{index}]: Invalid operational_status")
                
                if 'total_mileage_km' in record:
                    if not isinstance(record['total_mileage_km'], (int, float)) or record['total_mileage_km'] < 0:
                        errors.append(f"{section_name}[{index}]: total_mileage_km must be non-negative number")
            
            elif section_name == 'fitness_certificates':
                if 'status' in record:
                    if record['status'] not in cls.VALID_STATUSES['certificate']:
                        errors.append(f"{section_name}[{index}]: Invalid certificate status")
                
                # Validate dates
                for date_field in ['issue_date', 'expiry_date']:
                    if date_field in record and record[date_field] is not None:
                        try:
                            datetime.fromisoformat(record[date_field])
                        except ValueError:
                            errors.append(f"{section_name}[{index}]: Invalid {date_field} format")
            
            elif section_name == 'job_cards':
                if 'status' in record:
                    if record['status'] not in cls.VALID_STATUSES['job']:
                        errors.append(f"{section_name}[{index}]: Invalid job status")
                
                if 'priority' in record:
                    if record['priority'] not in ['Critical', 'High', 'Medium', 'Low']:
                        errors.append(f"{section_name}[{index}]: Invalid priority")
            
            elif section_name == 'component_health':
                if 'status' in record:
                    if record['status'] not in cls.VALID_STATUSES['component']:
                        errors.append(f"{section_name}[{index}]: Invalid component status")
                
                if 'wear_level' in record:
                    if not isinstance(record['wear_level'], (int, float)) or not (0 <= record['wear_level'] <= 100):
                        errors.append(f"{section_name}[{index}]: wear_level must be between 0-100")
        
        except Exception as e:
            errors.append(f"{section_name}[{index}]: Error validating record: {str(e)}")
        
        return errors
    
    @classmethod
    def _cross_validate(cls, data: Dict) -> List[str]:
        """Cross-validate data consistency across sections."""
        errors = []
        
        try:
            # Get all trainset IDs
            trainset_ids = {record['trainset_id'] for record in data['trainset_status']}
            
            # Check if all other sections reference valid trainset IDs
            for section_name in ['fitness_certificates', 'job_cards', 'component_health']:
                if section_name in data:
                    for record in data[section_name]:
                        if 'trainset_id' in record:
                            if record['trainset_id'] not in trainset_ids:
                                errors.append(f"{section_name}: References unknown trainset_id '{record['trainset_id']}'")
            
            # Check minimum data requirements
            if len(trainset_ids) < 10:
                errors.append("Insufficient trainsets for optimization (minimum 10 required)")
            
            # Count available trainsets (both legacy and new formats)
            available_statuses = {'Available', 'In-Service', 'Standby', 'IN_SERVICE', 'STANDBY'}
            available_trainsets = sum(1 for record in data['trainset_status'] 
                                    if record.get('operational_status') in available_statuses)
            if available_trainsets < 15:
                errors.append(f"Insufficient available trainsets for optimization ({available_trainsets} available, need at least 15)")
        
        except Exception as e:
            errors.append(f"Error in cross-validation: {str(e)}")
        
        return errors


class ErrorHandler:
    """Centralized error handling for optimization system."""
    
    def __init__(self, log_file: Optional[str] = None):
        self.logger = self._setup_logger(log_file)
    
    def _setup_logger(self, log_file: Optional[str]) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger('optimization')
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_format)
            logger.addHandler(file_handler)
        
        return logger
    
    def validate_and_prepare_data(self, data: Dict) -> Dict:
        """Validate data and prepare for optimization.
        
        Raises:
            DataValidationError: If data validation fails
        """
        self.logger.info("Validating input data...")
        
        try:
            # Validate data structure
            validation_errors = DataValidator.validate_data(data)
            
            if validation_errors:
                error_msg = "Data validation failed:\n" + "\n".join(f"  • {error}" for error in validation_errors)
                self.logger.error(error_msg)
                raise DataValidationError(error_msg)
            
            # Data cleanup and preparation
            cleaned_data = self._clean_data(data)
            
            self.logger.info("Data validation successful")
            return cleaned_data
            
        except DataValidationError:
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error during data validation: {str(e)}")
            raise DataValidationError(f"Unexpected validation error: {str(e)}")
    
    def _clean_data(self, data: Dict) -> Dict:
        """Clean and prepare data for optimization."""
        cleaned_data = data.copy()
        
        try:
            # Remove records with missing critical data
            for section_name in ['trainset_status', 'fitness_certificates', 'job_cards', 'component_health']:
                if section_name in cleaned_data:
                    original_count = len(cleaned_data[section_name])
                    cleaned_data[section_name] = [
                        record for record in cleaned_data[section_name]
                        if record.get('trainset_id') is not None
                    ]
                    removed_count = original_count - len(cleaned_data[section_name])
                    if removed_count > 0:
                        self.logger.warning(f"Removed {removed_count} records from {section_name} due to missing trainset_id")
            
            # Ensure consistent trainset IDs across sections
            valid_trainset_ids = {record['trainset_id'] for record in cleaned_data['trainset_status']}
            
            for section_name in ['fitness_certificates', 'job_cards', 'component_health']:
                if section_name in cleaned_data:
                    original_count = len(cleaned_data[section_name])
                    cleaned_data[section_name] = [
                        record for record in cleaned_data[section_name]
                        if record.get('trainset_id') in valid_trainset_ids
                    ]
                    removed_count = original_count - len(cleaned_data[section_name])
                    if removed_count > 0:
                        self.logger.warning(f"Removed {removed_count} records from {section_name} with invalid trainset_id")
            
            return cleaned_data
            
        except Exception as e:
            self.logger.error(f"Error during data cleaning: {str(e)}")
            raise DataValidationError(f"Data cleaning failed: {str(e)}")
    
    def handle_optimization_error(self, error: Exception, context: str = "") -> None:
        """Handle optimization errors with appropriate logging and re-raising."""
        error_msg = f"Optimization error{' in ' + context if context else ''}: {str(error)}"
        
        if isinstance(error, (DataValidationError, ConstraintViolationError, ConfigurationError)):
            self.logger.error(error_msg)
            raise
        else:
            self.logger.exception(f"Unexpected {error_msg}")
            raise OptimizationError(f"Unexpected error: {str(error)}")
    
    def log_optimization_start(self, method: str, config: Any) -> None:
        """Log optimization start with parameters."""
        self.logger.info(f"Starting optimization with method: {method}")
        if hasattr(config, '__dict__'):
            for key, value in config.__dict__.items():
                self.logger.info(f"  {key}: {value}")
    
    def log_optimization_result(self, result: Any, method: str) -> None:
        """Log optimization results."""
        if hasattr(result, 'fitness_score') and hasattr(result, 'selected_trainsets'):
            self.logger.info(f"Optimization completed with {method}")
            self.logger.info(f"  Fitness score: {result.fitness_score:.2f}")
            self.logger.info(f"  Service trainsets: {len(result.selected_trainsets)}")
            self.logger.info(f"  Standby trainsets: {len(getattr(result, 'standby_trainsets', []))}")
            self.logger.info(f"  Maintenance trainsets: {len(getattr(result, 'maintenance_trainsets', []))}")
        else:
            self.logger.info(f"Optimization completed with {method}")


def safe_optimize(data: Dict, method: str = 'ga', config: Any = None, 
                 log_file: Optional[str] = None, **kwargs) -> Any:
    """Safely run optimization with comprehensive error handling.
    
    Args:
        data: Input data dictionary
        method: Optimization method
        config: Optimization configuration
        log_file: Path to log file (optional)
        **kwargs: Additional method-specific parameters
    
    Returns:
        OptimizationResult
        
    Raises:
        OptimizationError: For any optimization-related errors
    """
    error_handler = ErrorHandler(log_file)
    
    try:
        # Validate and prepare data
        cleaned_data = error_handler.validate_and_prepare_data(data)
        
        # Import here to avoid circular imports
        from .scheduler import optimize_trainset_schedule
        
        # Log optimization start
        error_handler.log_optimization_start(method, config)
        
        # Run optimization
        result = optimize_trainset_schedule(cleaned_data, method, config, **kwargs)
        
        # Log results
        error_handler.log_optimization_result(result, method)
        
        return result
        
    except (DataValidationError, ConstraintViolationError, ConfigurationError):
        raise
    except Exception as e:
        error_handler.handle_optimization_error(e, f"method={method}")


# Usage example
if __name__ == "__main__":
    # Test data validation
    test_data = {
        "trainset_status": [
            {"trainset_id": "TS-001", "operational_status": "Available", "total_mileage_km": 150000},
            {"trainset_id": "TS-002", "operational_status": "Invalid"}  # This should cause an error
        ],
        "fitness_certificates": [
            {"trainset_id": "TS-001", "department": "Rolling Stock", "status": "Valid"}
        ],
        "job_cards": [],
        "component_health": []
    }
    
    # Validate data
    errors = DataValidator.validate_data(test_data)
    
    if errors:
        print("Validation errors found:")
        for error in errors:
            print(f"  • {error}")
    else:
        print("✅ Data validation passed!")
        
    # Test error handler
    try:
        handler = ErrorHandler()
        cleaned_data = handler.validate_and_prepare_data(test_data)
        print("✅ Data preparation successful!")
    except DataValidationError as e:
        print(f"❌ Data validation failed: {e}")