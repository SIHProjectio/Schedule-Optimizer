"""
Unit tests for greedyOptim.core.utils module.
"""
import pytest
import numpy as np
from greedyOptim.core.utils import (
    normalize_certificate_status,
    normalize_component_status,
    normalize_operational_status,
    decode_solution,
    create_block_assignment,
    extract_solution_groups,
    repair_block_assignment,
    mutate_block_assignment,
    build_block_assignments_dict,
    CERTIFICATE_STATUS_MAP,
    COMPONENT_STATUS_MAP,
    OPERATIONAL_STATUS_MAP
)


class TestStatusNormalization:
    """Tests for status normalization functions."""
    
    def test_normalize_certificate_status_valid(self):
        """Test normalizing valid certificate statuses."""
        assert normalize_certificate_status('PENDING') == 'Expiring-Soon'
        assert normalize_certificate_status('ISSUED') == 'Valid'
        assert normalize_certificate_status('EXPIRED') == 'Expired'
    
    def test_normalize_certificate_status_unknown(self):
        """Test normalizing unknown status returns same value."""
        assert normalize_certificate_status('Unknown') == 'Unknown'
    
    def test_normalize_component_status(self):
        """Test normalizing component statuses."""
        assert normalize_component_status('EXCELLENT') == 'Good'
        assert normalize_component_status('CRITICAL') == 'Critical'
        assert normalize_component_status('POOR') == 'Warning'
    
    def test_normalize_operational_status(self):
        """Test normalizing operational statuses."""
        assert normalize_operational_status('IN_SERVICE') == 'In-Service'
        assert normalize_operational_status('MAINTENANCE') == 'Maintenance'
        assert normalize_operational_status('OUT_OF_SERVICE') == 'Out-of-Order'


class TestDecodeSolution:
    """Tests for decode_solution function."""
    
    def test_decode_continuous_values(self):
        """Test decoding continuous values to discrete."""
        x = np.array([0.1, 0.9, 1.2, 1.8, 2.5])
        decoded = decode_solution(x)
        assert np.array_equal(decoded, np.array([0, 1, 1, 2, 2]))
    
    def test_decode_clipping(self):
        """Test that values are clipped to [0, 2]."""
        x = np.array([-1.0, 3.0, 5.0])
        decoded = decode_solution(x)
        assert np.array_equal(decoded, np.array([0, 2, 2]))


class TestBlockAssignment:
    """Tests for block assignment functions."""
    
    def test_create_block_assignment(self):
        """Test creating block assignments."""
        trainset_solution = np.array([0, 0, 1, 2])  # 2 in service
        num_blocks = 10
        block_sol = create_block_assignment(trainset_solution, num_blocks)
        
        # All blocks should be assigned to service trainsets (0 or 1)
        service_indices = np.where(trainset_solution == 0)[0]
        for block_idx in range(num_blocks):
            assert block_sol[block_idx] in service_indices
    
    def test_create_block_assignment_no_service(self):
        """Test block assignment when no trainsets in service."""
        trainset_solution = np.array([1, 2, 2])  # None in service
        num_blocks = 5
        block_sol = create_block_assignment(trainset_solution, num_blocks)
        
        # All should be -1 (unassigned)
        assert np.all(block_sol == -1)


class TestExtractSolutionGroups:
    """Tests for extract_solution_groups function."""
    
    def test_extract_groups(self):
        """Test extracting solution groups."""
        solution = np.array([0, 1, 2, 0, 1])
        trainset_ids = ['TS-001', 'TS-002', 'TS-003', 'TS-004', 'TS-005']
        
        service, standby, maintenance = extract_solution_groups(solution, trainset_ids)
        
        assert service == ['TS-001', 'TS-004']
        assert standby == ['TS-002', 'TS-005']
        assert maintenance == ['TS-003']


class TestRepairBlockAssignment:
    """Tests for repair_block_assignment function."""
    
    def test_repair_invalid_assignments(self):
        """Test repairing block assignments to service trains only."""
        block_solution = np.array([0, 1, 2, 3])  # 2, 3 not in service
        trainset_solution = np.array([0, 0, 1, 2])  # Only 0, 1 in service
        
        np.random.seed(42)
        repaired = repair_block_assignment(block_solution, trainset_solution)
        
        service_indices = np.where(trainset_solution == 0)[0]
        for block_idx in range(len(repaired)):
            assert repaired[block_idx] in service_indices


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
