"""
Unit tests for greedyOptim.scheduling.service_blocks module.
"""
import pytest
from greedyOptim.scheduling import ServiceBlockGenerator, create_service_blocks_for_schedule


class TestServiceBlockGenerator:
    """Tests for ServiceBlockGenerator class."""
    
    def test_generator_initialization(self):
        """Test generator initializes correctly."""
        generator = ServiceBlockGenerator()
        assert generator is not None
        assert generator.route_length_km > 0
        assert len(generator.terminals) >= 2
    
    def test_get_all_service_blocks(self):
        """Test getting all service blocks."""
        generator = ServiceBlockGenerator()
        blocks = generator.get_all_service_blocks()
        assert len(blocks) > 0
        
        # Check block structure
        block = blocks[0]
        assert 'block_id' in block
        assert 'departure_time' in block
        assert 'origin' in block
        assert 'destination' in block
        assert 'trip_count' in block
        assert 'estimated_km' in block
    
    def test_get_block_count(self):
        """Test getting block count."""
        generator = ServiceBlockGenerator()
        count = generator.get_block_count()
        assert count > 0
        assert count == len(generator.get_all_service_blocks())
    
    def test_get_peak_block_indices(self):
        """Test getting peak block indices."""
        generator = ServiceBlockGenerator()
        peak_indices = generator.get_peak_block_indices()
        all_blocks = generator.get_all_service_blocks()
        
        # All peak indices should point to peak blocks
        for idx in peak_indices:
            assert all_blocks[idx]['is_peak'] is True
    
    def test_generate_service_blocks_for_train(self):
        """Test generating service blocks for a specific train."""
        generator = ServiceBlockGenerator()
        blocks = generator.generate_service_blocks(0, 10)
        assert len(blocks) > 0
        
        # Check block structure
        for block in blocks:
            assert 'block_id' in block
            assert 'estimated_km' in block


class TestCreateServiceBlocksForSchedule:
    """Tests for create_service_blocks_for_schedule function."""
    
    def test_create_blocks_for_trainsets(self):
        """Test creating blocks for a list of trainsets."""
        trainsets = ['TS-001', 'TS-002', 'TS-003']
        blocks_by_train = create_service_blocks_for_schedule(trainsets)
        
        assert len(blocks_by_train) == 3
        assert 'TS-001' in blocks_by_train
        assert 'TS-002' in blocks_by_train
        assert 'TS-003' in blocks_by_train
        
        # Each trainset should have some blocks
        for ts_id, blocks in blocks_by_train.items():
            assert len(blocks) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
