"""
Unit tests for metro data generator
"""
from ..metro_generator import MetroDataGenerator


def test_generator_initialization():
    """Test MetroDataGenerator initialization"""
    generator = MetroDataGenerator(num_trains=25, num_stations=25)
    assert generator.num_trains == 25
    assert generator.num_stations == 25


def test_generate_route():
    """Test route generation"""
    generator = MetroDataGenerator(num_stations=25)
    route = generator.generate_route("Aluva-Pettah Line")
    
    assert route.name == "Aluva-Pettah Line"
    assert len(route.stations) == 25
    assert route.total_distance_km > 0
    assert route.avg_speed_kmh == 35


def test_generate_train_health():
    """Test train health status generation"""
    generator = MetroDataGenerator(num_trains=10)
    health_statuses = generator.generate_train_health_statuses()
    
    assert len(health_statuses) == 10
    assert all(h.trainset_id.startswith("TS-") for h in health_statuses)
    assert all(h.cumulative_mileage >= 0 for h in health_statuses)


def test_generate_depot_layout():
    """Test depot layout generation"""
    generator = MetroDataGenerator()
    layout = generator.generate_depot_layout()
    
    assert "stabling_bays" in layout
    assert "ibl_bays" in layout
    assert "washing_bays" in layout
    assert len(layout["stabling_bays"]) > 0
