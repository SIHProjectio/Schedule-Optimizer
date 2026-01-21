"""
Unit tests for schedule optimizer
"""
import pytest
from datetime import datetime
from ...core.models import Route, Station, TrainHealthStatus
from ..schedule_optimizer import MetroScheduleOptimizer


@pytest.fixture
def sample_route():
    """Create a sample route for testing"""
    stations = [
        Station(
            station_id=f"ST-{i:03d}",
            name=f"Station {i}",
            sequence=i,
            distance_from_origin_km=float(i * 2.5)
        )
        for i in range(1, 11)
    ]
    
    return Route(
        route_id="TEST-ROUTE",
        name="Test Line",
        stations=stations,
        total_distance_km=25.0,
        avg_speed_kmh=35
    )


@pytest.fixture
def sample_train_health():
    """Create sample train health data"""
    return [
        TrainHealthStatus(
            trainset_id=f"TS-{i:03d}",
            is_fully_healthy=True,
            cumulative_mileage=50000 + i * 1000,
            days_since_maintenance=i,
            component_health={
                "brakes": 0.9,
                "hvac": 0.95,
                "doors": 0.85
            }
        )
        for i in range(1, 11)
    ]


def test_optimizer_initialization(sample_route, sample_train_health):
    """Test MetroScheduleOptimizer initialization"""
    optimizer = MetroScheduleOptimizer(
        date="2025-10-25",
        num_trains=10,
        route=sample_route,
        train_health=sample_train_health,
        depot_name="Test_Depot"
    )
    
    assert optimizer.date == "2025-10-25"
    assert optimizer.num_trains == 10
    assert optimizer.depot_name == "Test_Depot"


def test_optimize_schedule(sample_route, sample_train_health):
    """Test schedule optimization"""
    optimizer = MetroScheduleOptimizer(
        date="2025-10-25",
        num_trains=10,
        route=sample_route,
        train_health=sample_train_health
    )
    
    schedule = optimizer.optimize_schedule(
        min_service_trains=7,
        min_standby=2
    )
    
    assert schedule.schedule_id is not None
    assert len(schedule.trainsets) == 10
    assert schedule.fleet_summary.total_trainsets == 10
    assert schedule.fleet_summary.revenue_service >= 7
