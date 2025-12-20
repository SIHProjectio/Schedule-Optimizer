"""
Unit tests for greedyOptim.routing.station_loader module.
"""
import pytest
import os
from greedyOptim.routing import (
    StationDataLoader,
    get_station_loader,
    get_route_distance,
    get_terminals,
    Station,
    RouteInfo
)


class TestStationDataLoader:
    """Tests for StationDataLoader class."""
    
    def test_loader_initialization(self):
        """Test loader initializes correctly."""
        loader = StationDataLoader()
        assert loader is not None
    
    def test_load_stations(self):
        """Test loading station data."""
        loader = get_station_loader()
        try:
            stations = loader.stations
            assert len(stations) > 0
            assert isinstance(stations[0], Station)
        except FileNotFoundError:
            pytest.skip("Station data file not found")
    
    def test_get_terminals(self):
        """Test getting terminal stations."""
        try:
            terminals = get_terminals()
            assert len(terminals) >= 2  # Should have at least 2 terminals
            assert isinstance(terminals[0], str)
        except FileNotFoundError:
            pytest.skip("Station data file not found")
    
    def test_get_route_distance(self):
        """Test getting total route distance."""
        try:
            distance = get_route_distance()
            assert distance > 0
            assert isinstance(distance, float)
        except FileNotFoundError:
            pytest.skip("Station data file not found")
    
    def test_station_by_name(self):
        """Test getting station by name."""
        try:
            loader = get_station_loader()
            terminals = loader.terminals
            if terminals:
                station = loader.get_station_by_name(terminals[0])
                assert station is not None
                assert station.is_terminal is True
        except FileNotFoundError:
            pytest.skip("Station data file not found")
    
    def test_distance_between_stations(self):
        """Test calculating distance between stations."""
        try:
            loader = get_station_loader()
            terminals = loader.terminals
            if len(terminals) >= 2:
                distance = loader.get_distance_between(terminals[0], terminals[1])
                assert distance > 0
                assert distance == loader.total_distance_km
        except FileNotFoundError:
            pytest.skip("Station data file not found")


class TestRouteInfo:
    """Tests for RouteInfo dataclass."""
    
    def test_route_info_properties(self):
        """Test RouteInfo properties."""
        try:
            loader = get_station_loader()
            route_info = loader.route_info
            assert isinstance(route_info, RouteInfo)
            assert route_info.name is not None
            assert route_info.operator is not None
            assert len(route_info.stations) > 0
        except FileNotFoundError:
            pytest.skip("Station data file not found")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
