"""
Station Data Loader
Loads and manages station/route information from JSON configuration.
Enables flexible route configuration for different metro lines.
"""
import json
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from functools import lru_cache


@dataclass
class Station:
    """Represents a metro station."""
    sr_no: int
    code: str
    name: str
    distance_from_prev_km: float
    cumulative_distance_km: float
    is_terminal: bool
    has_depot: bool
    platform_count: int
    interchange: Optional[str] = None
    depot_name: Optional[str] = None


@dataclass
class RouteInfo:
    """Complete route information."""
    name: str
    operator: str
    stations: List[Station]
    total_distance_km: float
    terminal_stations: List[str]
    depot_stations: List[str]
    operational_params: Dict


class StationDataLoader:
    """Loads and manages station data from JSON configuration."""
    
    # Default path to station data (absolute path from project root)
    DEFAULT_DATA_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'data', 
        'metro_stations.json'
    )
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize station data loader.
        
        Args:
            config_path: Path to station JSON file. If None, uses default.
        """
        self.config_path = config_path or self.DEFAULT_DATA_PATH
        self._data: Optional[Dict] = None
        self._stations: Optional[List[Station]] = None
        self._route_info: Optional[RouteInfo] = None
        
    def load(self) -> Optional[Dict]:
        """Load station data from JSON file."""
        if self._data is not None:
            return self._data
            
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(
                f"Station data file not found: {self.config_path}\n"
                f"Please ensure the station configuration exists."
            )
        
        with open(self.config_path, 'r') as f:
            self._data = json.load(f)
        
        return self._data
    
    @property
    def stations(self) -> List[Station]:
        """Get list of Station objects."""
        if self._stations is not None:
            return self._stations
        
        data = self.load()
        self._stations = []
        
        for s in data['stations']:
            station = Station(
                sr_no=s['sr_no'],
                code=s['code'],
                name=s['name'],
                distance_from_prev_km=s['distance_from_prev_km'],
                cumulative_distance_km=s['cumulative_distance_km'],
                is_terminal=s['is_terminal'],
                has_depot=s['has_depot'],
                platform_count=s['platform_count'],
                interchange=s.get('interchange'),
                depot_name=s.get('depot_name')
            )
            self._stations.append(station)
        
        return self._stations
    
    @property
    def route_info(self) -> RouteInfo:
        """Get complete route information."""
        if self._route_info is not None:
            return self._route_info
        
        data = self.load()
        stations = self.stations
        
        self._route_info = RouteInfo(
            name=data['line_info']['name'],
            operator=data['line_info']['operator'],
            stations=stations,
            total_distance_km=stations[-1].cumulative_distance_km if stations else 0,
            terminal_stations=[s.name for s in stations if s.is_terminal],
            depot_stations=[s.name for s in stations if s.has_depot],
            operational_params=data.get('operational_params', {})
        )
        
        return self._route_info
    
    @property
    def total_distance_km(self) -> float:
        """Get total route distance in km."""
        return self.route_info.total_distance_km
    
    @property
    def terminals(self) -> List[str]:
        """Get terminal station names."""
        return self.route_info.terminal_stations
    
    @property
    def station_count(self) -> int:
        """Get number of stations."""
        return len(self.stations)
    
    def get_station_by_name(self, name: str) -> Optional[Station]:
        """Get station by name (case-insensitive)."""
        name_lower = name.lower()
        for station in self.stations:
            if station.name.lower() == name_lower:
                return station
        return None
    
    def get_station_by_code(self, code: str) -> Optional[Station]:
        """Get station by code."""
        code_upper = code.upper()
        for station in self.stations:
            if station.code.upper() == code_upper:
                return station
        return None
    
    def get_distance_between(self, station1: str, station2: str) -> float:
        """Get distance between two stations (by name or code).
        
        Args:
            station1: Name or code of first station
            station2: Name or code of second station
            
        Returns:
            Distance in km (absolute value)
        """
        s1 = self.get_station_by_name(station1) or self.get_station_by_code(station1)
        s2 = self.get_station_by_name(station2) or self.get_station_by_code(station2)
        
        if not s1 or not s2:
            raise ValueError(f"Station not found: {station1 if not s1 else station2}")
        
        return abs(s2.cumulative_distance_km - s1.cumulative_distance_km)
    
    def get_intermediate_stations(self, origin: str, destination: str) -> List[Station]:
        """Get all intermediate stations between origin and destination.
        
        Args:
            origin: Origin station name or code
            destination: Destination station name or code
            
        Returns:
            List of stations between origin and destination (inclusive)
        """
        s1 = self.get_station_by_name(origin) or self.get_station_by_code(origin)
        s2 = self.get_station_by_name(destination) or self.get_station_by_code(destination)
        
        if not s1 or not s2:
            raise ValueError(f"Station not found: {origin if not s1 else destination}")
        
        # Get indices
        idx1 = s1.sr_no - 1  # sr_no is 1-based
        idx2 = s2.sr_no - 1
        
        # Ensure correct order
        start_idx, end_idx = min(idx1, idx2), max(idx1, idx2)
        
        return self.stations[start_idx:end_idx + 1]
    
    def calculate_journey_time(
        self, 
        origin: str, 
        destination: str, 
        avg_speed_kmh: Optional[float] = None
    ) -> float:
        """Calculate journey time between two stations.
        
        Args:
            origin: Origin station name or code
            destination: Destination station name or code
            avg_speed_kmh: Average speed (uses config default if None)
            
        Returns:
            Journey time in minutes (including dwell times)
        """
        data = self.load()
        
        if avg_speed_kmh is None:
            avg_speed_kmh = data['line_info'].get('average_speed_kmh', 35)
        
        dwell_time_sec = data['operational_params'].get('dwell_time_seconds', 30)
        
        distance = self.get_distance_between(origin, destination)
        intermediate = self.get_intermediate_stations(origin, destination)
        
        # Travel time
        travel_time_hours = distance / avg_speed_kmh
        travel_time_minutes = travel_time_hours * 60
        
        # Add dwell time at each intermediate station (except destination)
        num_stops = len(intermediate) - 1  # Exclude destination
        total_dwell_minutes = (num_stops * dwell_time_sec) / 60
        
        return travel_time_minutes + total_dwell_minutes
    
    def calculate_round_trip_time(self, avg_speed_kmh: Optional[float] = None) -> float:
        """Calculate round trip time between terminals.
        
        Returns:
            Round trip time in minutes (including turnaround)
        """
        data = self.load()
        terminals = self.terminals
        
        if len(terminals) < 2:
            raise ValueError("Need at least 2 terminal stations for round trip")
        
        turnaround_sec = data['operational_params'].get('terminal_turnaround_seconds', 180)
        
        # One-way journey time
        one_way_time = self.calculate_journey_time(terminals[0], terminals[-1], avg_speed_kmh)
        
        # Round trip = 2 * one_way + 2 * turnaround
        return (2 * one_way_time) + (2 * turnaround_sec / 60)
    
    def get_station_sequence_for_trip(
        self, 
        origin: str, 
        destination: str, 
        include_times: bool = True,
        departure_time: str = "07:00"
    ) -> List[Dict]:
        """Get detailed station sequence for a trip.
        
        Args:
            origin: Origin station name
            destination: Destination station name
            include_times: Whether to calculate arrival times
            departure_time: Departure time from origin (HH:MM format)
            
        Returns:
            List of dicts with station info and arrival times
        """
        data = self.load()
        avg_speed = data['line_info'].get('average_speed_kmh', 35)
        dwell_time_sec = data['operational_params'].get('dwell_time_seconds', 30)
        
        stations = self.get_intermediate_stations(origin, destination)
        
        # Parse departure time
        from datetime import datetime, timedelta
        current_time = datetime.strptime(departure_time, "%H:%M")
        
        sequence = []
        prev_cumulative = stations[0].cumulative_distance_km
        
        for i, station in enumerate(stations):
            entry = {
                'sr_no': station.sr_no,
                'code': station.code,
                'name': station.name,
                'distance_from_origin_km': round(station.cumulative_distance_km - stations[0].cumulative_distance_km, 3),
                'is_terminal': station.is_terminal
            }
            
            if include_times:
                if i == 0:
                    # Origin station - departure time
                    entry['arrival_time'] = None
                    entry['departure_time'] = current_time.strftime("%H:%M")
                else:
                    # Calculate travel time from previous station
                    segment_distance = station.cumulative_distance_km - prev_cumulative
                    travel_time_min = (segment_distance / avg_speed) * 60
                    current_time += timedelta(minutes=travel_time_min)
                    
                    entry['arrival_time'] = current_time.strftime("%H:%M")
                    
                    if i < len(stations) - 1:
                        # Not destination - add dwell time
                        current_time += timedelta(seconds=dwell_time_sec)
                        entry['departure_time'] = current_time.strftime("%H:%M")
                    else:
                        # Destination - no departure
                        entry['departure_time'] = None
                
                prev_cumulative = station.cumulative_distance_km
            
            sequence.append(entry)
        
        return sequence
    
    def to_dict(self) -> Dict:
        """Export route info as dictionary."""
        return {
            'line_name': self.route_info.name,
            'operator': self.route_info.operator,
            'total_distance_km': self.total_distance_km,
            'station_count': self.station_count,
            'terminals': self.terminals,
            'stations': [
                {
                    'sr_no': s.sr_no,
                    'code': s.code,
                    'name': s.name,
                    'distance_from_prev_km': s.distance_from_prev_km,
                    'cumulative_distance_km': s.cumulative_distance_km,
                    'is_terminal': s.is_terminal,
                    'has_depot': s.has_depot
                }
                for s in self.stations
            ],
            'operational_params': self.route_info.operational_params
        }


# Global cached instance for performance
_default_loader: Optional[StationDataLoader] = None


def get_station_loader(config_path: Optional[str] = None) -> StationDataLoader:
    """Get station data loader (cached for default path).
    
    Args:
        config_path: Custom config path, or None for default
        
    Returns:
        StationDataLoader instance
    """
    global _default_loader
    
    if config_path is None:
        if _default_loader is None:
            _default_loader = StationDataLoader()
        return _default_loader
    
    return StationDataLoader(config_path)


def get_route_distance() -> float:
    """Get total route distance (convenience function)."""
    return get_station_loader().total_distance_km


def get_terminals() -> List[str]:
    """Get terminal station names (convenience function)."""
    return get_station_loader().terminals
