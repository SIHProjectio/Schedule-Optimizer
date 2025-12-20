"""
Service Block Generator
Generates realistic service blocks with departure times for train schedules.
Uses station data from JSON configuration for flexible route support.
"""
from typing import List, Dict, Tuple, Optional
from datetime import time, datetime, timedelta
import logging

# Import station loader for route configuration
try:
    from greedyOptim.routing.station_loader import get_station_loader, StationDataLoader
    STATION_LOADER_AVAILABLE = True
except ImportError:
    STATION_LOADER_AVAILABLE = False

logger = logging.getLogger(__name__)


class ServiceBlockGenerator:
    """Generates service blocks for trains based on operational requirements.
    
    Loads route parameters from station configuration JSON, allowing
    easy customization for different metro lines.
    """
    
    # Fallback defaults (used if station config not available)
    DEFAULT_ROUTE_LENGTH_KM = 24.843  # Aluva to Pettah
    DEFAULT_AVG_SPEED_KMH = 35.0
    DEFAULT_TERMINALS = ['Aluva', 'Pettah']
    DEFAULT_OPERATIONAL_START = time(5, 0)
    DEFAULT_OPERATIONAL_END = time(23, 0)
    DEFAULT_PEAK_HEADWAY_MINUTES = 6.0
    DEFAULT_OFFPEAK_HEADWAY_MINUTES = 15.0
    DEFAULT_DWELL_TIME_SECONDS = 30
    DEFAULT_TURNAROUND_SECONDS = 180
    
    def __init__(self, station_config_path: Optional[str] = None):
        """Initialize service block generator.
        
        Args:
            station_config_path: Optional path to station JSON config.
                                If None, uses default configuration.
        """
        self._station_loader: Optional[StationDataLoader] = None
        self._all_blocks_cache = None
        self._stations_cache = None
        
        self._load_station_config(station_config_path)
        
        self.round_trip_time_hours = (self.route_length_km * 2) / self.avg_speed_kmh
        self.round_trip_time_minutes = self.round_trip_time_hours * 60
    
    def _load_station_config(self, config_path: Optional[str] = None):
        """Load station configuration from JSON."""
        if STATION_LOADER_AVAILABLE:
            try:
                self._station_loader = get_station_loader(config_path)
                route_info = self._station_loader.route_info
                op_params = route_info.operational_params
                
                self.route_length_km = self._station_loader.total_distance_km
                self.terminals = self._station_loader.terminals
                self.avg_speed_kmh = self._station_loader.load()['line_info'].get(
                    'average_speed_kmh', self.DEFAULT_AVG_SPEED_KMH
                )
                
                self.dwell_time_seconds = op_params.get(
                    'dwell_time_seconds', self.DEFAULT_DWELL_TIME_SECONDS
                )
                self.turnaround_seconds = op_params.get(
                    'terminal_turnaround_seconds', self.DEFAULT_TURNAROUND_SECONDS
                )
                self.peak_headway_minutes = op_params.get(
                    'min_headway_peak_minutes', self.DEFAULT_PEAK_HEADWAY_MINUTES
                )
                self.offpeak_headway_minutes = op_params.get(
                    'min_headway_offpeak_minutes', self.DEFAULT_OFFPEAK_HEADWAY_MINUTES
                )
                
                op_start = op_params.get('operational_start', '05:00')
                op_end = op_params.get('operational_end', '23:00')
                self.operational_start = datetime.strptime(op_start, '%H:%M').time()
                self.operational_end = datetime.strptime(op_end, '%H:%M').time()
                
                self.peak_hours = []
                for peak in op_params.get('peak_hours', []):
                    start = int(peak['start'].split(':')[0])
                    end = int(peak['end'].split(':')[0])
                    self.peak_hours.append((start, end))
                
                if not self.peak_hours:
                    self.peak_hours = [(7, 10), (17, 21)]  # Default peaks
                
                logger.info(f"Loaded station config: {len(self._station_loader.stations)} stations, "
                           f"{self.route_length_km:.3f} km route")
                return
                
            except Exception as e:
                logger.warning(f"Failed to load station config: {e}. Using defaults.")
        
        self._use_defaults()
    
    def _use_defaults(self):
        """Set default values when config loading fails."""
        self.route_length_km = self.DEFAULT_ROUTE_LENGTH_KM
        self.terminals = self.DEFAULT_TERMINALS
        self.avg_speed_kmh = self.DEFAULT_AVG_SPEED_KMH
        self.dwell_time_seconds = self.DEFAULT_DWELL_TIME_SECONDS
        self.turnaround_seconds = self.DEFAULT_TURNAROUND_SECONDS
        self.peak_headway_minutes = self.DEFAULT_PEAK_HEADWAY_MINUTES
        self.offpeak_headway_minutes = self.DEFAULT_OFFPEAK_HEADWAY_MINUTES
        self.operational_start = self.DEFAULT_OPERATIONAL_START
        self.operational_end = self.DEFAULT_OPERATIONAL_END
        self.peak_hours = [(7, 10), (17, 21)]
    
    @property
    def stations(self) -> List[Dict]:
        """Get list of all stations with their details."""
        if self._stations_cache is not None:
            return self._stations_cache
        
        if self._station_loader:
            self._stations_cache = [
                {
                    'sr_no': s.sr_no,
                    'code': s.code,
                    'name': s.name,
                    'distance_from_prev_km': s.distance_from_prev_km,
                    'cumulative_distance_km': s.cumulative_distance_km,
                    'is_terminal': s.is_terminal,
                    'has_depot': s.has_depot
                }
                for s in self._station_loader.stations
            ]
        else:
            self._stations_cache = [
                {'sr_no': 1, 'code': 'ALV', 'name': 'Aluva', 'cumulative_distance_km': 0, 'is_terminal': True},
                {'sr_no': 22, 'code': 'PTH', 'name': 'Pettah', 'cumulative_distance_km': self.route_length_km, 'is_terminal': True}
            ]
        
        return self._stations_cache
    
    def get_station_sequence(self, origin: str, destination: str, departure_time: str = "07:00") -> List[Dict]:
        """Get detailed station sequence for a trip with arrival times.
        
        Args:
            origin: Origin station name
            destination: Destination station name  
            departure_time: Departure time (HH:MM format)
            
        Returns:
            List of dicts with station info and calculated times
        """
        if self._station_loader:
            return self._station_loader.get_station_sequence_for_trip(
                origin, destination, 
                include_times=True, 
                departure_time=departure_time
            )
        
        return [
            {'name': origin, 'departure_time': departure_time, 'arrival_time': None},
            {'name': destination, 'arrival_time': self._estimate_arrival(departure_time), 'departure_time': None}
        ]
    
    def _estimate_arrival(self, departure_time: str) -> str:
        """Estimate arrival time at destination (fallback)."""
        dep = datetime.strptime(departure_time, '%H:%M')
        travel_minutes = (self.route_length_km / self.avg_speed_kmh) * 60
        arr = dep + timedelta(minutes=travel_minutes)
        return arr.strftime('%H:%M')
    
    def get_all_service_blocks(self) -> List[Dict]:
        """Get all available service blocks for the day.
        
        Pre-generates all possible service blocks that need to be assigned to trainsets.
        These represent the "slots" that the optimizer will fill.
        Includes intermediate station information for each block.
        
        Returns:
            List of all service block dictionaries with block_id, departure_time, etc.
        """
        if self._all_blocks_cache is not None:
            return self._all_blocks_cache
        
        all_blocks = []
        block_counter = 0
        
        current_hour = self.operational_start.hour
        end_hour = self.operational_end.hour
        
        while current_hour < end_hour:
            is_peak = any(start <= current_hour < end for start, end in self.peak_hours)
            headway = self.peak_headway_minutes if is_peak else self.offpeak_headway_minutes
            
            if current_hour < 10:
                period = 'morning_peak' if is_peak else 'early_morning'
            elif current_hour < 17:
                period = 'midday'
            elif current_hour < 21:
                period = 'evening_peak' if is_peak else 'evening'
            else:
                period = 'late_evening'
            
            for minute in range(0, 60, int(headway)):
                block_counter += 1
                origin = self.terminals[block_counter % 2]
                destination = self.terminals[(block_counter + 1) % 2]
                
                if is_peak:
                    trip_count = 3
                elif current_hour >= 21:
                    trip_count = 1
                else:
                    trip_count = 2
                
                departure_time = f'{current_hour:02d}:{minute:02d}'
                
                block = {
                    'block_id': f'BLK-{block_counter:03d}',
                    'departure_time': departure_time,
                    'origin': origin,
                    'destination': destination,
                    'trip_count': trip_count,
                    'estimated_km': int(trip_count * self.route_length_km * 2),
                    'period': period,
                    'is_peak': is_peak
                }
                
                if self._station_loader:
                    block['station_count'] = len(self._station_loader.stations)
                    block['intermediate_stops'] = len(self._station_loader.stations) - 2
                    block['journey_time_minutes'] = round(
                        self._station_loader.calculate_journey_time(origin, destination), 1
                    )
                
                all_blocks.append(block)
            
            current_hour += 1
        
        self._all_blocks_cache = all_blocks
        return all_blocks
    
    def get_block_count(self) -> int:
        """Get total number of service blocks."""
        return len(self.get_all_service_blocks())
    
    def get_peak_block_indices(self) -> List[int]:
        """Get indices of peak hour blocks."""
        blocks = self.get_all_service_blocks()
        return [i for i, b in enumerate(blocks) if b['is_peak']]
    
    def get_blocks_by_ids(self, block_ids: List[str]) -> List[Dict]:
        """Get blocks by their IDs."""
        all_blocks = self.get_all_service_blocks()
        block_map = {b['block_id']: b for b in all_blocks}
        return [block_map[bid] for bid in block_ids if bid in block_map]
    
    def generate_service_blocks(self, train_index: int, num_service_trains: int) -> List[Dict]:
        """Generate service blocks for a train with staggered departures.
        
        Args:
            train_index: Index of this train in the service fleet (0-based)
            num_service_trains: Total number of trains in service
            
        Returns:
            List of service block dictionaries with station details
        """
        blocks = []
        
        peak_interval = max(5, int(self.peak_headway_minutes))
        
        offset_minutes = (train_index * peak_interval) % 60
        
        morning_start_hour = 7 + (train_index * peak_interval) // 60
        if morning_start_hour < 10:
            origin = self.terminals[0] if train_index % 2 == 0 else self.terminals[1]
            destination = self.terminals[1] if train_index % 2 == 0 else self.terminals[0]
            departure_time = f'{morning_start_hour:02d}:{offset_minutes:02d}'
            
            block = {
                'block_id': f'BLK-M-{train_index+1:03d}',
                'departure_time': departure_time,
                'origin': origin,
                'destination': destination,
                'trip_count': self._calculate_trips(3.0),  # 3 hours
                'estimated_km': self._calculate_km(3.0)
            }
            
            if self._station_loader:
                block['stations'] = self.get_station_sequence(origin, destination, departure_time)
                block['journey_time_minutes'] = round(
                    self._station_loader.calculate_journey_time(origin, destination), 1
                )
            
            blocks.append(block)
        
        midday_start_hour = 11 + (train_index * 15) // 60
        midday_minute = (train_index * 15) % 60
        if midday_start_hour < 16:
            origin = self.terminals[1] if train_index % 2 == 0 else self.terminals[0]
            destination = self.terminals[0] if train_index % 2 == 0 else self.terminals[1]
            departure_time = f'{midday_start_hour:02d}:{midday_minute:02d}'
            
            block = {
                'block_id': f'BLK-D-{train_index+1:03d}',
                'departure_time': departure_time,
                'origin': origin,
                'destination': destination,
                'trip_count': self._calculate_trips(5.0, peak=False),
                'estimated_km': self._calculate_km(5.0, peak=False)
            }
            
            if self._station_loader:
                block['stations'] = self.get_station_sequence(origin, destination, departure_time)
                block['journey_time_minutes'] = round(
                    self._station_loader.calculate_journey_time(origin, destination), 1
                )
            
            blocks.append(block)
        
        evening_start_hour = 17 + (train_index * peak_interval) // 60
        evening_minute = (train_index * peak_interval) % 60
        if evening_start_hour < 20:
            origin = self.terminals[0] if train_index % 2 == 0 else self.terminals[1]
            destination = self.terminals[1] if train_index % 2 == 0 else self.terminals[0]
            departure_time = f'{evening_start_hour:02d}:{evening_minute:02d}'
            
            block = {
                'block_id': f'BLK-E-{train_index+1:03d}',
                'departure_time': departure_time,
                'origin': origin,
                'destination': destination,
                'trip_count': self._calculate_trips(3.0),
                'estimated_km': self._calculate_km(3.0)
            }
            
            if self._station_loader:
                block['stations'] = self.get_station_sequence(origin, destination, departure_time)
                block['journey_time_minutes'] = round(
                    self._station_loader.calculate_journey_time(origin, destination), 1
                )
            
            blocks.append(block)
        
        if train_index % 2 == 0:
            origin = self.terminals[1]
            destination = self.terminals[0]
            departure_time = f'20:{(train_index * 20) % 60:02d}'
            
            block = {
                'block_id': f'BLK-L-{train_index+1:03d}',
                'departure_time': departure_time,
                'origin': origin,
                'destination': destination,
                'trip_count': self._calculate_trips(2.0, peak=False),
                'estimated_km': self._calculate_km(2.0, peak=False)
            }
            
            if self._station_loader:
                block['stations'] = self.get_station_sequence(origin, destination, departure_time)
                block['journey_time_minutes'] = round(
                    self._station_loader.calculate_journey_time(origin, destination), 1
                )
            
            blocks.append(block)
        
        return blocks
    
    def _calculate_trips(self, duration_hours: float, peak: bool = True) -> int:
        """Calculate number of round trips in a time block."""
        headway = self.peak_headway_minutes if peak else self.offpeak_headway_minutes
        trips_per_hour = 60 / headway
        trips_per_hour = trips_per_hour / 2
        total_trips = int(duration_hours * trips_per_hour)
        return max(1, total_trips)
    
    def _calculate_km(self, duration_hours: float, peak: bool = True) -> int:
        """Calculate estimated kilometers for a time block."""
        trips = self._calculate_trips(duration_hours, peak)
        km = trips * self.route_length_km * 2
        return int(km)
    
    def generate_all_service_blocks(self, num_service_trains: int) -> Dict[int, List[Dict]]:
        """Generate service blocks for all trains.
        
        Args:
            num_service_trains: Number of trains in service
            
        Returns:
            Dictionary mapping train index to list of service blocks
        """
        all_blocks = {}
        
        for i in range(num_service_trains):
            all_blocks[i] = self.generate_service_blocks(i, num_service_trains)
        
        return all_blocks
    
    def calculate_daily_km(self, service_blocks: List[Dict]) -> int:
        """Calculate total daily kilometers from service blocks."""
        total_km = sum(block['estimated_km'] for block in service_blocks)
        return total_km


# Convenience function
def create_service_blocks_for_schedule(selected_trainsets: List[str]) -> Dict[str, List[Dict]]:
    """Create service blocks for a list of selected trainsets.
    
    Args:
        selected_trainsets: List of trainset IDs in service
        
    Returns:
        Dictionary mapping trainset_id to service blocks
    """
    generator = ServiceBlockGenerator()
    num_trains = len(selected_trainsets)
    
    blocks_by_train = {}
    for i, trainset_id in enumerate(selected_trainsets):
        blocks = generator.generate_service_blocks(i, num_trains)
        blocks_by_train[trainset_id] = blocks
    
    return blocks_by_train
