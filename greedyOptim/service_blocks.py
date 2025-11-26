"""
Service Block Generator
Generates realistic service blocks with departure times for train schedules.
"""
from typing import List, Dict
from datetime import time


class ServiceBlockGenerator:
    """Generates service blocks for trains based on operational requirements."""
    
    # Kochi Metro operational parameters
    OPERATIONAL_START = time(5, 0)  # 5:00 AM
    OPERATIONAL_END = time(23, 0)  # 11:00 PM
    
    # Service patterns
    PEAK_HOURS = [(7, 9), (18, 21)]  # Morning and evening peaks (7-9 AM, 6-9 PM)
    PEAK_HEADWAY_MINUTES = 6.0  # 6 minutes between trains during peak (target 5-7)
    OFFPEAK_HEADWAY_MINUTES = 15.0  # 15 minutes during off-peak
    
    # Route parameters
    ROUTE_LENGTH_KM = 25.612
    AVG_SPEED_KMH = 35.0
    TERMINALS = ['Aluva', 'Pettah']
    
    def __init__(self):
        """Initialize service block generator."""
        self.round_trip_time_hours = (self.ROUTE_LENGTH_KM * 2) / self.AVG_SPEED_KMH
        self.round_trip_time_minutes = self.round_trip_time_hours * 60
    
    def generate_service_blocks(self, train_index: int, num_service_trains: int) -> List[Dict]:
        """Generate service blocks for a train with staggered departures.
        
        Args:
            train_index: Index of this train in the service fleet (0-based)
            num_service_trains: Total number of trains in service
            
        Returns:
            List of service block dictionaries
        """
        blocks = []
        
        # Calculate departure interval based on number of trains
        # Distribute trains evenly throughout peak hours
        peak_interval = max(5, int(self.PEAK_HEADWAY_MINUTES))
        
        # Stagger departures so trains are evenly spaced
        offset_minutes = (train_index * peak_interval) % 60
        
        # Morning peak block (7-10 AM, 3 hours)
        morning_start_hour = 7 + (train_index * peak_interval) // 60
        if morning_start_hour < 10:  # Only if within morning peak
            blocks.append({
                'block_id': f'BLK-M-{train_index+1:03d}',
                'departure_time': f'{morning_start_hour:02d}:{offset_minutes:02d}',
                'origin': self.TERMINALS[0] if train_index % 2 == 0 else self.TERMINALS[1],
                'destination': self.TERMINALS[1] if train_index % 2 == 0 else self.TERMINALS[0],
                'trip_count': self._calculate_trips(3.0),  # 3 hours
                'estimated_km': self._calculate_km(3.0)
            })
        
        # Midday block (11-16, 5 hours)
        midday_start_hour = 11 + (train_index * 15) // 60  # 15 min intervals
        midday_minute = (train_index * 15) % 60
        if midday_start_hour < 16:
            blocks.append({
                'block_id': f'BLK-D-{train_index+1:03d}',
                'departure_time': f'{midday_start_hour:02d}:{midday_minute:02d}',
                'origin': self.TERMINALS[1] if train_index % 2 == 0 else self.TERMINALS[0],
                'destination': self.TERMINALS[0] if train_index % 2 == 0 else self.TERMINALS[1],
                'trip_count': self._calculate_trips(5.0, peak=False),
                'estimated_km': self._calculate_km(5.0, peak=False)
            })
        
        # Evening peak block (17-20, 3 hours)
        evening_start_hour = 17 + (train_index * peak_interval) // 60
        evening_minute = (train_index * peak_interval) % 60
        if evening_start_hour < 20:
            blocks.append({
                'block_id': f'BLK-E-{train_index+1:03d}',
                'departure_time': f'{evening_start_hour:02d}:{evening_minute:02d}',
                'origin': self.TERMINALS[0] if train_index % 2 == 0 else self.TERMINALS[1],
                'destination': self.TERMINALS[1] if train_index % 2 == 0 else self.TERMINALS[0],
                'trip_count': self._calculate_trips(3.0),
                'estimated_km': self._calculate_km(3.0)
            })
        
        # Late evening block (20-22, 2 hours) - lower frequency
        if train_index % 2 == 0:  # Only half the fleet for late evening
            blocks.append({
                'block_id': f'BLK-L-{train_index+1:03d}',
                'departure_time': f'20:{(train_index * 20) % 60:02d}',
                'origin': self.TERMINALS[1],
                'destination': self.TERMINALS[0],
                'trip_count': self._calculate_trips(2.0, peak=False),
                'estimated_km': self._calculate_km(2.0, peak=False)
            })
        
        return blocks
    
    def _calculate_trips(self, duration_hours: float, peak: bool = True) -> int:
        """Calculate number of round trips in a time block."""
        trips_per_hour = 60 / (self.PEAK_HEADWAY_MINUTES if peak else self.OFFPEAK_HEADWAY_MINUTES)
        trips_per_hour = trips_per_hour / 2  # One-way trips, so divide by 2 for round trips
        total_trips = int(duration_hours * trips_per_hour)
        return max(1, total_trips)
    
    def _calculate_km(self, duration_hours: float, peak: bool = True) -> int:
        """Calculate estimated kilometers for a time block."""
        trips = self._calculate_trips(duration_hours, peak)
        km = trips * self.ROUTE_LENGTH_KM * 2  # Round trips
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
