"""
Service Block Generator
Generates realistic service blocks with departure times for train schedules.
"""
from typing import List, Dict, Tuple
from datetime import time, datetime, timedelta


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
        self._all_blocks_cache = None
    
    def get_all_service_blocks(self) -> List[Dict]:
        """Get all available service blocks for the day.
        
        Pre-generates all possible service blocks that need to be assigned to trainsets.
        These represent the "slots" that the optimizer will fill.
        
        Returns:
            List of all service block dictionaries with block_id, departure_time, etc.
        """
        if self._all_blocks_cache is not None:
            return self._all_blocks_cache
        
        all_blocks = []
        block_counter = 0
        
        # Morning peak blocks (7:00 - 10:00)
        # Need departures every 6 minutes = 10 per hour = 30 blocks
        for hour in [7, 8, 9]:
            for minute in range(0, 60, 6):
                block_counter += 1
                origin = self.TERMINALS[block_counter % 2]
                destination = self.TERMINALS[(block_counter + 1) % 2]
                all_blocks.append({
                    'block_id': f'BLK-{block_counter:03d}',
                    'departure_time': f'{hour:02d}:{minute:02d}',
                    'origin': origin,
                    'destination': destination,
                    'trip_count': 3,  # ~3 hours of peak service
                    'estimated_km': int(3 * self.ROUTE_LENGTH_KM * 2),
                    'period': 'morning_peak',
                    'is_peak': True
                })
        
        # Midday off-peak blocks (10:00 - 17:00)
        # Need departures every 15 minutes = 4 per hour = 28 blocks
        for hour in range(10, 17):
            for minute in range(0, 60, 15):
                block_counter += 1
                origin = self.TERMINALS[block_counter % 2]
                destination = self.TERMINALS[(block_counter + 1) % 2]
                all_blocks.append({
                    'block_id': f'BLK-{block_counter:03d}',
                    'departure_time': f'{hour:02d}:{minute:02d}',
                    'origin': origin,
                    'destination': destination,
                    'trip_count': 2,
                    'estimated_km': int(2 * self.ROUTE_LENGTH_KM * 2),
                    'period': 'midday',
                    'is_peak': False
                })
        
        # Evening peak blocks (17:00 - 21:00)
        # Need departures every 6 minutes = 10 per hour = 40 blocks
        for hour in range(17, 21):
            for minute in range(0, 60, 6):
                block_counter += 1
                origin = self.TERMINALS[block_counter % 2]
                destination = self.TERMINALS[(block_counter + 1) % 2]
                all_blocks.append({
                    'block_id': f'BLK-{block_counter:03d}',
                    'departure_time': f'{hour:02d}:{minute:02d}',
                    'origin': origin,
                    'destination': destination,
                    'trip_count': 3,
                    'estimated_km': int(3 * self.ROUTE_LENGTH_KM * 2),
                    'period': 'evening_peak',
                    'is_peak': True
                })
        
        # Late evening blocks (21:00 - 23:00)
        # Need departures every 15 minutes = 4 per hour = 8 blocks
        for hour in range(21, 23):
            for minute in range(0, 60, 15):
                block_counter += 1
                origin = self.TERMINALS[block_counter % 2]
                destination = self.TERMINALS[(block_counter + 1) % 2]
                all_blocks.append({
                    'block_id': f'BLK-{block_counter:03d}',
                    'departure_time': f'{hour:02d}:{minute:02d}',
                    'origin': origin,
                    'destination': destination,
                    'trip_count': 1,
                    'estimated_km': int(1 * self.ROUTE_LENGTH_KM * 2),
                    'period': 'late_evening',
                    'is_peak': False
                })
        
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
