"""
Core utility functions for DataService
"""
from datetime import datetime, time, timedelta
from typing import List, Tuple


def time_to_minutes(t: time) -> int:
    """Convert time to minutes since midnight"""
    return t.hour * 60 + t.minute


def minutes_to_time(minutes: int) -> time:
    """Convert minutes since midnight to time"""
    hours = minutes // 60
    mins = minutes % 60
    return time(hours % 24, mins)


def time_range_overlap(range1: Tuple[time, time], range2: Tuple[time, time]) -> bool:
    """Check if two time ranges overlap"""
    start1, end1 = time_to_minutes(range1[0]), time_to_minutes(range1[1])
    start2, end2 = time_to_minutes(range2[0]), time_to_minutes(range2[1])
    return max(start1, start2) < min(end1, end2)


def is_peak_hour(current_time: time, peak_hours: List[Tuple[time, time]]) -> bool:
    """Check if current time is within peak hours"""
    for start, end in peak_hours:
        if time_to_minutes(start) <= time_to_minutes(current_time) <= time_to_minutes(end):
            return True
    return False


def format_iso_datetime(dt: datetime) -> str:
    """Format datetime as ISO string"""
    return dt.isoformat()


def parse_iso_date(date_str: str) -> datetime:
    """Parse ISO date string to datetime"""
    return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
