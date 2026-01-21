"""
Unit tests for core utilities
"""
from datetime import time
from ..utils import (
    time_to_minutes, minutes_to_time,
    time_range_overlap, is_peak_hour
)


def test_time_to_minutes():
    """Test time to minutes conversion"""
    assert time_to_minutes(time(0, 0)) == 0
    assert time_to_minutes(time(1, 0)) == 60
    assert time_to_minutes(time(5, 30)) == 330
    assert time_to_minutes(time(12, 0)) == 720


def test_minutes_to_time():
    """Test minutes to time conversion"""
    assert minutes_to_time(0) == time(0, 0)
    assert minutes_to_time(60) == time(1, 0)
    assert minutes_to_time(330) == time(5, 30)
    assert minutes_to_time(720) == time(12, 0)


def test_time_conversion_round_trip():
    """Test round-trip conversion"""
    t = time(14, 45)
    assert minutes_to_time(time_to_minutes(t)) == t


def test_time_range_overlap():
    """Test time range overlap detection"""
    # Overlapping ranges
    assert time_range_overlap(
        (time(8, 0), time(10, 0)),
        (time(9, 0), time(11, 0))
    ) is True
    
    # Non-overlapping ranges
    assert time_range_overlap(
        (time(8, 0), time(10, 0)),
        (time(11, 0), time(13, 0))
    ) is False
    
    # Touching ranges
    assert time_range_overlap(
        (time(8, 0), time(10, 0)),
        (time(10, 0), time(12, 0))
    ) is False


def test_is_peak_hour():
    """Test peak hour detection"""
    peak_hours = [
        (time(7, 0), time(10, 0)),
        (time(17, 0), time(20, 0))
    ]
    
    # During peak
    assert is_peak_hour(time(8, 0), peak_hours) is True
    assert is_peak_hour(time(18, 30), peak_hours) is True
    
    # Outside peak
    assert is_peak_hour(time(12, 0), peak_hours) is False
    assert is_peak_hour(time(22, 0), peak_hours) is False
