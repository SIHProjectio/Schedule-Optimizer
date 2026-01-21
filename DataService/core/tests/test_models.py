"""
Unit tests for core models
"""
import pytest
from datetime import time
from ..models import (
    TrainStatus, CertificateStatus, Station, Route,
    FitnessCertificate, FitnessCertificates
)


def test_train_status_enum():
    """Test TrainStatus enum values"""
    assert TrainStatus.REVENUE_SERVICE == "REVENUE_SERVICE"
    assert TrainStatus.STANDBY == "STANDBY"
    assert TrainStatus.MAINTENANCE == "MAINTENANCE"


def test_station_model():
    """Test Station model creation"""
    station = Station(
        station_id="ST-001",
        name="Aluva",
        sequence=1,
        distance_from_origin_km=0.0,
        avg_dwell_time_seconds=30
    )
    assert station.name == "Aluva"
    assert station.sequence == 1


def test_route_model():
    """Test Route model creation"""
    stations = [
        Station(
            station_id=f"ST-{i:03d}",
            name=f"Station {i}",
            sequence=i,
            distance_from_origin_km=float(i * 2.5)
        )
        for i in range(1, 6)
    ]
    
    route = Route(
        route_id="ROUTE-001",
        name="Test Line",
        stations=stations,
        total_distance_km=10.0,
        avg_speed_kmh=35
    )
    
    assert len(route.stations) == 5
    assert route.total_distance_km == 10.0


def test_fitness_certificates():
    """Test FitnessCertificates model"""
    certs = FitnessCertificates(
        rolling_stock=FitnessCertificate(
            valid_until="2025-12-31",
            status=CertificateStatus.VALID
        ),
        signalling=FitnessCertificate(
            valid_until="2025-12-31",
            status=CertificateStatus.VALID
        ),
        telecom=FitnessCertificate(
            valid_until="2025-12-31",
            status=CertificateStatus.VALID
        )
    )
    
    assert certs.rolling_stock.status == CertificateStatus.VALID
    assert certs.signalling.status == CertificateStatus.VALID
