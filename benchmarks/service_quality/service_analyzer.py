"""
Service Quality Analysis Engine
Analyzes headway consistency, wait times, and service coverage from schedules.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta, time
from collections import defaultdict


@dataclass
class ServiceQualityMetrics:
    """Service quality metrics for a schedule."""
    
    # Headway Consistency
    peak_headway_mean: float  # Average minutes between trains (peak hours)
    peak_headway_std: float  # Standard deviation (peak)
    offpeak_headway_mean: float  # Average minutes between trains (off-peak)
    offpeak_headway_std: float  # Standard deviation (off-peak)
    peak_headway_coefficient_variation: float  # CV = std/mean for peak
    offpeak_headway_coefficient_variation: float  # CV = std/mean for off-peak
    
    # Passenger Wait Time
    avg_wait_time_peak: float  # Average wait time during peak (minutes)
    max_wait_time_peak: float  # Maximum wait time during peak (minutes)
    avg_wait_time_offpeak: float  # Average wait time during off-peak (minutes)
    max_wait_time_offpeak: float  # Maximum wait time during off-peak (minutes)
    wait_time_reduction_vs_baseline: float  # % improvement over baseline
    
    # Service Coverage
    operational_hours: float  # Total hours of service
    peak_hours_covered: float  # Hours during peak with adequate frequency
    offpeak_hours_covered: float  # Hours during off-peak with adequate frequency
    service_coverage_percent: float  # % of operational hours with adequate frequency
    peak_coverage_percent: float  # % of peak hours with adequate frequency
    gaps_in_service: int  # Number of service gaps > threshold
    
    # Overall Quality Score
    headway_consistency_score: float  # 0-100 based on CV
    wait_time_score: float  # 0-100 based on wait times
    coverage_score: float  # 0-100 based on coverage
    overall_quality_score: float  # Weighted average


class ServiceQualityAnalyzer:
    """Analyzes service quality metrics from train schedules."""
    
    # Kochi Metro operational parameters
    PEAK_HOURS = [(7, 10), (17, 20)]  # Morning and evening peaks
    OPERATIONAL_START = time(6, 0)  # 6:00 AM
    OPERATIONAL_END = time(22, 0)  # 10:00 PM
    
    # Service standards (minutes)
    TARGET_PEAK_HEADWAY = 7.5  # Target: 7.5 minutes during peak
    TARGET_OFFPEAK_HEADWAY = 15.0  # Target: 15 minutes during off-peak
    MAX_ACCEPTABLE_HEADWAY = 30.0  # Beyond this is considered a gap
    
    # Baseline for comparison (poor service)
    BASELINE_PEAK_HEADWAY = 12.0
    BASELINE_OFFPEAK_HEADWAY = 25.0
    
    def __init__(self, route_length_km: float = 25.612):
        """Initialize analyzer.
        
        Args:
            route_length_km: Length of metro route (default: Kochi Metro length)
        """
        self.route_length_km = route_length_km
        self.avg_speed_kmh = 35.0  # Kochi Metro average operating speed
    
    def analyze_schedule(self, schedule: Dict) -> ServiceQualityMetrics:
        """Analyze service quality from a generated schedule.
        
        Args:
            schedule: Schedule dictionary from optimizer
            
        Returns:
            ServiceQualityMetrics with all quality measurements
        """
        # Extract service trains and their blocks
        service_trains = self._extract_service_trains(schedule)
        
        if not service_trains:
            return self._empty_metrics()
        
        # Generate departure timeline
        departures = self._extract_departures(service_trains)
        
        if len(departures) < 2:
            return self._empty_metrics()
        
        # Calculate headways
        headways = self._calculate_headways(departures)
        peak_headways, offpeak_headways = self._classify_headways(headways)
        
        # Calculate wait times (half of headway on average)
        wait_metrics = self._calculate_wait_times(peak_headways, offpeak_headways)
        
        # Calculate coverage
        coverage_metrics = self._calculate_coverage(departures)
        
        # Calculate scores
        scores = self._calculate_scores(
            peak_headways, offpeak_headways, 
            wait_metrics, coverage_metrics
        )
        
        return ServiceQualityMetrics(
            # Headway consistency
            peak_headway_mean=np.mean(peak_headways) if peak_headways else 0.0,
            peak_headway_std=np.std(peak_headways) if peak_headways else 0.0,
            offpeak_headway_mean=np.mean(offpeak_headways) if offpeak_headways else 0.0,
            offpeak_headway_std=np.std(offpeak_headways) if offpeak_headways else 0.0,
            peak_headway_coefficient_variation=self._coefficient_of_variation(peak_headways),
            offpeak_headway_coefficient_variation=self._coefficient_of_variation(offpeak_headways),
            
            # Wait times
            avg_wait_time_peak=wait_metrics['avg_peak'],
            max_wait_time_peak=wait_metrics['max_peak'],
            avg_wait_time_offpeak=wait_metrics['avg_offpeak'],
            max_wait_time_offpeak=wait_metrics['max_offpeak'],
            wait_time_reduction_vs_baseline=wait_metrics['reduction_percent'],
            
            # Coverage
            operational_hours=coverage_metrics['operational_hours'],
            peak_hours_covered=coverage_metrics['peak_hours_covered'],
            offpeak_hours_covered=coverage_metrics['offpeak_hours_covered'],
            service_coverage_percent=coverage_metrics['coverage_percent'],
            peak_coverage_percent=coverage_metrics['peak_coverage_percent'],
            gaps_in_service=coverage_metrics['gaps'],
            
            # Scores
            headway_consistency_score=scores['headway_score'],
            wait_time_score=scores['wait_score'],
            coverage_score=scores['coverage_score'],
            overall_quality_score=scores['overall_score']
        )
    
    def _extract_service_trains(self, schedule: Dict) -> List[Dict]:
        """Extract trains in revenue service with their service blocks."""
        service_trains = []
        
        trainsets = schedule.get('trainsets', schedule.get('schedule', {}).get('trainsets', []))
        
        for train in trainsets:
            status = train.get('status', '')
            if status == 'REVENUE_SERVICE' or 'service_blocks' in train:
                service_blocks = train.get('service_blocks', [])
                if service_blocks:
                    service_trains.append({
                        'trainset_id': train.get('trainset_id', 'unknown'),
                        'service_blocks': service_blocks
                    })
        
        return service_trains
    
    def _extract_departures(self, service_trains: List[Dict]) -> List[datetime]:
        """Extract all departure times from service blocks."""
        departures = []
        base_date = datetime.now().date()
        
        for train in service_trains:
            for block in train['service_blocks']:
                departure_str = block.get('departure_time', '')
                if departure_str:
                    try:
                        # Parse HH:MM format
                        hour, minute = map(int, departure_str.split(':'))
                        departure = datetime.combine(base_date, time(hour, minute))
                        departures.append(departure)
                    except (ValueError, AttributeError):
                        continue
        
        # Sort chronologically
        departures.sort()
        return departures
    
    def _calculate_headways(self, departures: List[datetime]) -> List[float]:
        """Calculate time intervals between consecutive departures (headways)."""
        headways = []
        
        for i in range(1, len(departures)):
            delta = (departures[i] - departures[i-1]).total_seconds() / 60.0  # minutes
            headways.append(delta)
        
        return headways
    
    def _classify_headways(self, headways: List[float]) -> Tuple[List[float], List[float]]:
        """Classify headways into peak and off-peak periods."""
        peak_headways = []
        offpeak_headways = []
        
        # For now, use time-based classification
        # In real implementation, would track which time each headway corresponds to
        # Simplified: assume first 40% are peak, rest off-peak
        if len(headways) > 0:
            split_point = max(1, int(len(headways) * 0.4))
            peak_headways = headways[:split_point]
            offpeak_headways = headways[split_point:]
        
        return peak_headways, offpeak_headways
    
    def _calculate_wait_times(self, peak_headways: List[float], 
                             offpeak_headways: List[float]) -> Dict[str, float]:
        """Calculate passenger wait time metrics.
        
        Average wait time = headway / 2 (for random arrival)
        """
        if peak_headways:
            avg_wait_peak = np.mean(peak_headways) / 2.0
            max_wait_peak = max(peak_headways)
        else:
            avg_wait_peak = 0.0
            max_wait_peak = 0.0
        
        if offpeak_headways:
            avg_wait_offpeak = np.mean(offpeak_headways) / 2.0
            max_wait_offpeak = max(offpeak_headways)
        else:
            avg_wait_offpeak = 0.0
            max_wait_offpeak = 0.0
        
        # Calculate improvement vs baseline
        baseline_wait_peak = self.BASELINE_PEAK_HEADWAY / 2.0
        baseline_wait_offpeak = self.BASELINE_OFFPEAK_HEADWAY / 2.0
        baseline_avg = (baseline_wait_peak + baseline_wait_offpeak) / 2.0
        
        current_avg = (avg_wait_peak + avg_wait_offpeak) / 2.0
        if baseline_avg > 0:
            reduction_percent = ((baseline_avg - current_avg) / baseline_avg) * 100
        else:
            reduction_percent = 0.0
        
        return {
            'avg_peak': avg_wait_peak,
            'max_peak': max_wait_peak,
            'avg_offpeak': avg_wait_offpeak,
            'max_offpeak': max_wait_offpeak,
            'reduction_percent': reduction_percent
        }
    
    def _calculate_coverage(self, departures: List[datetime]) -> Dict[str, float]:
        """Calculate service coverage metrics."""
        if not departures:
            return {
                'operational_hours': 0.0,
                'peak_hours_covered': 0.0,
                'offpeak_hours_covered': 0.0,
                'coverage_percent': 0.0,
                'peak_coverage_percent': 0.0,
                'gaps': 0
            }
        
        # Total operational hours
        total_minutes = (datetime.combine(datetime.now().date(), self.OPERATIONAL_END) - 
                        datetime.combine(datetime.now().date(), self.OPERATIONAL_START)).total_seconds() / 60.0
        operational_hours = total_minutes / 60.0
        
        # Peak hours (4 hours total: 7-10, 17-20)
        peak_hours_total = 6.0
        offpeak_hours_total = operational_hours - peak_hours_total
        
        # Count hours with adequate frequency
        # Divide day into hour buckets and check frequency
        hour_buckets = defaultdict(int)
        for departure in departures:
            hour_buckets[departure.hour] += 1
        
        peak_hours_covered = 0.0
        offpeak_hours_covered = 0.0
        
        for hour, count in hour_buckets.items():
            trains_per_hour = count
            
            # Check if peak hour
            is_peak = any(start <= hour < end for start, end in self.PEAK_HOURS)
            
            if is_peak:
                # Need at least 4 trains per hour in peak (every 15 min)
                if trains_per_hour >= 4:
                    peak_hours_covered += 1.0
            else:
                # Need at least 2 trains per hour off-peak (every 30 min)
                if trains_per_hour >= 2:
                    offpeak_hours_covered += 1.0
        
        # Calculate coverage percentages
        peak_coverage_percent = (peak_hours_covered / peak_hours_total * 100) if peak_hours_total > 0 else 0.0
        total_covered = peak_hours_covered + offpeak_hours_covered
        coverage_percent = (total_covered / operational_hours * 100) if operational_hours > 0 else 0.0
        
        # Count service gaps (headways > 30 minutes)
        gaps = 0
        for i in range(1, len(departures)):
            headway = (departures[i] - departures[i-1]).total_seconds() / 60.0
            if headway > self.MAX_ACCEPTABLE_HEADWAY:
                gaps += 1
        
        return {
            'operational_hours': operational_hours,
            'peak_hours_covered': peak_hours_covered,
            'offpeak_hours_covered': offpeak_hours_covered,
            'coverage_percent': coverage_percent,
            'peak_coverage_percent': peak_coverage_percent,
            'gaps': gaps
        }
    
    def _calculate_scores(self, peak_headways: List[float], offpeak_headways: List[float],
                         wait_metrics: Dict, coverage_metrics: Dict) -> Dict[str, float]:
        """Calculate quality scores (0-100 scale)."""
        
        # Headway consistency score (lower CV is better)
        peak_cv = self._coefficient_of_variation(peak_headways)
        offpeak_cv = self._coefficient_of_variation(offpeak_headways)
        avg_cv = (peak_cv + offpeak_cv) / 2.0
        
        # Score: 100 - (CV * 200), capped at 0-100
        # CV of 0.1 = 80 points, CV of 0.3 = 40 points, CV of 0.5 = 0 points
        headway_score = max(0, min(100, 100 - (avg_cv * 200)))
        
        # Wait time score (lower wait is better)
        avg_wait = (wait_metrics['avg_peak'] + wait_metrics['avg_offpeak']) / 2.0
        target_wait = (self.TARGET_PEAK_HEADWAY + self.TARGET_OFFPEAK_HEADWAY) / 4.0  # half of avg headway
        
        if avg_wait > 0:
            # Score based on how close to target
            ratio = target_wait / avg_wait
            wait_score = min(100, ratio * 100)
        else:
            wait_score = 0
        
        # Coverage score (direct percentage)
        coverage_score = coverage_metrics['coverage_percent']
        
        # Overall score (weighted average)
        # Headway consistency: 40%, Wait time: 30%, Coverage: 30%
        overall_score = (
            headway_score * 0.4 +
            wait_score * 0.3 +
            coverage_score * 0.3
        )
        
        return {
            'headway_score': headway_score,
            'wait_score': wait_score,
            'coverage_score': coverage_score,
            'overall_score': overall_score
        }
    
    def _coefficient_of_variation(self, values: List[float]) -> float:
        """Calculate coefficient of variation (std/mean)."""
        if not values or len(values) < 2:
            return 0.0
        
        mean = np.mean(values)
        if mean == 0:
            return 0.0
        
        std = np.std(values)
        return std / mean
    
    def _empty_metrics(self) -> ServiceQualityMetrics:
        """Return empty metrics when no data available."""
        return ServiceQualityMetrics(
            peak_headway_mean=0.0,
            peak_headway_std=0.0,
            offpeak_headway_mean=0.0,
            offpeak_headway_std=0.0,
            peak_headway_coefficient_variation=0.0,
            offpeak_headway_coefficient_variation=0.0,
            avg_wait_time_peak=0.0,
            max_wait_time_peak=0.0,
            avg_wait_time_offpeak=0.0,
            max_wait_time_offpeak=0.0,
            wait_time_reduction_vs_baseline=0.0,
            operational_hours=0.0,
            peak_hours_covered=0.0,
            offpeak_hours_covered=0.0,
            service_coverage_percent=0.0,
            peak_coverage_percent=0.0,
            gaps_in_service=0,
            headway_consistency_score=0.0,
            wait_time_score=0.0,
            coverage_score=0.0,
            overall_quality_score=0.0
        )
    
    def compare_schedules(self, schedules: List[Dict]) -> Dict:
        """Compare multiple schedules and return comparative analysis.
        
        Args:
            schedules: List of schedule dictionaries
            
        Returns:
            Dictionary with comparison metrics
        """
        results = []
        
        for i, schedule in enumerate(schedules):
            metrics = self.analyze_schedule(schedule)
            results.append({
                'schedule_id': i + 1,
                'metrics': metrics
            })
        
        # Find best performers
        if results:
            best_headway = max(results, key=lambda x: x['metrics'].headway_consistency_score)
            best_wait = max(results, key=lambda x: x['metrics'].wait_time_score)
            best_coverage = max(results, key=lambda x: x['metrics'].coverage_score)
            best_overall = max(results, key=lambda x: x['metrics'].overall_quality_score)
            
            return {
                'individual_results': results,
                'best_performers': {
                    'headway_consistency': best_headway['schedule_id'],
                    'wait_time': best_wait['schedule_id'],
                    'coverage': best_coverage['schedule_id'],
                    'overall': best_overall['schedule_id']
                },
                'summary': {
                    'avg_headway_score': np.mean([r['metrics'].headway_consistency_score for r in results]),
                    'avg_wait_score': np.mean([r['metrics'].wait_time_score for r in results]),
                    'avg_coverage_score': np.mean([r['metrics'].coverage_score for r in results]),
                    'avg_overall_score': np.mean([r['metrics'].overall_quality_score for r in results])
                }
            }
        
        return {'individual_results': [], 'best_performers': {}, 'summary': {}}
