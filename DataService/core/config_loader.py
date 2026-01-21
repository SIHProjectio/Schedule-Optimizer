"""
Configuration loader for DataService
Loads and manages configuration from JSON files
"""
import json
import os
from datetime import time
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path


class ConfigLoader:
    """Load and manage DataService configuration"""
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize config loader
        
        Args:
            config_dir: Path to config directory. If None, uses default location.
        """
        if config_dir is None:
            # Default to config directory in parent of core module
            config_dir = str(Path(__file__).parent.parent / 'config')
        self.config_dir = Path(config_dir)
        
        self._operational_config: Optional[Dict] = None
        self._routes_config: Optional[Dict] = None
        self._infrastructure_config: Optional[Dict] = None
    
    def _load_json(self, filename: str) -> Dict:
        """Load JSON configuration file"""
        filepath = self.config_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            return json.load(f)
    
    @property
    def operational_config(self) -> Dict:
        """Get operational configuration"""
        if self._operational_config is None:
            self._operational_config = self._load_json('operational_config.json')
        return self._operational_config
    
    @property
    def routes_config(self) -> Dict:
        """Get routes configuration"""
        if self._routes_config is None:
            self._routes_config = self._load_json('routes.json')
        return self._routes_config
    
    @property
    def infrastructure_config(self) -> Dict:
        """Get infrastructure configuration"""
        if self._infrastructure_config is None:
            self._infrastructure_config = self._load_json('infrastructure.json')
        return self._infrastructure_config
    
    def get_operational_hours(self) -> Dict[str, Any]:
        """Get operational hours configuration"""
        return self.operational_config['operational']
    
    def get_peak_hours(self) -> List[Tuple[time, time]]:
        """Get peak hours as list of (start_time, end_time) tuples"""
        peak_hours_config = self.operational_config['operational']['peak_hours']
        result = []
        for period in peak_hours_config:
            start_h, start_m = map(int, period['start'].split(':'))
            end_h, end_m = map(int, period['end'].split(':'))
            result.append((time(start_h, start_m), time(end_h, end_m)))
        return result
    
    def get_start_time(self) -> time:
        """Get service start time"""
        h, m = map(int, self.operational_config['operational']['start_time'].split(':'))
        return time(h, m)
    
    def get_end_time(self) -> time:
        """Get service end time"""
        h, m = map(int, self.operational_config['operational']['end_time'].split(':'))
        return time(h, m)
    
    def get_route_config(self, route_name: str = "Aluva-Pettah Line") -> Dict:
        """Get configuration for a specific route"""
        routes = self.routes_config['routes']
        if route_name not in routes:
            # Return first available route as fallback
            route_name = list(routes.keys())[0]
        return routes[route_name]
    
    def get_available_routes(self) -> List[str]:
        """Get list of available route names"""
        return list(self.routes_config['routes'].keys())
    
    def get_depot_config(self, depot_name: str = "Muttom_Depot") -> Dict:
        """Get configuration for a specific depot"""
        depots = self.infrastructure_config['depots']
        if depot_name not in depots:
            # Return first available depot as fallback
            depot_name = list(depots.keys())[0]
        return depots[depot_name]
    
    def get_available_depots(self) -> List[str]:
        """Get list of available depot names"""
        return list(self.infrastructure_config['depots'].keys())
    
    def get_advertisers(self) -> List[str]:
        """Get list of advertisers"""
        return self.infrastructure_config['advertisers']
    
    def get_unavailable_reasons(self) -> List[str]:
        """Get list of train unavailable reasons"""
        return self.infrastructure_config['unavailable_reasons']
    
    def get_route_defaults(self) -> Dict:
        """Get route default parameters"""
        return self.operational_config['route']
    
    def get_fleet_defaults(self) -> Dict:
        """Get fleet default parameters"""
        return self.operational_config['fleet']
    
    def get_optimization_config(self) -> Dict:
        """Get optimization configuration"""
        return self.operational_config['optimization']


# Global config loader instance
_config_loader: Optional[ConfigLoader] = None


def get_config_loader(config_dir: Optional[str] = None) -> ConfigLoader:
    """
    Get or create global config loader instance
    
    Args:
        config_dir: Path to config directory. If None, uses default.
        
    Returns:
        ConfigLoader instance
    """
    global _config_loader
    if _config_loader is None or config_dir is not None:
        _config_loader = ConfigLoader(config_dir)
    return _config_loader


def reload_config():
    """Reload configuration from files"""
    global _config_loader
    _config_loader = None
