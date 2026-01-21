# DataService Configuration System

Configuration files for customizing DataService behavior without code changes.

## 📁 Configuration Files

### `operational_config.json`
Operational parameters for metro service:
- **operational**: Service hours, peak hours, frequencies
- **route**: Speed, turnaround times, dwell times  
- **fleet**: Fleet size constraints
- **optimization**: Optimization weights and parameters

### `routes.json`
Route definitions with stations:
- **routes**: Named routes with station lists and distances
- Easy to add new metro lines
- Supports multiple route configurations

### `infrastructure.json`
Infrastructure and resources:
- **depots**: Depot layouts with bay configurations
- **advertisers**: List of advertising partners
- **unavailable_reasons**: Maintenance/fault categories

## 🔧 Usage

### Basic Usage (Default Config)
```python
from DataService import MetroDataGenerator

# Uses default config from DataService/config/
generator = MetroDataGenerator(num_trains=30)
route = generator.generate_route("Aluva-Pettah Line")
```

### Custom Config Directory
```python
from DataService import MetroDataGenerator

# Use custom config files
generator = MetroDataGenerator(
    num_trains=30,
    config_dir="/path/to/custom/config"
)
```

### Direct Config Access
```python
from DataService import get_config_loader

config = get_config_loader()

# Get operational hours
op_hours = config.get_operational_hours()
peak_hours = config.get_peak_hours()

# Get available routes
routes = config.get_available_routes()
print(f"Available routes: {routes}")

# Get depot configuration
depot = config.get_depot_config("Muttom_Depot")
print(f"Stabling bays: {depot['stabling_bays']}")
```

### Creating Custom Routes
```json
{
  "routes": {
    "My Custom Line": {
      "route_id": "CUSTOM-001",
      "total_distance_km": 20.0,
      "stations": [
        "Station A",
        "Station B",
        "Station C"
      ]
    }
  }
}
```

### Modifying Peak Hours
```json
{
  "operational": {
    "peak_hours": [
      {"start": "06:00", "end": "09:00"},
      {"start": "16:00", "end": "19:00"},
      {"start": "21:00", "end": "23:00"}
    ]
  }
}
```

## 🎯 Benefits

✅ **No Code Changes** - Modify behavior via JSON files  
✅ **Environment-Specific** - Different configs for dev/test/prod  
✅ **Multi-Metro Support** - Easy to add new metro systems  
✅ **Reusable** - Same codebase for different metros  
✅ **Testable** - Easy to test with different configurations

## 📊 Configuration Parameters

### Operational Hours
- `start_time`: Service start (format: "HH:MM")
- `end_time`: Service end (format: "HH:MM")
- `peak_hours`: List of peak periods
- `peak_frequency_minutes`: Train frequency during peak
- `off_peak_frequency_minutes`: Train frequency off-peak

### Route Defaults
- `avg_speed_kmh`: Average operating speed
- `turnaround_time_minutes`: Time at terminals
- `avg_dwell_time_seconds`: Station stop duration

### Fleet Parameters
- `min_trains`: Minimum fleet size
- `max_trains`: Maximum fleet size
- `default_trains`: Default number of trains
- `max_daily_km_per_train`: Maximum distance per train

### Optimization Weights
- `service_readiness`: Weight for train readiness (0-1)
- `mileage_balance`: Weight for mileage balancing (0-1)
- `branding_priority`: Weight for ad exposure (0-1)
- `operational_cost`: Weight for cost optimization (0-1)

## 🔄 Reloading Configuration

```python
from DataService import reload_config

# Make changes to JSON files...

# Reload configuration
reload_config()

# Now new config is active
```

## 🌍 Multi-Metro Support Example

```python
# Delhi Metro
delhi_generator = MetroDataGenerator(
    num_trains=200,
    config_dir="./config/delhi_metro"
)

# Bangalore Metro
bangalore_generator = MetroDataGenerator(
    num_trains=150,
    config_dir="./config/bangalore_metro"
)

# Kochi Metro (default)
kochi_generator = MetroDataGenerator(num_trains=30)
```

## 📝 Adding New Metro Systems

1. Create config directory: `config/your_metro/`
2. Copy JSON files from `DataService/config/`
3. Modify with your metro's parameters
4. Pass `config_dir` when creating generators

---

**Version**: 2.1.0 - Configuration-based architecture
