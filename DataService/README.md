# Metro Train Scheduling System - DataService API

A comprehensive FastAPI-based service for generating synthetic metro train scheduling data and optimizing daily train operations.

## 🎯 Overview

This system generates realistic metro train schedules for a single-line metro network with:
- **25-40 trainsets** with varying health status
- **25 stations** on a bidirectional route
- **Operating hours**: 5:00 AM - 11:00 PM
- **Real-world constraints**: maintenance, fitness certificates, branding priorities, mileage balancing

## 🚇 Features

### Data Generation
- **Train Health Status**: Fully healthy, partially available, or under maintenance
- **Fitness Certificates**: Rolling stock, signalling, and telecom certificates with expiry tracking
- **Job Cards**: Open maintenance tasks with blocking status
- **Component Health**: IoT-style monitoring of brakes, HVAC, doors, bogies, etc.
- **Branding/Advertising**: Contract tracking with exposure priorities
- **Depot Layout**: Stabling bays, IBL bays, and washing bays

### Schedule Optimization
- **Multi-objective optimization** balancing:
  - Service readiness (35%)
  - Mileage balancing (25%)
  - Branding priority (20%)
  - Operational cost (20%)
- **Constraint satisfaction**: Fitness certificates, maintenance requirements, availability windows
- **Service block generation**: Optimal trip assignments throughout the day
- **Fleet allocation**: Revenue service, standby, maintenance, and cleaning assignments

### API Endpoints

#### Generate Complete Schedule
```bash
POST /api/v1/generate
Content-Type: application/json

{
  "date": "2025-10-25",
  "num_trains": 30,
  "num_stations": 25,
  "route_name": "Aluva-Pettah Line",
  "depot_name": "Muttom_Depot",
  "min_service_trains": 22,
  "min_standby_trains": 3,
  "max_daily_km_per_train": 300,
  "balance_mileage": true,
  "prioritize_branding": true
}
```

#### Quick Schedule Generation
```bash
POST /api/v1/generate/quick?date=2025-10-25&num_trains=30&num_stations=25
```

#### Get Example Schedule
```bash
GET /api/v1/schedule/example
```

#### Get Route Information
```bash
GET /api/v1/route/25
```

#### Get Train Health Status
```bash
GET /api/v1/trains/health/30
```

#### Get Depot Layout
```bash
GET /api/v1/depot/layout
```

## 📦 Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. **Clone the repository**
```bash
cd /home/arpbansal/code/sih2025/mlservice
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

Requirements include:
- `fastapi>=0.104.1` - Web framework
- `uvicorn[standard]>=0.24.0` - ASGI server
- `pydantic>=2.5.0` - Data validation
- `ortools>=9.14.6206` - Optimization (optional)

## 🚀 Usage

### Option 1: Run Demo Script

Test the system without starting the API:

```bash
python demo_schedule.py
```

This will:
- Generate synthetic metro data
- Optimize a daily schedule
- Display comprehensive results
- Save output to `sample_schedule.json`

### Option 2: Start FastAPI Server

```bash
python run_api.py
```

The API will be available at:
- **Base URL**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

### Option 3: Use uvicorn directly

```bash
uvicorn DataService.api:app --reload --host 0.0.0.0 --port 8000
```

## 📊 Schedule Output Structure

```json
{
  "schedule_id": "KMRL-2025-10-25-DAWN",
  "generated_at": "2025-10-24T23:45:00+05:30",
  "valid_from": "2025-10-25T05:00:00+05:30",
  "valid_until": "2025-10-25T23:00:00+05:30",
  "depot": "Muttom_Depot",
  
  "trainsets": [
    {
      "trainset_id": "TS-001",
      "status": "REVENUE_SERVICE",
      "priority_rank": 1,
      "assigned_duty": "DUTY-A1",
      "service_blocks": [
        {
          "block_id": "BLK-001",
          "departure_time": "05:30",
          "origin": "Aluva",
          "destination": "Pettah",
          "trip_count": 3,
          "estimated_km": 96
        }
      ],
      "daily_km_allocation": 224,
      "cumulative_km": 145620,
      "fitness_certificates": {
        "rolling_stock": {"valid_until": "2025-11-15", "status": "VALID"},
        "signalling": {"valid_until": "2025-10-30", "status": "VALID"},
        "telecom": {"valid_until": "2025-11-20", "status": "VALID"}
      },
      "job_cards": {"open": 0, "blocking": []},
      "branding": {
        "advertiser": "COCACOLA-2024",
        "contract_hours_remaining": 340,
        "exposure_priority": "HIGH"
      },
      "readiness_score": 0.98,
      "constraints_met": true
    }
  ],
  
  "fleet_summary": {
    "total_trainsets": 30,
    "revenue_service": 22,
    "standby": 4,
    "maintenance": 2,
    "cleaning": 2,
    "availability_percent": 93.3
  },
  
  "optimization_metrics": {
    "mileage_variance_coefficient": 0.042,
    "avg_readiness_score": 0.91,
    "branding_sla_compliance": 1.0,
    "shunting_movements_required": 8,
    "total_planned_km": 5280,
    "fitness_expiry_violations": 0
  },
  
  "conflicts_and_alerts": [...],
  "decision_rationale": {...}
}
```

## 🏗️ Architecture

```
mlservice/
├── DataService/
│   ├── __init__.py
│   ├── api.py                    # FastAPI application
│   ├── metro_models.py           # Pydantic data models
│   ├── metro_data_generator.py   # Synthetic data generation
│   ├── schedule_optimizer.py     # Schedule optimization logic
│   ├── enhanced_generator.py     # (existing)
│   ├── synthetic_base.py         # (existing)
│   └── synthetic_extend.py       # (existing)
├── greedyOptim/                  # Optimization algorithms
├── demo_schedule.py              # Demo/test script
├── run_api.py                    # API startup script
├── requirements.txt
└── README.md
```

## 🔧 Configuration

### Train Health Categories

- **Fully Healthy** (65%): Available entire operational day
- **Partially Healthy** (20%): Available for limited hours
- **Unavailable** (15%): Not available for service (maintenance/repairs)

### Train Status Types

- `REVENUE_SERVICE`: Active passenger service
- `STANDBY`: Ready for deployment
- `MAINTENANCE`: Under repair/inspection
- `CLEANING`: Washing/interior cleaning
- `OUT_OF_SERVICE`: Long-term unavailable

### Optimization Weights

Default objective weights (configurable):
```python
{
  "service_readiness": 0.35,
  "mileage_balancing": 0.25,
  "branding_priority": 0.20,
  "operational_cost": 0.20
}
```

## 📝 API Examples

### cURL Examples

**Generate schedule:**
```bash
curl -X POST "http://localhost:8000/api/v1/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "date": "2025-10-25",
    "num_trains": 30,
    "num_stations": 25,
    "min_service_trains": 22
  }'
```

**Quick generation:**
```bash
curl "http://localhost:8000/api/v1/generate/quick?date=2025-10-25&num_trains=30"
```

**Health check:**
```bash
curl "http://localhost:8000/health"
```

### Python Client Example

```python
import requests

# Generate schedule
response = requests.post(
    "http://localhost:8000/api/v1/generate",
    json={
        "date": "2025-10-25",
        "num_trains": 30,
        "num_stations": 25,
        "min_service_trains": 22,
        "min_standby_trains": 3
    }
)

schedule = response.json()
print(f"Schedule ID: {schedule['schedule_id']}")
print(f"Trains in service: {schedule['fleet_summary']['revenue_service']}")
```

## 🧪 Testing

Run the demo script to test all functionality:

```bash
python demo_schedule.py
```

Expected output:
- ✓ Data generation statistics
- ✓ Route information
- ✓ Train health summary
- ✓ Optimization results
- ✓ Fleet status breakdown
- ✓ Sample train details
- ✓ JSON export

## 🎨 Key Concepts

### Service Blocks
Continuous operating periods with specific origin/destination and trip counts. Each block represents a trainset's assignment for part of the day.

### Readiness Score
Computed metric (0.0-1.0) considering:
- Fitness certificate validity
- Open/blocking job cards
- Component health scores
- Days since maintenance

### Mileage Balancing
Distributes daily kilometers to equalize cumulative mileage across the fleet, extending overall fleet life.

### Branding Priority
Trains with active advertising contracts get preferential assignment to maximize exposure (revenue optimization).

## 🔍 Monitoring & Alerts

The system generates alerts for:
- Certificate expirations (EXPIRING_SOON, EXPIRED)
- Blocking maintenance (job cards preventing service)
- Fitness violations
- Constraint conflicts

## 🤝 Integration

### With Existing Optimizer

The DataService can integrate with existing `greedyOptim` algorithms:

```python
from greedyOptim.scheduler import TrainsetSchedulingOptimizer
from DataService.metro_data_generator import MetroDataGenerator

# Generate synthetic data
generator = MetroDataGenerator(num_trains=30)
# ... use with existing optimizer
```

### As Microservice

Deploy as standalone microservice:
```bash
docker build -t metro-scheduler .
docker run -p 8000:8000 metro-scheduler
```

## 📈 Future Enhancements

- [ ] Real-time schedule adjustments
- [ ] Machine learning for demand prediction
- [ ] Driver/crew scheduling integration
- [ ] Energy consumption optimization
- [ ] Passenger flow simulation
- [ ] Weather impact modeling
- [ ] Multi-line network support

## 📄 License

[Add your license information]

## 👥 Contributors

[Add contributor information]

## 📞 Support

For issues, questions, or contributions:
- GitHub Issues: [repository URL]
- Email: [contact email]

---

**Built for Smart India Hackathon 2025** 🇮🇳
