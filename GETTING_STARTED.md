# Getting Started - Metro Train Scheduling System

## ⚡ Quick Start (5 minutes)

### Step 1: Install Dependencies
```bash
cd /home/arpbansal/code/sih2025/mlservice
pip install fastapi uvicorn pydantic python-multipart
```

### Step 2: Test the System
```bash
python test_system.py
```

Expected output:
```
✓ PASS: Imports
✓ PASS: Data Generation
✓ PASS: Schedule Optimization
✓ PASS: Data Models
✓ PASS: JSON Export

Results: 5/5 tests passed
🎉 All tests passed! System is ready to use.
```

### Step 3: Run Demo
```bash
python demo_schedule.py
```

This generates a complete schedule and saves it to `sample_schedule.json`.

### Step 4: Start API Server
```bash
python run_api.py
```

Then open in browser:
- **API Docs**: http://localhost:8000/docs
- **Example Schedule**: http://localhost:8000/api/v1/schedule/example

---

## 🎓 Learning Path

### Beginner: Understand the Basics

1. **Read the schedule example**
   ```bash
   # Generate and view a sample schedule
   python demo_schedule.py
   cat sample_schedule.json | head -50
   ```

2. **Try the quick examples**
   ```bash
   python quickstart.py
   ```
   This shows 6 different usage patterns.

### Intermediate: Use the API

1. **Start the server**
   ```bash
   python run_api.py
   ```

2. **Test with curl**
   ```bash
   # Quick generation
   curl "http://localhost:8000/api/v1/generate/quick?date=2025-10-25&num_trains=30"
   
   # Custom parameters
   curl -X POST "http://localhost:8000/api/v1/generate" \
     -H "Content-Type: application/json" \
     -d '{"date":"2025-10-25","num_trains":30,"min_service_trains":22}'
   ```

3. **Use from Python**
   ```python
   import requests
   
   response = requests.post(
       "http://localhost:8000/api/v1/generate/quick",
       params={"date": "2025-10-25", "num_trains": 30}
   )
   schedule = response.json()
   print(f"Schedule ID: {schedule['schedule_id']}")
   ```

### Advanced: Integrate & Customize

1. **Import as library**
   ```python
   from DataService import MetroDataGenerator, MetroScheduleOptimizer
   
   generator = MetroDataGenerator(num_trains=30)
   route = generator.generate_route()
   health = generator.generate_train_health_statuses()
   
   optimizer = MetroScheduleOptimizer(
       date="2025-10-25",
       num_trains=30,
       route=route,
       train_health=health
   )
   schedule = optimizer.optimize_schedule()
   ```

2. **Customize optimization**
   ```python
   schedule = optimizer.optimize_schedule(
       min_service_trains=25,  # More trains in service
       min_standby=5,          # More standby
       max_daily_km=280        # Lower km limit
   )
   ```

3. **Access detailed data**
   ```python
   # Fleet summary
   print(f"In service: {schedule.fleet_summary.revenue_service}")
   
   # Individual train
   train = schedule.trainsets[0]
   print(f"Train: {train.trainset_id}")
   print(f"Status: {train.status}")
   print(f"Readiness: {train.readiness_score}")
   
   # Service blocks
   for block in train.service_blocks:
       print(f"{block.origin} → {block.destination} at {block.departure_time}")
   ```

---

## 📖 Understanding the Output

### Schedule Structure

```
DaySchedule
├── schedule_id          # Unique identifier
├── generated_at         # Timestamp
├── valid_from/until     # Operating period
├── depot                # Depot name
├── trainsets[]          # Array of all trains
│   ├── trainset_id
│   ├── status          # REVENUE_SERVICE, STANDBY, MAINTENANCE, CLEANING
│   ├── service_blocks[] # Operating assignments
│   ├── fitness_certificates
│   ├── job_cards
│   └── readiness_score
├── fleet_summary        # Status counts
├── optimization_metrics # Performance data
└── conflicts_and_alerts # Issues
```

### Train Status Types

1. **REVENUE_SERVICE**: Active passenger service
   - Has `assigned_duty` (e.g., "DUTY-A1")
   - Has `service_blocks` with trips
   - Primary operational trains

2. **STANDBY**: Ready for deployment
   - No active service blocks
   - Can be called up if needed
   - `standby_reason` explains why

3. **MAINTENANCE**: Under repair
   - Has `maintenance_type` (e.g., "SCHEDULED_INSPECTION")
   - `ibl_bay` location
   - `estimated_completion` time
   - May have blocking job cards

4. **CLEANING**: Washing/interior cleaning
   - `cleaning_bay` location
   - `cleaning_type` (DEEP_INTERIOR, EXTERIOR, FULL)
   - `scheduled_service_start` for later service

### Service Blocks

Each block represents a continuous operating period:
```json
{
  "block_id": "BLK-001",
  "departure_time": "05:30",      // Start time (HH:MM)
  "origin": "Aluva",              // Starting station
  "destination": "Pettah",        // Ending station
  "trip_count": 3,                // Number of round trips
  "estimated_km": 96              // Total kilometers
}
```

### Readiness Score

Calculated from:
- **Fitness certificates** (rolling stock, signalling, telecom)
- **Job cards** (open/blocking maintenance)
- **Component health** (brakes, HVAC, doors, etc.)

Range: 0.0 (not ready) to 1.0 (perfect)

---

## 🔧 Common Tasks

### Generate Schedule for Specific Date
```python
from DataService import MetroDataGenerator, MetroScheduleOptimizer

generator = MetroDataGenerator(num_trains=30)
route = generator.generate_route()
health = generator.generate_train_health_statuses()

optimizer = MetroScheduleOptimizer(
    date="2025-11-01",  # Change date here
    num_trains=30,
    route=route,
    train_health=health
)
schedule = optimizer.optimize_schedule()
```

### Save Schedule to File
```python
import json

schedule_dict = schedule.dict()
with open(f"schedule_{schedule.schedule_id}.json", 'w') as f:
    json.dump(schedule_dict, f, indent=2, default=str)
```

### Filter Trains by Status
```python
# Get all trains in revenue service
service_trains = [
    t for t in schedule.trainsets 
    if t.status.value == "REVENUE_SERVICE"
]

# Get trains with alerts
alerted_trains = [
    t for t in schedule.trainsets 
    if len(t.alerts) > 0
]
```

### Calculate Total Service Hours
```python
from datetime import datetime, time

start = datetime.combine(datetime.today(), time(5, 0))
end = datetime.combine(datetime.today(), time(23, 0))
hours = (end - start).total_seconds() / 3600
print(f"Total service hours: {hours}")  # 18 hours
```

---

## 🐛 Troubleshooting

### "Module not found"
```bash
# Make sure you're in the project directory
cd /home/arpbansal/code/sih2025/mlservice

# Install dependencies
pip install -r requirements.txt
```

### "Port 8000 already in use"
```bash
# Use a different port
python -c "from DataService.api import app; import uvicorn; uvicorn.run(app, port=8001)"
```

### "Import errors"
```bash
# Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Tests failing
```bash
# Reinstall dependencies
pip install --force-reinstall -r requirements.txt

# Run individual test
python -c "from test_system import test_imports; test_imports()"
```

---

## 📚 Next Steps

1. ✅ **Completed**: Basic setup and testing
2. 📖 **Read**: [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) for full details
3. 📖 **Read**: [DataService/README.md](DataService/README.md) for API docs
4. 🔧 **Customize**: Modify parameters for your use case
5. 🚀 **Deploy**: Use Docker for production deployment
6. 🧪 **Integrate**: Connect with existing systems

---

## 💡 Tips

- **Start simple**: Use `quickstart.py` to learn the API
- **Use the demo**: `demo_schedule.py` shows all features
- **Check the docs**: http://localhost:8000/docs when API is running
- **Read the code**: All files are well-documented
- **Test first**: Run `test_system.py` before making changes

---

## 🎯 Key Files

| File | Purpose |
|------|---------|
| `test_system.py` | Verify installation |
| `quickstart.py` | 6 usage examples |
| `demo_schedule.py` | Full demonstration |
| `run_api.py` | Start API server |
| `DataService/api.py` | API endpoints |
| `DataService/metro_models.py` | Data models |
| `IMPLEMENTATION_SUMMARY.md` | Complete guide |

---

## ⚡ One-Liner Tests

```bash
# Test imports
python -c "from DataService import MetroDataGenerator; print('✓ OK')"

# Generate quick data
python -c "from DataService import MetroDataGenerator; g = MetroDataGenerator(10); r = g.generate_route(); print(f'✓ Route: {r.name}')"

# Test API (if running)
curl -s http://localhost:8000/health | python -m json.tool
```

---

**Ready to start scheduling! 🚇**

For questions, see:
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - Full implementation details
- [README_NEW.md](README_NEW.md) - Project overview
- [DataService/README.md](DataService/README.md) - API documentation
