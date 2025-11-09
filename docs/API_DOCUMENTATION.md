# GreedyOptim Scheduling API Documentation

## Overview

The GreedyOptim API provides advanced train scheduling optimization using multiple algorithms including Genetic Algorithms, Particle Swarm Optimization, CMA-ES, and more. This API allows you to submit your own trainset data and receive optimized scheduling recommendations.

**Base URL:** `http://localhost:8001`

**API Version:** 2.0.0

---

## Quick Start

### 1. Start the API Server

```bash
python api/run_greedyoptim_api.py
```

The API will be available at:
- **API Endpoint:** http://localhost:8001
- **Interactive Docs:** http://localhost:8001/docs
- **Alternative Docs:** http://localhost:8001/redoc

### 2. Test the API

```bash
python api/test_greedyoptim_api.py
```

---

## Authentication

Currently, the API does not require authentication. Configure authentication as needed for production use.

---

## Endpoints

### 1. Health Check

**GET** `/health`

Check if the API is running.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-11-09T10:30:00",
  "service": "greedyoptim-api"
}
```

---

### 2. List Available Methods

**GET** `/methods`

Get information about all available optimization methods.

**Response:**
```json
{
  "available_methods": {
    "ga": {
      "name": "Genetic Algorithm",
      "description": "Evolutionary optimization using selection, crossover, and mutation",
      "typical_time": "medium",
      "solution_quality": "high"
    },
    "pso": {
      "name": "Particle Swarm Optimization",
      "description": "Swarm intelligence-based optimization",
      "typical_time": "medium",
      "solution_quality": "high"
    },
    "cmaes": {
      "name": "CMA-ES",
      "description": "Covariance Matrix Adaptation Evolution Strategy",
      "typical_time": "medium-high",
      "solution_quality": "very high"
    },
    ...
  },
  "default_method": "ga",
  "recommended_for_speed": "ga",
  "recommended_for_quality": "ensemble"
}
```

---

### 3. Optimize Schedule

**POST** `/optimize`

Submit trainset data and receive an optimized schedule.

**Request Body:**
```json
{
  "trainset_status": [
    {
      "trainset_id": "KMRL-01",
      "operational_status": "Available",
      "last_maintenance_date": "2025-10-01",
      "total_mileage_km": 45000.0,
      "age_years": 3.5
    },
    ...
  ],
  "fitness_certificates": [
    {
      "trainset_id": "KMRL-01",
      "department": "Safety",
      "status": "Valid",
      "issue_date": "2025-01-01",
      "expiry_date": "2026-01-01"
    },
    ...
  ],
  "job_cards": [
    {
      "trainset_id": "KMRL-01",
      "job_id": "JOB-001",
      "priority": "Medium",
      "status": "Closed",
      "description": "Routine inspection",
      "estimated_hours": 2.0
    },
    ...
  ],
  "component_health": [
    {
      "trainset_id": "KMRL-01",
      "component": "Brakes",
      "status": "Good",
      "wear_level": 25.0,
      "last_inspection": "2025-10-15"
    },
    ...
  ],
  "method": "ga",
  "config": {
    "required_service_trains": 15,
    "min_standby": 2,
    "population_size": 50,
    "generations": 100,
    "mutation_rate": 0.1,
    "crossover_rate": 0.8,
    "elite_size": 5
  }
}
```

**Field Descriptions:**

**trainset_status** (required):
- `trainset_id`: Unique identifier (string)
- `operational_status`: One of: `Available`, `In-Service`, `Maintenance`, `Standby`, `Out-of-Order`
- `last_maintenance_date`: ISO date string (optional)
- `total_mileage_km`: Total kilometers traveled (optional)
- `age_years`: Age of trainset in years (optional)

**fitness_certificates** (required):
- `trainset_id`: Must match trainset_status
- `department`: One of: `Safety`, `Operations`, `Technical`, `Electrical`, `Mechanical`
- `status`: One of: `Valid`, `Expired`, `Expiring-Soon`, `Suspended`
- `issue_date`: ISO date string (optional)
- `expiry_date`: ISO date string (optional)

**job_cards** (required, can be empty array):
- `trainset_id`: Must match trainset_status
- `job_id`: Unique job identifier
- `priority`: One of: `Critical`, `High`, `Medium`, `Low`
- `status`: One of: `Open`, `In-Progress`, `Closed`, `Pending-Parts`
- `description`: Job description (optional)
- `estimated_hours`: Estimated completion time (optional)

**component_health** (required):
- `trainset_id`: Must match trainset_status
- `component`: Component name (e.g., `Brakes`, `HVAC`, `Doors`, `Propulsion`)
- `status`: One of: `Good`, `Fair`, `Warning`, `Critical`
- `wear_level`: Wear percentage 0-100 (optional)
- `last_inspection`: ISO date string (optional)

**method** (optional, default: "ga"):
- `ga`: Genetic Algorithm (recommended for most cases)
- `pso`: Particle Swarm Optimization
- `cmaes`: CMA-ES (best quality, slower)
- `sa`: Simulated Annealing
- `nsga2`: Multi-objective optimization
- `adaptive`: Auto-selects best method
- `ensemble`: Runs multiple methods (best quality, slowest)

**config** (optional):
- `required_service_trains`: Minimum trains needed in service (default: 15)
- `min_standby`: Minimum standby trains (default: 2)
- `population_size`: Algorithm population size (default: 50, range: 10-200)
- `generations`: Number of iterations (default: 100, range: 10-1000)
- `mutation_rate`: Mutation probability (default: 0.1, range: 0.0-1.0)
- `crossover_rate`: Crossover probability (default: 0.8, range: 0.0-1.0)
- `elite_size`: Number of elite solutions preserved (default: 5)

**Response:**
```json
{
  "success": true,
  "method": "ga",
  "fitness_score": 0.8542,
  "service_trains": ["KMRL-01", "KMRL-02", "KMRL-03", ...],
  "standby_trains": ["KMRL-15", "KMRL-16"],
  "maintenance_trains": ["KMRL-17", "KMRL-18"],
  "unavailable_trains": [],
  "num_service": 15,
  "num_standby": 2,
  "num_maintenance": 2,
  "num_unavailable": 0,
  "service_score": 0.95,
  "standby_score": 0.85,
  "health_score": 0.78,
  "certificate_score": 0.92,
  "execution_time_seconds": 2.341,
  "timestamp": "2025-11-09T10:35:00",
  "constraints_satisfied": true,
  "warnings": null
}
```

---

### 4. Compare Methods

**POST** `/compare`

Compare multiple optimization methods on the same data.

**Request Body:**
```json
{
  "trainset_status": [...],
  "fitness_certificates": [...],
  "job_cards": [...],
  "component_health": [...],
  "methods": ["ga", "pso", "cmaes"],
  "config": {
    "required_service_trains": 15,
    "min_standby": 2,
    "population_size": 30,
    "generations": 50
  }
}
```

**Response:**
```json
{
  "methods": {
    "ga": {
      "success": true,
      "method": "ga",
      "fitness_score": 0.8542,
      "service_trains": [...],
      "execution_time_seconds": 1.234,
      ...
    },
    "pso": {
      "success": true,
      "method": "pso",
      "fitness_score": 0.8398,
      ...
    },
    "cmaes": {
      "success": true,
      "method": "cmaes",
      "fitness_score": 0.8721,
      ...
    }
  },
  "summary": {
    "total_execution_time": 5.678,
    "methods_compared": 3,
    "best_method": "cmaes",
    "best_score": 0.8721,
    "timestamp": "2025-11-09T10:40:00"
  }
}
```

---

### 5. Generate Synthetic Data

**POST** `/generate-synthetic`

Generate synthetic test data for testing purposes.

**Request Body:**
```json
{
  "num_trainsets": 25,
  "maintenance_rate": 0.1,
  "availability_rate": 0.8
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "trainset_status": [...],
    "fitness_certificates": [...],
    "job_cards": [...],
    "component_health": [...],
    "metadata": {...}
  },
  "metadata": {
    "num_trainsets": 25,
    "num_fitness_certificates": 125,
    "num_job_cards": 50,
    "num_component_health": 150,
    "generated_at": "2025-11-09T10:45:00"
  }
}
```

---

### 6. Validate Data

**POST** `/validate`

Validate your data structure before submitting for optimization.

**Request Body:**
```json
{
  "trainset_status": [...],
  "fitness_certificates": [...],
  "job_cards": [...],
  "component_health": [...],
  "method": "ga"
}
```

**Response (Valid):**
```json
{
  "valid": true,
  "message": "Data structure is valid",
  "num_trainsets": 25,
  "num_certificates": 125,
  "num_job_cards": 50,
  "num_component_health": 150
}
```

**Response (Invalid):**
```json
{
  "valid": false,
  "validation_errors": [
    "Missing required data section: trainset_status",
    "Invalid operational_status value: 'Running' for trainset KMRL-05"
  ],
  "suggestions": [
    "Check that all trainset_ids are consistent across sections",
    "Ensure operational_status values are valid (Available, In-Service, Maintenance, Standby, Out-of-Order)",
    "Verify certificate status values are valid (Valid, Expired, Expiring-Soon, Suspended)",
    "Verify certificate expiry dates are in ISO format",
    "Confirm component wear_level is between 0-100 if provided"
  ]
}
```

---

## Usage Examples

### Example 1: Basic Optimization (Python)

```python
import requests

# Your trainset data
data = {
    "trainset_status": [
        {"trainset_id": "KMRL-01", "operational_status": "Available"},
        {"trainset_id": "KMRL-02", "operational_status": "Available"},
        # ... more trainsets
    ],
    "fitness_certificates": [
        {
            "trainset_id": "KMRL-01",
            "department": "Safety",
            "status": "Valid",
            "expiry_date": "2026-01-01"
        },
        # ... more certificates
    ],
    "job_cards": [],  # No pending jobs
    "component_health": [
        {
            "trainset_id": "KMRL-01",
            "component": "Brakes",
            "status": "Good",
            "wear_level": 20.0
        },
        # ... more components
    ],
    "method": "ga",
    "config": {
        "required_service_trains": 15,
        "min_standby": 2
    }
}

# Send request
response = requests.post("http://localhost:8001/optimize", json=data)
result = response.json()

print(f"Fitness Score: {result['fitness_score']}")
print(f"Service Trains: {result['num_service']}")
print(f"Execution Time: {result['execution_time_seconds']}s")
```

### Example 2: Compare Methods (cURL)

```bash
curl -X POST "http://localhost:8001/compare" \
  -H "Content-Type: application/json" \
  -d '{
    "trainset_status": [...],
    "fitness_certificates": [...],
    "job_cards": [],
    "component_health": [...],
    "methods": ["ga", "pso"],
    "config": {
      "population_size": 30,
      "generations": 50
    }
  }'
```

### Example 3: Validate Before Optimizing (JavaScript)

```javascript
const data = {
  trainset_status: [...],
  fitness_certificates: [...],
  job_cards: [],
  component_health: [...],
  method: "ga"
};

// Validate first
const validateResponse = await fetch('http://localhost:8001/validate', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(data)
});

const validation = await validateResponse.json();

if (validation.valid) {
  // Now optimize
  const optimizeResponse = await fetch('http://localhost:8001/optimize', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data)
  });
  
  const result = await optimizeResponse.json();
  console.log('Optimization successful:', result);
} else {
  console.error('Validation errors:', validation.validation_errors);
}
```

---

## Error Handling

### HTTP Status Codes

- **200**: Success
- **400**: Bad Request (validation error)
- **500**: Internal Server Error

### Error Response Format

```json
{
  "error": "Data validation failed",
  "validation_errors": [
    "Missing required field: trainset_id in trainset_status",
    "Invalid operational_status value"
  ],
  "message": "Please fix the data structure and try again"
}
```

---

## Data Requirements

### Minimum Required Data

To successfully optimize a schedule, you must provide:

1. **At least 10 trainsets** in `trainset_status`
2. **At least one fitness certificate** per trainset
3. **Component health data** for each trainset
4. **job_cards** can be an empty array if no maintenance is pending

### Valid Status Values

**Operational Status:**
- `Available` - Ready for service
- `In-Service` - Currently operating
- `Maintenance` - Under maintenance
- `Standby` - On standby
- `Out-of-Order` - Not operational

**Certificate Status:**
- `Valid` - Certificate is valid
- `Expired` - Certificate has expired
- `Expiring-Soon` - Certificate expires within 30 days
- `Suspended` - Certificate suspended

**Job Priority:**
- `Critical` - Must be addressed immediately
- `High` - High priority
- `Medium` - Medium priority
- `Low` - Low priority

**Job Status:**
- `Open` - Not started
- `In-Progress` - Currently being worked on
- `Closed` - Completed
- `Pending-Parts` - Waiting for parts

**Component Status:**
- `Good` - Component in good condition
- `Fair` - Component acceptable
- `Warning` - Component needs attention soon
- `Critical` - Component requires immediate attention

---

## Performance Tips

### For Faster Results:
- Use `method: "ga"` (Genetic Algorithm)
- Reduce `population_size` (e.g., 30)
- Reduce `generations` (e.g., 50)
- Test with fewer trainsets first

### For Best Quality:
- Use `method: "ensemble"` (runs multiple algorithms)
- Increase `population_size` (e.g., 100)
- Increase `generations` (e.g., 200)
- Use `method: "cmaes"` for single-method optimization

### Recommended Configurations:

**Quick Testing (< 1 second):**
```json
{
  "method": "ga",
  "config": {
    "population_size": 20,
    "generations": 30
  }
}
```

**Production Use (2-5 seconds):**
```json
{
  "method": "ga",
  "config": {
    "population_size": 50,
    "generations": 100
  }
}
```

**High Quality (10-30 seconds):**
```json
{
  "method": "ensemble",
  "config": {
    "population_size": 100,
    "generations": 200
  }
}
```

---

## Rate Limits

Currently, no rate limits are enforced. Implement rate limiting for production use.

---

## Support

For issues or questions:
- Check the interactive documentation: http://localhost:8001/docs
- Run the test suite: `python api/test_greedyoptim_api.py`
- Review validation errors carefully - they indicate exactly what's wrong

---

## Changelog

### Version 2.0.0 (2025-11-09)
- Initial release of GreedyOptim API
- Support for multiple optimization algorithms
- Custom data input support
- Validation and synthetic data generation
- Method comparison capabilities
