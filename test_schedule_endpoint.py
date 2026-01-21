"""
Test the /schedule endpoint with properly generated data.
Saves both input and output to JSON files.
Tests with both 25 and 30 trainsets.
Uses 3 data generation methods:
1. API's /generate-synthetic endpoint
2. DataService.enhanced_generator (our synthetic generator)
3. Local inline generator
"""
import requests
import json
from datetime import datetime, timedelta
import os
import random
import sys

# Add parent path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from DataService.generators.enhanced_generator import EnhancedMetroDataGenerator

BASE_URL = "http://localhost:7860"


def generate_synthetic_data_local(num_trainsets: int) -> dict:
    """Generate synthetic data locally without using API."""
    
    trainset_status = []
    fitness_certificates = []
    component_health = []
    
    # Status distribution: 80% available (IN_SERVICE/STANDBY), 20% other
    available_count = int(num_trainsets * 0.80)
    
    departments = ["ROLLING_STOCK", "ELECTRICAL", "MECHANICAL", "SAFETY", "OPERATIONS"]
    components = ["Traction_Motor", "Brake_System", "HVAC", "Door_System", "Pantograph", 
                  "Bogie", "Coupler", "Battery"]
    
    for i in range(1, num_trainsets + 1):
        ts_id = f"TS-{i:03d}"
        
        # Determine operational status
        if i <= available_count:
            if i % 3 == 0:
                status = "STANDBY"
            else:
                status = "IN_SERVICE"
        else:
            status = random.choice(["MAINTENANCE", "OUT_OF_SERVICE"])
        
        # Trainset status
        trainset_status.append({
            "trainset_id": ts_id,
            "operational_status": status,
            "last_maintenance_date": (datetime.now() - timedelta(days=random.randint(10, 90))).strftime("%Y-%m-%d"),
            "total_mileage_km": round(random.uniform(50000, 200000), 2),
            "age_years": random.randint(1, 8)
        })
        
        # Fitness certificates - one per department
        for dept in departments:
            issue_date = datetime.now() - timedelta(days=random.randint(30, 180))
            expiry_date = issue_date + timedelta(days=365)
            
            # 90% valid, 10% expired or other
            if random.random() < 0.90:
                cert_status = "ISSUED"
            else:
                cert_status = random.choice(["EXPIRED", "PENDING", "SUSPENDED"])
            
            fitness_certificates.append({
                "trainset_id": ts_id,
                "department": dept,
                "status": cert_status,
                "issue_date": issue_date.strftime("%Y-%m-%d"),
                "expiry_date": expiry_date.strftime("%Y-%m-%d")
            })
        
        # Component health - one per component
        for comp in components:
            # 85% healthy, 15% with issues
            if random.random() < 0.85:
                comp_status = random.choice(["EXCELLENT", "GOOD"])
                wear = round(random.uniform(10, 50), 1)
            else:
                comp_status = random.choice(["FAIR", "POOR", "CRITICAL"])
                wear = round(random.uniform(60, 95), 1)
            
            component_health.append({
                "trainset_id": ts_id,
                "component": comp,
                "status": comp_status,
                "wear_level": wear,
                "last_inspection": (datetime.now() - timedelta(days=random.randint(1, 30))).strftime("%Y-%m-%d")
            })
    
    return {
        "trainset_status": trainset_status,
        "fitness_certificates": fitness_certificates,
        "component_health": component_health
    }


def test_with_api_generator(num_trainsets: int, output_dir: str):
    """Test using API's /generate-synthetic endpoint."""
    
    print(f"\n{'='*70}")
    print(f"Test with API Generator - {num_trainsets} trainsets")
    print(f"{'='*70}")
    
    # Step 1: Generate synthetic data via API
    print("\nStep 1: Generating synthetic data via /generate-synthetic...")
    
    gen_response = requests.post(
        f"{BASE_URL}/generate-synthetic",
        json={"num_trainsets": num_trainsets}
    )
    
    if gen_response.status_code != 200:
        print(f"❌ Failed to generate synthetic data: {gen_response.status_code}")
        print(gen_response.text)
        return False
    
    gen_result = gen_response.json()
    synthetic_data = gen_result["data"]
    
    # Remove job_cards
    synthetic_data.pop("job_cards", None)
    
    print(f"✓ Generated: {len(synthetic_data['trainset_status'])} trainsets, "
          f"{len(synthetic_data['fitness_certificates'])} certificates, "
          f"{len(synthetic_data['component_health'])} component records")
    
    # Count by status
    status_counts = {}
    for ts in synthetic_data['trainset_status']:
        status = ts['operational_status']
        status_counts[status] = status_counts.get(status, 0) + 1
    print(f"  Status breakdown: {status_counts}")
    
    # Step 2: Prepare schedule request
    schedule_request = {
        "trainset_status": synthetic_data["trainset_status"],
        "fitness_certificates": synthetic_data["fitness_certificates"],
        "component_health": synthetic_data["component_health"],
        "date": datetime.now().strftime("%Y-%m-%d"),
        "method": "ga",
        "config": {
            "required_service_trains": 15,
            "min_standby": 2,
            "population_size": 50,
            "generations": 100
        }
    }
    
    # Save input
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_file = f"{output_dir}/api_gen_{num_trainsets}_input_{timestamp}.json"
    output_file = f"{output_dir}/api_gen_{num_trainsets}_output_{timestamp}.json"
    
    with open(input_file, 'w') as f:
        json.dump(schedule_request, f, indent=2)
    print(f"✓ Input saved: {input_file}")
    
    # Step 3: Call /schedule endpoint
    print("\nStep 2: Calling /schedule endpoint...")
    
    schedule_response = requests.post(
        f"{BASE_URL}/schedule",
        json=schedule_request
    )
    
    if schedule_response.status_code != 200:
        print(f"❌ Schedule endpoint failed: {schedule_response.status_code}")
        error_file = f"{output_dir}/api_gen_{num_trainsets}_error_{timestamp}.json"
        with open(error_file, 'w') as f:
            json.dump({"status_code": schedule_response.status_code, "error": schedule_response.text}, f, indent=2)
        print(f"Error saved: {error_file}")
        return False
    
    schedule_result = schedule_response.json()
    
    with open(output_file, 'w') as f:
        json.dump(schedule_result, f, indent=2)
    print(f"✓ Output saved: {output_file}")
    
    # Print summary
    print_schedule_summary(schedule_result)
    
    return True


def test_with_local_generator(num_trainsets: int, output_dir: str):
    """Test using locally generated data."""
    
    print(f"\n{'='*70}")
    print(f"Test with Local Generator - {num_trainsets} trainsets")
    print(f"{'='*70}")
    
    # Step 1: Generate synthetic data locally
    print("\nStep 1: Generating synthetic data locally...")
    
    synthetic_data = generate_synthetic_data_local(num_trainsets)
    
    print(f"✓ Generated: {len(synthetic_data['trainset_status'])} trainsets, "
          f"{len(synthetic_data['fitness_certificates'])} certificates, "
          f"{len(synthetic_data['component_health'])} component records")
    
    # Count by status
    status_counts = {}
    for ts in synthetic_data['trainset_status']:
        status = ts['operational_status']
        status_counts[status] = status_counts.get(status, 0) + 1
    print(f"  Status breakdown: {status_counts}")
    
    # Step 2: Prepare schedule request
    schedule_request = {
        "trainset_status": synthetic_data["trainset_status"],
        "fitness_certificates": synthetic_data["fitness_certificates"],
        "component_health": synthetic_data["component_health"],
        "date": datetime.now().strftime("%Y-%m-%d"),
        "method": "ga",
        "config": {
            "required_service_trains": 15,
            "min_standby": 2,
            "population_size": 50,
            "generations": 100
        }
    }
    
    # Save input
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_file = f"{output_dir}/local_gen_{num_trainsets}_input_{timestamp}.json"
    output_file = f"{output_dir}/local_gen_{num_trainsets}_output_{timestamp}.json"
    
    with open(input_file, 'w') as f:
        json.dump(schedule_request, f, indent=2)
    print(f"✓ Input saved: {input_file}")
    
    # Step 3: Call /schedule endpoint
    print("\nStep 2: Calling /schedule endpoint...")
    
    schedule_response = requests.post(
        f"{BASE_URL}/schedule",
        json=schedule_request
    )
    
    if schedule_response.status_code != 200:
        print(f"❌ Schedule endpoint failed: {schedule_response.status_code}")
        error_file = f"{output_dir}/local_gen_{num_trainsets}_error_{timestamp}.json"
        with open(error_file, 'w') as f:
            json.dump({"status_code": schedule_response.status_code, "error": schedule_response.text}, f, indent=2)
        print(f"Error saved: {error_file}")
        return False
    
    schedule_result = schedule_response.json()
    
    with open(output_file, 'w') as f:
        json.dump(schedule_result, f, indent=2)
    print(f"✓ Output saved: {output_file}")
    
    # Print summary
    print_schedule_summary(schedule_result)
    
    return True


def print_schedule_summary(schedule_result: dict):
    """Print a summary of the schedule result."""
    
    print(f"\n--- Schedule Summary ---")
    print(f"Schedule ID: {schedule_result.get('schedule_id')}")
    print(f"Generated At: {schedule_result.get('generated_at')}")
    print(f"Valid: {schedule_result.get('valid_from')} to {schedule_result.get('valid_until')}")
    print(f"Depot: {schedule_result.get('depot')}")
    
    # Fleet summary
    fleet = schedule_result.get('fleet_summary', {})
    print(f"\nFleet Summary:")
    print(f"  Total: {fleet.get('total_trainsets')}")
    print(f"  Revenue Service: {fleet.get('revenue_service')}")
    print(f"  Standby: {fleet.get('standby')}")
    print(f"  Maintenance: {fleet.get('maintenance')}")
    print(f"  Availability: {fleet.get('availability_percent', 0):.1f}%")
    
    # Optimization metrics
    metrics = schedule_result.get('optimization_metrics', {})
    print(f"\nOptimization:")
    print(f"  Fitness: {metrics.get('fitness_score', 0):.4f}")
    print(f"  Method: {metrics.get('method')}")
    print(f"  Total Planned KM: {metrics.get('total_planned_km', 0):.1f}")
    print(f"  Runtime: {metrics.get('optimization_runtime_ms', 0)} ms")
    
    # Trainset details
    trainsets = schedule_result.get('trainsets', [])
    print(f"\nTrainsets ({len(trainsets)}):")
    
    # Group by status
    by_status = {}
    for ts in trainsets:
        status = ts.get('status', 'UNKNOWN')
        if status not in by_status:
            by_status[status] = []
        by_status[status].append(ts)
    
    for status in ['REVENUE_SERVICE', 'STANDBY', 'MAINTENANCE']:
        if status in by_status:
            count = len(by_status[status])
            print(f"\n  {status} ({count}):")
            for ts in by_status[status][:3]:
                blocks = ts.get('service_blocks', [])
                block_info = f", {len(blocks)} blocks" if blocks else ""
                print(f"    {ts['trainset_id']}: km={ts.get('daily_km_allocation', 0):.1f}{block_info}")
            if count > 3:
                print(f"    ... and {count - 3} more")
    
    # Alerts
    alerts = schedule_result.get('alerts', [])
    if alerts:
        print(f"\nAlerts ({len(alerts)}):")
        for alert in alerts[:3]:
            print(f"  [{alert.get('severity')}] {alert.get('trainset_id')}: {alert.get('message')[:50]}")
        if len(alerts) > 3:
            print(f"  ... and {len(alerts) - 3} more")


def test_with_dataservice_generator(num_trainsets: int, output_dir: str):
    """Test using DataService.enhanced_generator (our synthetic generator)."""
    
    print(f"\n{'='*70}")
    print(f"Test with DataService Generator - {num_trainsets} trainsets")
    print(f"{'='*70}")
    
    # Step 1: Generate synthetic data using EnhancedMetroDataGenerator
    print("\nStep 1: Generating data with EnhancedMetroDataGenerator...")
    
    try:
        generator = EnhancedMetroDataGenerator(num_trainsets=num_trainsets)
        data = generator.generate_complete_enhanced_dataset()
    except Exception as e:
        print(f"❌ Failed to generate data: {e}")
        return False
    
    # Remove job_cards (not needed)
    synthetic_data = {
        "trainset_status": data["trainset_status"],
        "fitness_certificates": data["fitness_certificates"],
        "component_health": data["component_health"]
    }
    
    print(f"✓ Generated: {len(synthetic_data['trainset_status'])} trainsets, "
          f"{len(synthetic_data['fitness_certificates'])} certificates, "
          f"{len(synthetic_data['component_health'])} component records")
    
    # Count by status
    status_counts = {}
    for ts in synthetic_data['trainset_status']:
        status = ts['operational_status']
        status_counts[status] = status_counts.get(status, 0) + 1
    print(f"  Status breakdown: {status_counts}")
    
    # Step 2: Prepare schedule request
    schedule_request = {
        "trainset_status": synthetic_data["trainset_status"],
        "fitness_certificates": synthetic_data["fitness_certificates"],
        "component_health": synthetic_data["component_health"],
        "date": datetime.now().strftime("%Y-%m-%d"),
        "method": "ga",
        "config": {
            "required_service_trains": 15,
            "min_standby": 2,
            "population_size": 50,
            "generations": 100
        }
    }
    
    # Save input
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_file = f"{output_dir}/dataservice_gen_{num_trainsets}_input_{timestamp}.json"
    output_file = f"{output_dir}/dataservice_gen_{num_trainsets}_output_{timestamp}.json"
    
    with open(input_file, 'w') as f:
        json.dump(schedule_request, f, indent=2)
    print(f"✓ Input saved: {input_file}")
    
    # Step 3: Call /schedule endpoint
    print("\nStep 2: Calling /schedule endpoint...")
    
    schedule_response = requests.post(
        f"{BASE_URL}/schedule",
        json=schedule_request
    )
    
    if schedule_response.status_code != 200:
        print(f"❌ Schedule endpoint failed: {schedule_response.status_code}")
        error_file = f"{output_dir}/dataservice_gen_{num_trainsets}_error_{timestamp}.json"
        with open(error_file, 'w') as f:
            json.dump({"status_code": schedule_response.status_code, "error": schedule_response.text}, f, indent=2)
        print(f"Error saved: {error_file}")
        return False
    
    schedule_result = schedule_response.json()
    
    with open(output_file, 'w') as f:
        json.dump(schedule_result, f, indent=2)
    print(f"✓ Output saved: {output_file}")
    
    # Print summary
    print_schedule_summary(schedule_result)
    
    return True


def main():
    """Run all tests."""
    
    # Create output directory
    output_dir = "test_output"
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*70)
    print("Schedule Endpoint Test Suite")
    print("="*70)
    print(f"API URL: {BASE_URL}")
    print(f"Output directory: {output_dir}/")
    print("\nData generators being tested:")
    print("  1. API /generate-synthetic endpoint")
    print("  2. DataService.enhanced_generator (EnhancedMetroDataGenerator)")
    print("  3. Local inline generator")
    
    # Check API health
    try:
        health = requests.get(f"{BASE_URL}/health", timeout=5)
        if health.status_code != 200:
            print(f"\n❌ API not healthy: {health.status_code}")
            return
        print("\n✓ API is healthy")
    except Exception as e:
        print(f"\n❌ Cannot connect to API: {e}")
        print("Make sure the API is running: python api/greedyoptim_api.py")
        return
    
    results = []
    
    # Test 1: API generator with 25 trainsets
    results.append(("API Generator - 25 trainsets", test_with_api_generator(25, output_dir)))
    
    # Test 2: API generator with 30 trainsets
    results.append(("API Generator - 30 trainsets", test_with_api_generator(30, output_dir)))
    
    # Test 3: DataService generator with 25 trainsets
    results.append(("DataService Generator - 25 trainsets", test_with_dataservice_generator(25, output_dir)))
    
    # Test 4: DataService generator with 30 trainsets
    results.append(("DataService Generator - 30 trainsets", test_with_dataservice_generator(30, output_dir)))
    
    # Test 5: Local generator with 25 trainsets
    results.append(("Local Generator - 25 trainsets", test_with_local_generator(25, output_dir)))
    
    # Test 6: Local generator with 30 trainsets
    results.append(("Local Generator - 30 trainsets", test_with_local_generator(30, output_dir)))
    
    # Summary
    print("\n" + "="*70)
    print("Test Results Summary")
    print("="*70)
    
    for name, success in results:
        status = "✓ PASS" if success else "❌ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\nOutput files saved to: {output_dir}/")
    print("="*70)


if __name__ == "__main__":
    main()
