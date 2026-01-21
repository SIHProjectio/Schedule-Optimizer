"""
Simple Test Script - Verify Metro Scheduling System
Tests core functionality without requiring full API setup
"""
import sys
import traceback


def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    try:
        from DataService.core import models
        from DataService.generators import metro_generator
        from DataService.optimizers import schedule_optimizer
        print("  ✓ DataService modules imported successfully")
        return True
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        traceback.print_exc()
        return False


def test_data_generation():
    """Test data generation"""
    print("\nTesting data generation...")
    try:
        from DataService.generators.metro_generator import MetroDataGenerator
        
        generator = MetroDataGenerator(num_trains=10, num_stations=10)
        print(f"  ✓ Generator created for {len(generator.trainset_ids)} trains")
        
        # Test route generation
        route = generator.generate_route()
        print(f"  ✓ Route generated: {route.name} with {len(route.stations)} stations")
        
        # Test train health
        health = generator.generate_train_health_statuses()
        print(f"  ✓ Generated health status for {len(health)} trains")
        
        # Test certificates
        certs = generator.generate_fitness_certificates("TS-001")
        print(f"  ✓ Generated fitness certificates")
        
        return True
    except Exception as e:
        print(f"  ✗ Data generation failed: {e}")
        traceback.print_exc()
        return False


def test_schedule_optimization():
    """Test schedule optimization"""
    print("\nTesting schedule optimization...")
    try:
        from DataService.generators.metro_generator import MetroDataGenerator
        from DataService.optimizers.schedule_optimizer import MetroScheduleOptimizer
        from datetime import datetime
        
        # Setup
        generator = MetroDataGenerator(num_trains=15, num_stations=15)
        route = generator.generate_route()
        health = generator.generate_train_health_statuses()
        
        # Create optimizer
        optimizer = MetroScheduleOptimizer(
            date=datetime.now().strftime("%Y-%m-%d"),
            num_trains=15,
            route=route,
            train_health=health
        )
        print(f"  ✓ Optimizer created")
        
        # Generate schedule
        schedule = optimizer.optimize_schedule(min_service_trains=10, min_standby=2)
        print(f"  ✓ Schedule generated: {schedule.schedule_id}")
        print(f"    - Trains in service: {schedule.fleet_summary.revenue_service}")
        print(f"    - Total planned km: {schedule.optimization_metrics.total_planned_km}")
        print(f"    - Optimization time: {schedule.optimization_metrics.optimization_runtime_ms} ms")
        
        return True
    except Exception as e:
        print(f"  ✗ Schedule optimization failed: {e}")
        traceback.print_exc()
        return False


def test_models():
    """Test Pydantic models"""
    print("\nTesting data models...")
    try:
        from DataService.core.models import (
            ScheduleRequest, TrainHealthStatus, Route, Station
        )
        
        # Test ScheduleRequest
        request = ScheduleRequest(
            date="2025-10-25",
            num_trains=25,
            num_stations=25
        )
        print(f"  ✓ ScheduleRequest model validated")
        
        # Test Station
        station = Station(
            station_id="STN-001",
            name="Test Station",
            sequence=1,
            distance_from_origin_km=0.0
        )
        print(f"  ✓ Station model validated")
        
        return True
    except Exception as e:
        print(f"  ✗ Model validation failed: {e}")
        traceback.print_exc()
        return False


def test_json_export():
    """Test JSON export"""
    print("\nTesting JSON export...")
    try:
        import json
        from DataService.generators.metro_generator import MetroDataGenerator
        from DataService.optimizers.schedule_optimizer import MetroScheduleOptimizer
        from datetime import datetime
        
        generator = MetroDataGenerator(num_trains=10, num_stations=10)
        route = generator.generate_route()
        health = generator.generate_train_health_statuses()
        
        optimizer = MetroScheduleOptimizer(
            date=datetime.now().strftime("%Y-%m-%d"),
            num_trains=10,
            route=route,
            train_health=health
        )
        
        schedule = optimizer.optimize_schedule()
        
        # Convert to dict and save
        schedule_dict = schedule.model_dump()
        
        # Try to serialize to JSON
        json_str = json.dumps(schedule_dict, indent=2, default=str)
        
        print(f"  ✓ Schedule exported to JSON ({len(json_str)} chars)")
        print(f"    - Contains {len(schedule_dict['trainsets'])} trainsets")
        
        return True
    except Exception as e:
        print(f"  ✗ JSON export failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 70)
    print("  METRO SCHEDULING SYSTEM - VERIFICATION TESTS")
    print("=" * 70)
    
    tests = [
        ("Imports", test_imports),
        ("Data Generation", test_data_generation),
        ("Schedule Optimization", test_schedule_optimization),
        ("Data Models", test_models),
        ("JSON Export", test_json_export)
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("  TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print("\n" + "-" * 70)
    print(f"  Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  🎉 All tests passed! System is ready to use.")
        print("\n  Next steps:")
        print("    1. Run: python demo_schedule.py")
        print("    2. Run: python run_api.py")
        print("    3. Visit: http://localhost:8000/docs")
    else:
        print("\n  ⚠️  Some tests failed. Please check the errors above.")
        print("     Make sure all dependencies are installed:")
        print("     pip install -r requirements.txt")
    
    print("=" * 70)
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
