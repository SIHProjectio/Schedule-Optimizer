"""
Quick Start Guide - Metro Train Scheduling System

This script shows the basic usage patterns for the Metro Train Scheduling System.
"""

from datetime import datetime
from DataService.generators.metro_generator import MetroDataGenerator
from DataService.optimizers.schedule_optimizer import MetroScheduleOptimizer
from DataService.core.models import ScheduleRequest


def example_1_basic_data_generation():
    """Example 1: Generate basic metro data"""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Basic Data Generation")
    print("=" * 60)
    
    # Create generator for 25 trains
    generator = MetroDataGenerator(num_trains=25, num_stations=25)
    
    # Generate route
    route = generator.generate_route("Aluva-Pettah Line")
    print(f"\nRoute: {route.name}")
    print(f"Distance: {route.total_distance_km} km")
    print(f"Stations: {len(route.stations)}")
    
    # Generate train health status
    health_statuses = generator.generate_train_health_statuses()
    print(f"\nGenerated health status for {len(health_statuses)} trains")
    
    # Count by category
    healthy = sum(1 for h in health_statuses if h.is_fully_healthy)
    print(f"  - Fully healthy: {healthy}")
    print(f"  - Need attention: {len(health_statuses) - healthy}")
    
    return generator, route, health_statuses


def example_2_simple_schedule():
    """Example 2: Generate a simple schedule"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Generate Simple Schedule")
    print("=" * 60)
    
    # Setup
    generator = MetroDataGenerator(num_trains=30)
    route = generator.generate_route()
    health_statuses = generator.generate_train_health_statuses()
    
    # Create optimizer
    optimizer = MetroScheduleOptimizer(
        date="2025-10-25",
        num_trains=30,
        route=route,
        train_health=health_statuses
    )
    
    # Generate schedule
    schedule = optimizer.optimize_schedule(
        min_service_trains=22,
        min_standby=3
    )
    
    print(f"\nSchedule ID: {schedule.schedule_id}")
    print(f"Valid: {schedule.valid_from} to {schedule.valid_until}")
    print(f"\nFleet Status:")
    print(f"  - In service: {schedule.fleet_summary.revenue_service}")
    print(f"  - Standby: {schedule.fleet_summary.standby}")
    print(f"  - Maintenance: {schedule.fleet_summary.maintenance}")
    print(f"  - Cleaning: {schedule.fleet_summary.cleaning}")
    
    return schedule


def example_3_detailed_schedule():
    """Example 3: Generate schedule with custom parameters"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Custom Schedule Parameters")
    print("=" * 60)
    
    generator = MetroDataGenerator(num_trains=35)
    route = generator.generate_route()
    health_statuses = generator.generate_train_health_statuses()
    
    optimizer = MetroScheduleOptimizer(
        date=datetime.now().strftime("%Y-%m-%d"),
        num_trains=35,
        route=route,
        train_health=health_statuses,
        depot_name="Custom_Depot"
    )
    
    # Custom optimization parameters
    schedule = optimizer.optimize_schedule(
        min_service_trains=25,  # More trains in service
        min_standby=5,          # More standby trains
        max_daily_km=280        # Lower km limit per train
    )
    
    print(f"\nSchedule optimized with custom parameters:")
    print(f"  - Total planned km: {schedule.optimization_metrics.total_planned_km:,}")
    print(f"  - Avg readiness: {schedule.optimization_metrics.avg_readiness_score:.2f}")
    print(f"  - Runtime: {schedule.optimization_metrics.optimization_runtime_ms} ms")
    
    return schedule


def example_4_train_details():
    """Example 4: Access detailed train information"""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Detailed Train Information")
    print("=" * 60)
    
    generator = MetroDataGenerator(num_trains=30)
    route = generator.generate_route()
    health_statuses = generator.generate_train_health_statuses()
    
    optimizer = MetroScheduleOptimizer(
        date="2025-10-25",
        num_trains=30,
        route=route,
        train_health=health_statuses
    )
    
    schedule = optimizer.optimize_schedule()
    
    # Find first train in revenue service
    service_train = next(
        (t for t in schedule.trainsets if t.status.value == "REVENUE_SERVICE"),
        None
    )
    
    if service_train:
        print(f"\nTrain: {service_train.trainset_id}")
        print(f"Status: {service_train.status.value}")
        print(f"Duty: {service_train.assigned_duty}")
        print(f"Daily km: {service_train.daily_km_allocation} km")
        print(f"Readiness: {service_train.readiness_score:.2f}")
        
        if service_train.service_blocks:
            print(f"\nService Blocks: {len(service_train.service_blocks)}")
            for i, block in enumerate(service_train.service_blocks[:3], 1):
                print(f"  {i}. {block.origin} → {block.destination}")
                print(f"     Depart: {block.departure_time}, Trips: {block.trip_count}")
        
        print(f"\nFitness Certificates:")
        certs = service_train.fitness_certificates
        print(f"  - Rolling Stock: {certs.rolling_stock.status.value}")
        print(f"  - Signalling: {certs.signalling.status.value}")
        print(f"  - Telecom: {certs.telecom.status.value}")
        
        if service_train.branding and service_train.branding.advertiser != "NONE":
            print(f"\nBranding:")
            print(f"  - Advertiser: {service_train.branding.advertiser}")
            print(f"  - Priority: {service_train.branding.exposure_priority}")


def example_5_schedule_request_model():
    """Example 5: Using ScheduleRequest model (for API)"""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Schedule Request Model")
    print("=" * 60)
    
    # Create a request (as would be done via API)
    request = ScheduleRequest(
        date="2025-10-25",
        num_trains=30,
        num_stations=25,
        route_name="Aluva-Pettah Line",
        depot_name="Muttom_Depot",
        min_service_trains=22,
        min_standby_trains=3,
        max_daily_km_per_train=300,
        balance_mileage=True,
        prioritize_branding=True
    )
    
    print(f"\nSchedule Request:")
    print(f"  - Date: {request.date}")
    print(f"  - Trains: {request.num_trains}")
    print(f"  - Stations: {request.num_stations}")
    print(f"  - Min service: {request.min_service_trains}")
    print(f"  - Max daily km: {request.max_daily_km_per_train}")
    
    # This request can be sent to the API:
    # POST /api/v1/generate with request.model_dump() as JSON
    
    return request


def example_6_save_schedule():
    """Example 6: Save schedule to JSON file"""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Save Schedule to File")
    print("=" * 60)
    
    import json
    
    generator = MetroDataGenerator(num_trains=25)
    route = generator.generate_route()
    health_statuses = generator.generate_train_health_statuses()
    
    optimizer = MetroScheduleOptimizer(
        date="2025-10-25",
        num_trains=25,
        route=route,
        train_health=health_statuses
    )
    
    schedule = optimizer.optimize_schedule()
    
    # Convert to dict and save
    schedule_dict = schedule.model_dump()
    
    filename = f"schedule_{schedule.schedule_id}.json"
    with open(filename, 'w') as f:
        json.dump(schedule_dict, f, indent=2, default=str)
    
    print(f"\nSchedule saved to: {filename}")
    print(f"Contains {len(schedule_dict['trainsets'])} trainsets")


def main():
    """Run all examples"""
    print("\n" + "🚇" * 30)
    print("  METRO TRAIN SCHEDULING - QUICK START EXAMPLES")
    print("🚇" * 30)
    
    try:
        # Run examples
        example_1_basic_data_generation()
        example_2_simple_schedule()
        example_3_detailed_schedule()
        example_4_train_details()
        example_5_schedule_request_model()
        example_6_save_schedule()
        
        print("\n" + "=" * 60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. Run 'python demo_schedule.py' for a comprehensive demo")
        print("  2. Run 'python run_api.py' to start the FastAPI service")
        print("  3. Visit http://localhost:8000/docs for API documentation")
        print()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
