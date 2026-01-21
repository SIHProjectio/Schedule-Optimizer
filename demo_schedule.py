"""
Demo script to test Metro Train Scheduling System
Generates sample schedules and displays key information
"""
import sys
import os
from datetime import datetime
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from DataService.generators.metro_generator import MetroDataGenerator
from DataService.optimizers.schedule_optimizer import MetroScheduleOptimizer
from DataService.core.models import ScheduleRequest


def print_section(title: str):
    """Print a section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demo_data_generation():
    """Demonstrate data generation capabilities"""
    print_section("DATA GENERATION DEMO")
    
    # Initialize generator
    generator = MetroDataGenerator(num_trains=30, num_stations=25)
    print(f"\n✓ Initialized generator for {len(generator.trainset_ids)} trains")
    
    # Generate route
    route = generator.generate_route()
    print(f"\n✓ Route: {route.name}")
    print(f"  - Total distance: {route.total_distance_km} km")
    print(f"  - Stations: {len(route.stations)}")
    print(f"  - First: {route.stations[0].name}")
    print(f"  - Last: {route.stations[-1].name}")
    
    # Generate train health
    health_statuses = generator.generate_train_health_statuses()
    fully_healthy = sum(1 for h in health_statuses if h.is_fully_healthy)
    partial = sum(1 for h in health_statuses if not h.is_fully_healthy and h.available_hours)
    unavailable = sum(1 for h in health_statuses if not h.is_fully_healthy and not h.available_hours)
    
    print(f"\n✓ Train Health Status:")
    print(f"  - Fully healthy: {fully_healthy} ({fully_healthy/len(health_statuses)*100:.1f}%)")
    print(f"  - Partially available: {partial} ({partial/len(health_statuses)*100:.1f}%)")
    print(f"  - Unavailable: {unavailable} ({unavailable/len(health_statuses)*100:.1f}%)")
    
    # Show sample train details
    sample_train = health_statuses[0]
    print(f"\n✓ Sample Train: {sample_train.trainset_id}")
    print(f"  - Healthy: {sample_train.is_fully_healthy}")
    print(f"  - Cumulative mileage: {sample_train.cumulative_mileage:,} km")
    print(f"  - Days since maintenance: {sample_train.days_since_maintenance}")
    print(f"  - Component health: {len(sample_train.component_health)} components monitored")
    
    return generator, route, health_statuses


def demo_schedule_optimization(generator, route, health_statuses):
    """Demonstrate schedule optimization"""
    print_section("SCHEDULE OPTIMIZATION DEMO")
    
    date = datetime.now().strftime("%Y-%m-%d")
    print(f"\n✓ Optimizing schedule for: {date}")
    
    # Initialize optimizer
    optimizer = MetroScheduleOptimizer(
        date=date,
        num_trains=30,
        route=route,
        train_health=health_statuses,
        depot_name="Muttom_Depot"
    )
    
    print(f"  - Operating hours: 5:00 AM - 11:00 PM")
    print(f"  - One-way trip time: {optimizer.one_way_time_minutes} minutes")
    print(f"  - Round trip time: {optimizer.round_trip_time_minutes} minutes")
    
    # Run optimization
    print("\n✓ Running optimization...")
    schedule = optimizer.optimize_schedule(
        min_service_trains=22,
        min_standby=3,
        max_daily_km=300
    )
    
    print(f"\n✓ Schedule Generated: {schedule.schedule_id}")
    print(f"  - Generated at: {schedule.generated_at}")
    print(f"  - Valid period: {schedule.valid_from} to {schedule.valid_until}")
    print(f"  - Depot: {schedule.depot}")
    
    return schedule


def display_schedule_summary(schedule):
    """Display comprehensive schedule summary"""
    print_section("SCHEDULE SUMMARY")
    
    # Fleet summary
    fs = schedule.fleet_summary
    print(f"\n📊 Fleet Status:")
    print(f"  - Total trainsets: {fs.total_trainsets}")
    print(f"  - Revenue service: {fs.revenue_service}")
    print(f"  - Standby: {fs.standby}")
    print(f"  - Maintenance: {fs.maintenance}")
    print(f"  - Cleaning: {fs.cleaning}")
    print(f"  - Availability: {fs.availability_percent}%")
    
    # Optimization metrics
    om = schedule.optimization_metrics
    print(f"\n📈 Optimization Metrics:")
    print(f"  - Total planned km: {om.total_planned_km:,} km")
    print(f"  - Avg readiness score: {om.avg_readiness_score:.2f}")
    print(f"  - Mileage variance: {om.mileage_variance_coefficient:.3f}")
    print(f"  - Branding SLA: {om.branding_sla_compliance:.1%}")
    print(f"  - Shunting movements: {om.shunting_movements_required}")
    print(f"  - Runtime: {om.optimization_runtime_ms} ms")
    
    # Conflicts and alerts
    if schedule.conflicts_and_alerts:
        print(f"\n⚠️  Alerts and Conflicts: {len(schedule.conflicts_and_alerts)}")
        for alert in schedule.conflicts_and_alerts[:5]:  # Show first 5
            print(f"  - [{alert.severity}] {alert.trainset_id}: {alert.message}")
    else:
        print(f"\n✓ No conflicts or alerts")
    
    # Decision rationale
    dr = schedule.decision_rationale
    print(f"\n🎯 Decision Rationale:")
    print(f"  - Algorithm version: {dr.algorithm_version}")
    print(f"  - Constraint violations: {dr.constraint_violations}")
    print(f"  - Objective weights:")
    for obj, weight in dr.objective_weights.items():
        print(f"    • {obj}: {weight:.0%}")


def display_train_details(schedule):
    """Display details for selected trains"""
    print_section("SAMPLE TRAIN DETAILS")
    
    # Show one train from each status category
    categories = {
        "REVENUE_SERVICE": None,
        "STANDBY": None,
        "MAINTENANCE": None,
        "CLEANING": None
    }
    
    for trainset in schedule.trainsets:
        status = trainset.status.value
        if status in categories and categories[status] is None:
            categories[status] = trainset
    
    for status, trainset in categories.items():
        if trainset is None:
            continue
            
        print(f"\n🚇 {trainset.trainset_id} - {status}")
        print(f"  - Readiness score: {trainset.readiness_score:.2f}")
        print(f"  - Cumulative km: {trainset.cumulative_km:,} km")
        print(f"  - Daily km allocation: {trainset.daily_km_allocation} km")
        
        if trainset.assigned_duty:
            print(f"  - Assigned duty: {trainset.assigned_duty}")
        
        if trainset.service_blocks:
            # Guard against non-iterable sentinel values (e.g., typing.Never) by
            # ensuring we have a real sequence before slicing/iterating.
            blocks = None
            if isinstance(trainset.service_blocks, (list, tuple)):
                blocks = trainset.service_blocks
            else:
                try:
                    blocks = list(trainset.service_blocks)
                except Exception:
                    blocks = []
            print(f"  - Service blocks: {len(blocks)}")
            for block in blocks[:2]:  # Show first 2
                print(f"    • {block.block_id}: {block.origin} → {block.destination}")
                print(f"      Depart: {block.departure_time}, Trips: {block.trip_count}, Est: {block.estimated_km} km")
        
        # Certificates
        certs = trainset.fitness_certificates
        print(f"  - Certificates:")
        print(f"    • Rolling Stock: {certs.rolling_stock.status.value}")
        print(f"    • Signalling: {certs.signalling.status.value}")
        print(f"    • Telecom: {certs.telecom.status.value}")
        
        # Job cards
        if trainset.job_cards.open > 0:
            print(f"  - Job cards: {trainset.job_cards.open} open")
            if trainset.job_cards.blocking:
                print(f"    • Blocking: {', '.join(trainset.job_cards.blocking)}")
        
        # Branding
        if trainset.branding and trainset.branding.advertiser != "NONE":
            print(f"  - Branding: {trainset.branding.advertiser}")
            print(f"    • Priority: {trainset.branding.exposure_priority}")
            print(f"    • Hours remaining: {trainset.branding.contract_hours_remaining}")
        
        if trainset.alerts:
            print(f"  - Alerts: {', '.join(trainset.alerts)}")


def save_schedule_json(schedule, filename="sample_schedule.json"):
    """Save schedule to JSON file"""
    print_section("SAVING SCHEDULE")
    
    schedule_dict = schedule.model_dump()
    
    with open(filename, 'w') as f:
        json.dump(schedule_dict, f, indent=2, default=str)
    
    print(f"\n✓ Schedule saved to: {filename}")
    print(f"  - Size: {os.path.getsize(filename) / 1024:.1f} KB")
    print(f"  - Trainsets: {len(schedule_dict['trainsets'])}")


def main():
    """Main demo function"""
    print("\n" + "🚇" * 35)
    print("  METRO TRAIN SCHEDULING SYSTEM - DEMO")
    print("🚇" * 35)
    
    try:
        # Step 1: Data generation
        generator, route, health_statuses = demo_data_generation()
        
        # Step 2: Schedule optimization
        schedule = demo_schedule_optimization(generator, route, health_statuses)
        
        # Step 3: Display results
        display_schedule_summary(schedule)
        display_train_details(schedule)
        
        # Step 4: Save to file
        save_schedule_json(schedule)
        
        print_section("DEMO COMPLETE")
        print("\n✓ All systems operational!")
        print("\nNext steps:")
        print("  1. Review sample_schedule.json for full schedule details")
        print("  2. Run 'python run_api.py' to start the FastAPI service")
        print("  3. Visit http://localhost:8000/docs for API documentation")
        print("  4. Test with: curl http://localhost:8000/api/v1/schedule/example")
        print()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
