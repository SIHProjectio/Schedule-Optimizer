#!/usr/bin/env python3
"""Test script to verify block optimization is working properly."""
import json
import sys
sys.path.insert(0, '.')

from DataService.enhanced_generator import EnhancedMetroDataGenerator
from greedyOptim.scheduler import TrainsetSchedulingOptimizer
from greedyOptim.schedule_generator import ScheduleGenerator
from greedyOptim.models import OptimizationConfig

def test_block_optimization():
    """Test that optimizers are actually producing block assignments."""
    
    # Generate test data
    generator = EnhancedMetroDataGenerator()
    data = generator.generate_complete_enhanced_dataset()
    
    # Configure optimizer with block optimization enabled
    config = OptimizationConfig(
        required_service_trains=6,
        min_standby=2,
        optimize_block_assignment=True,
        iterations=5  # Fewer iterations for quick test
    )
    
    print("=" * 60)
    print("TESTING BLOCK OPTIMIZATION")
    print("=" * 60)
    
    optimizer = TrainsetSchedulingOptimizer(data, config)
    
    methods_to_test = ['ga', 'cmaes', 'pso', 'sa', 'nsga2']
    
    results = {}
    for method in methods_to_test:
        print(f"\n{'='*60}")
        print(f"Testing {method.upper()}")
        print("=" * 60)
        
        try:
            result = optimizer.optimize(method=method)
            
            # Check for block assignments
            has_blocks = bool(result.service_block_assignments)
            num_assigned = sum(len(blocks) for blocks in result.service_block_assignments.values()) if has_blocks else 0
            
            print(f"\n{method.upper()} Results:")
            print(f"  - Selected trainsets: {len(result.selected_trainsets)}")
            print(f"  - Has block assignments: {has_blocks}")
            print(f"  - Total blocks assigned: {num_assigned}")
            print(f"  - Fitness score: {result.fitness_score:.2f}")
            
            if has_blocks:
                print(f"  - Block assignments per trainset:")
                for ts_id, blocks in result.service_block_assignments.items():
                    print(f"      {ts_id}: {len(blocks)} blocks")
            
            # Generate schedule using the result
            schedule_gen = ScheduleGenerator(data, config)
            schedule = schedule_gen.generate_schedule(result, method=method, runtime_ms=100)
            
            print(f"\n  Generated Schedule:")
            print(f"    - Schedule ID: {schedule.schedule_id}")
            print(f"    - Trainsets in schedule: {len(schedule.trainsets)}")
            
            # Check service trainsets have blocks
            for trainset in schedule.trainsets:
                if trainset.status.value == "REVENUE_SERVICE":
                    block_count = len(trainset.service_blocks) if trainset.service_blocks else 0
                    total_km = trainset.daily_km_allocation
                    print(f"    - {trainset.trainset_id}: {block_count} blocks, {total_km} km")
            
            results[method] = {
                'success': True,
                'has_blocks': has_blocks,
                'num_blocks': num_assigned,
                'fitness': result.fitness_score
            }
            
        except Exception as e:
            print(f"ERROR with {method}: {e}")
            import traceback
            traceback.print_exc()
            results[method] = {'success': False, 'error': str(e)}
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for method, res in results.items():
        if res['success']:
            status = "✓ PASS" if res['has_blocks'] else "⚠ NO BLOCKS"
            print(f"{method.upper()}: {status} (blocks: {res['num_blocks']}, fitness: {res['fitness']:.2f})")
            if not res['has_blocks']:
                all_passed = False
        else:
            print(f"{method.upper()}: ✗ FAIL ({res['error']})")
            all_passed = False
    
    if all_passed:
        print("\n✓ All optimizers producing block assignments correctly!")
    else:
        print("\n⚠ Some issues detected - check above for details")
    
    return all_passed

if __name__ == "__main__":
    test_block_optimization()
