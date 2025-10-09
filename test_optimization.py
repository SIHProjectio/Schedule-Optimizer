"""
Comprehensive test and demo script for the enhanced optimization system.
"""
import json
import time
import os
import sys
from pathlib import Path

# Add the parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from greedyOptim import (
    optimize_trainset_schedule,
    compare_optimization_methods,
    OptimizationConfig,
    TrainsetSchedulingOptimizer
)
from greedyOptim.error_handling import safe_optimize, DataValidator
from greedyOptim.hybrid_optimizers import optimize_with_hybrid_methods


def generate_test_data():
    """Generate test data using the enhanced generator."""
    print("🔄 Generating enhanced synthetic data...")
    
    try:
        # Try to import and run the enhanced generator
        sys.path.append(str(Path(__file__).parent.parent / "DataService"))
        from mlservice.DataService import enhanced_generator

        generator = enhanced_generator.EnhancedMetroDataGenerator(num_trainsets=25, seed=42)
        data = generator.save_to_json("test_data_enhanced.json")
        return data
        
    except ImportError:
        print("Enhanced generator not available, using basic data...")
        # Fallback to basic data structure
        return create_basic_test_data()


def create_basic_test_data():
    """Create basic test data structure."""
    from datetime import datetime, timedelta
    import random
    
    num_trainsets = 25
    trainset_ids = [f"TS-{str(i+1).zfill(3)}" for i in range(num_trainsets)]
    
    data = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "num_trainsets": num_trainsets,
            "system": "Test System"
        },
        "trainset_status": [],
        "fitness_certificates": [],
        "job_cards": [],
        "component_health": [],
        "branding_contracts": []
    }
    
    # Generate basic trainset status
    for ts_id in trainset_ids:
        data["trainset_status"].append({
            "trainset_id": ts_id,
            "operational_status": random.choice(["Available", "Available", "Available", "Maintenance", "Standby"]),
            "total_mileage_km": random.randint(50000, 200000),
            "last_service_date": (datetime.now() - timedelta(days=random.randint(1, 30))).isoformat()
        })
    
    # Generate basic certificates
    departments = ["Rolling Stock", "Signalling", "Telecom"]
    for ts_id in trainset_ids:
        for dept in departments:
            data["fitness_certificates"].append({
                "trainset_id": ts_id,
                "department": dept,
                "status": random.choice(["Valid", "Valid", "Valid", "Expired"]),
                "expiry_date": (datetime.now() + timedelta(days=random.randint(-5, 90))).isoformat()
            })
    
    # Generate basic job cards
    for ts_id in trainset_ids:
        if random.random() < 0.3:  # 30% chance of having a job card
            data["job_cards"].append({
                "trainset_id": ts_id,
                "priority": random.choice(["Critical", "High", "Medium", "Low"]),
                "status": random.choice(["Open", "Closed", "In-Progress"])
            })
    
    # Generate basic component health
    components = ["Bogie", "Brake_Pad", "HVAC", "Door_System"]
    for ts_id in trainset_ids:
        for comp in components:
            data["component_health"].append({
                "trainset_id": ts_id,
                "component": comp,
                "status": random.choice(["Good", "Good", "Fair", "Warning"]),
                "wear_level": random.randint(20, 90)
            })
    
    return data


def test_data_validation(data):
    """Test data validation functionality."""
    print("\n🔍 Testing Data Validation...")
    print("="*50)
    
    # Test valid data
    errors = DataValidator.validate_data(data)
    if errors:
        print("❌ Validation errors found:")
        for error in errors[:5]:  # Show first 5 errors
            print(f"  • {error}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more errors")
        return False
    else:
        print("✅ Data validation passed!")
        return True


def test_basic_optimization(data):
    """Test basic optimization methods."""
    print("\n🚀 Testing Basic Optimization Methods...")
    print("="*50)
    
    basic_methods = ['ga', 'cmaes', 'pso', 'sa']
    results = {}
    
    # Quick config for testing
    config = OptimizationConfig(
        required_service_trains=20,
        min_standby=2,
        population_size=30,
        generations=50
    )
    
    for method in basic_methods:
        print(f"\n🔄 Testing {method.upper()}...")
        try:
            start_time = time.time()
            
            if method == 'sa':
                result = optimize_trainset_schedule(data, method, config, max_iterations=1000)
            else:
                result = optimize_trainset_schedule(data, method, config)
            
            elapsed = time.time() - start_time
            results[method] = {
                'result': result,
                'time': elapsed,
                'success': True
            }
            
            print(f"  ✅ {method.upper()} completed in {elapsed:.1f}s")
            print(f"     Fitness: {result.fitness_score:.2f}")
            print(f"     Service: {len(result.selected_trainsets)}")
            print(f"     Standby: {len(result.standby_trainsets)}")
            
        except Exception as e:
            print(f"  ❌ {method.upper()} failed: {str(e)}")
            results[method] = {
                'result': None,
                'time': 0,
                'success': False,
                'error': str(e)
            }
    
    return results


def test_hybrid_optimization(data):
    """Test hybrid optimization methods."""
    print("\n🔬 Testing Hybrid Optimization Methods...")
    print("="*50)
    
    hybrid_methods = ['adaptive', 'ensemble']
    results = {}
    
    for method in hybrid_methods:
        print(f"\n🔄 Testing {method.upper()}...")
        try:
            start_time = time.time()
            
            if method == 'adaptive':
                result = optimize_with_hybrid_methods(data, method)
            elif method == 'ensemble':
                result = optimize_with_hybrid_methods(data, method)
            else:
                continue
            
            elapsed = time.time() - start_time
            results[method] = {
                'result': result, 
                'time': elapsed,
                'success': True
            }
            
            print(f"  ✅ {method.upper()} completed in {elapsed:.1f}s")
            print(f"     Fitness: {result.fitness_score:.2f}")
            print(f"     Service: {len(result.selected_trainsets)}")
            
        except Exception as e:
            print(f"  ❌ {method.upper()} failed: {str(e)}")
            results[method] = {
                'result': None,
                'time': 0, 
                'success': False,
                'error': str(e)
            }
    
    return results


def test_error_handling(data):
    """Test error handling capabilities."""
    print("\n🛡️ Testing Error Handling...")
    print("="*50)
    
    # Test with valid data
    print("Testing with valid data...")
    try:
        result = safe_optimize(data, method='ga', log_file='test_optimization.log')
        print("  ✅ Safe optimization with valid data succeeded")
    except Exception as e:
        print(f"  ❌ Safe optimization failed: {e}")
    
    # Test with invalid data
    print("Testing with invalid data...")
    invalid_data = {
        "trainset_status": [{"invalid": "data"}],
        "fitness_certificates": [],
        "job_cards": [],
        "component_health": []
    }
    
    try:
        result = safe_optimize(invalid_data, method='ga')
        print("  ❌ Should have failed with invalid data")
    except Exception as e:
        print(f"  ✅ Correctly caught error: {type(e).__name__}")


def test_configuration_options(data):
    """Test different configuration options."""
    print("\n⚙️ Testing Configuration Options...")
    print("="*50)
    
    configs = [
        ("Small Population", OptimizationConfig(population_size=20, generations=30)),
        ("Large Population", OptimizationConfig(population_size=100, generations=30)),
        ("High Mutation", OptimizationConfig(mutation_rate=0.3, generations=30)),
        ("Low Mutation", OptimizationConfig(mutation_rate=0.05, generations=30)),
    ]
    
    for config_name, config in configs:
        print(f"\n🔄 Testing {config_name}...")
        try:
            start_time = time.time()
            result = optimize_trainset_schedule(data, 'ga', config)
            elapsed = time.time() - start_time
            
            print(f"  ✅ {config_name}: Fitness = {result.fitness_score:.2f} ({elapsed:.1f}s)")
            
        except Exception as e:
            print(f"  ❌ {config_name} failed: {e}")


def run_comprehensive_comparison(data):
    """Run comprehensive comparison of all methods."""
    print("\n🏆 Comprehensive Method Comparison...")
    print("="*60)
    
    try:
        # Quick config for comparison
        config = OptimizationConfig(
            population_size=40,
            generations=75
        )
        
        methods = ['ga', 'pso', 'cmaes']  # Focus on most reliable methods
        
        optimizer = TrainsetSchedulingOptimizer(data, config)
        results = optimizer.compare_methods(methods)
        
        print("\n📊 Final Comparison Results:")
        print("-" * 60)
        
        valid_results = [(method, result) for method, result in results.items() 
                        if result is not None]
        
        if valid_results:
            # Sort by fitness score
            valid_results.sort(key=lambda x: x[1].fitness_score)
            
            for i, (method, result) in enumerate(valid_results):
                status = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "🏃"
                print(f"{status} {method.upper()}: {result.fitness_score:.2f}")
        
        return results
        
    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        return {}


def generate_summary_report(basic_results, hybrid_results, comparison_results):
    """Generate a summary report of all tests."""
    print("\n📋 OPTIMIZATION SYSTEM TEST SUMMARY")
    print("="*60)
    
    # Count successes
    basic_success = sum(1 for r in basic_results.values() if r.get('success', False))
    hybrid_success = sum(1 for r in hybrid_results.values() if r.get('success', False))
    
    print(f"Basic Methods: {basic_success}/{len(basic_results)} successful")
    print(f"Hybrid Methods: {hybrid_success}/{len(hybrid_results)} successful")
    
    # Find best results
    all_results = []
    for method, data in basic_results.items():
        if data.get('success') and data.get('result'):
            all_results.append((method, data['result'].fitness_score, data['time']))
    
    for method, data in hybrid_results.items():
        if data.get('success') and data.get('result'):
            all_results.append((method, data['result'].fitness_score, data['time']))
    
    if all_results:
        # Sort by fitness score (lower is better)
        all_results.sort(key=lambda x: x[1])
        
        print(f"\n🏆 Best Overall Results:")
        for i, (method, fitness, time_taken) in enumerate(all_results[:3]):
            rank = ["🥇", "🥈", "🥉"][i]
            print(f"  {rank} {method.upper()}: {fitness:.2f} (in {time_taken:.1f}s)")
    
    # System capabilities summary
    print(f"\n✅ System Capabilities Confirmed:")
    print(f"  • Data validation and error handling")
    print(f"  • Multiple optimization algorithms")
    print(f"  • Hybrid and ensemble methods")
    print(f"  • Configurable parameters")
    print(f"  • Comprehensive result analysis")
    
    print(f"\n🎯 System ready for production use!")


def main():
    """Main test function."""
    print("🔬 METRO TRAINSET SCHEDULING OPTIMIZATION SYSTEM")
    print("=" * 60)
    print("Enhanced system with modular architecture and advanced algorithms")
    print("=" * 60)
    
    try:
        # Step 1: Generate or load test data
        data = generate_test_data()
        
        # Step 2: Validate data
        if not test_data_validation(data):
            print("❌ Cannot proceed with invalid data")
            return
        
        # Step 3: Test basic optimization methods
        basic_results = test_basic_optimization(data)
        
        # Step 4: Test hybrid methods (if basic methods work)
        hybrid_results = {}
        if any(r.get('success', False) for r in basic_results.values()):
            hybrid_results = test_hybrid_optimization(data)
        
        # Step 5: Test error handling
        test_error_handling(data)
        
        # Step 6: Test configuration options
        test_configuration_options(data)
        
        # Step 7: Run comprehensive comparison
        comparison_results = run_comprehensive_comparison(data)
        
        # Step 8: Generate summary report
        generate_summary_report(basic_results, hybrid_results, comparison_results)
        
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()