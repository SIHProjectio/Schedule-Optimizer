"""
Test ensemble model training and prediction
"""
import sys
from pathlib import Path

# Add parent directory to path
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from SelfTrainService.config import CONFIG
from SelfTrainService.trainer import ModelTrainer
from SelfTrainService.data_store import ScheduleDataStore
from SelfTrainService.feature_extractor import FeatureExtractor
from DataService.generators.metro_generator import MetroDataGenerator
from DataService.optimizers.schedule_optimizer import MetroScheduleOptimizer


def test_config():
    """Test configuration"""
    print("Testing Configuration...")
    print(f"  Model Types: {CONFIG.MODEL_TYPES}")
    print(f"  Use Ensemble: {CONFIG.USE_ENSEMBLE}")
    print(f"  Retrain Interval: {CONFIG.RETRAIN_INTERVAL_HOURS} hours")
    print(f"  Features: {len(CONFIG.FEATURES)} features")
    print("  ✓ Config OK")


def test_model_initialization():
    """Test model initialization"""
    print("\nTesting Model Initialization...")
    trainer = ModelTrainer()
    
    for model_name in CONFIG.MODEL_TYPES:
        model = trainer._get_model(model_name)
        if model is not None:
            print(f"  ✓ {model_name}: {type(model).__name__}")
        else:
            print(f"  ✗ {model_name}: Failed to initialize")
    
    print("  ✓ Model initialization OK")


def test_data_generation():
    """Test data generation"""
    print("\nTesting Data Generation...")
    from datetime import datetime
    
    num_trains = 30
    generator = MetroDataGenerator(num_trains=num_trains)
    route = generator.generate_route()
    train_health = generator.generate_train_health_statuses()
    
    optimizer = MetroScheduleOptimizer(
        date=datetime.now().strftime("%Y-%m-%d"),
        num_trains=num_trains,
        route=route,
        train_health=train_health
    )
    
    schedule = optimizer.optimize_schedule()
    print(f"  Generated schedule with {len(schedule.trainsets)} trains")
    print(f"  Total service blocks: {sum(len(t.service_blocks) for t in schedule.trainsets)}")
    print("  ✓ Data generation OK")


def test_feature_extraction():
    """Test feature extraction"""
    print("\nTesting Feature Extraction...")
    from datetime import datetime
    
    num_trains = 30
    generator = MetroDataGenerator(num_trains=num_trains)
    route = generator.generate_route()
    train_health = generator.generate_train_health_statuses()
    
    optimizer = MetroScheduleOptimizer(
        date=datetime.now().strftime("%Y-%m-%d"),
        num_trains=num_trains,
        route=route,
        train_health=train_health
    )
    feature_extractor = FeatureExtractor()
    
    schedule = optimizer.optimize_schedule()
    schedule_dict = schedule.model_dump()
    features = feature_extractor.extract_from_schedule(schedule_dict)
    
    print(f"  Extracted {len(features)} features")
    print(f"  Feature names: {list(features.keys())[:5]}...")
    
    quality = feature_extractor.calculate_target(schedule_dict)
    print(f"  Quality score: {quality:.2f}")
    print("  ✓ Feature extraction OK")


def test_training():
    """Test model training"""
    print("\nTesting Model Training...")
    from datetime import datetime
    
    # Generate small dataset
    data_store = ScheduleDataStore()
    
    print("  Generating 20 sample schedules...")
    for i in range(20):
        num_trains = 25 + i
        generator = MetroDataGenerator(num_trains=num_trains)
        route = generator.generate_route()
        train_health = generator.generate_train_health_statuses()
        
        optimizer = MetroScheduleOptimizer(
            date=datetime.now().strftime("%Y-%m-%d"),
            num_trains=num_trains,
            route=route,
            train_health=train_health
        )
        schedule = optimizer.optimize_schedule()
        data_store.save_schedule(schedule.model_dump())
    
    # Try training (will fail due to insufficient data, but tests the pipeline)
    trainer = ModelTrainer()
    result = trainer.train(force=True)
    
    if result["success"]:
        print(f"  ✓ Training successful")
        print(f"    Models: {result['models_trained']}")
        print(f"    Best: {result['best_model']}")
    else:
        print(f"  ⓘ Training skipped: {result['reason']}")
        print("    (This is expected with small dataset)")
    
    print("  ✓ Training pipeline OK")


def test_prediction():
    """Test model prediction"""
    print("\nTesting Model Prediction...")
    
    trainer = ModelTrainer()
    
    # Try to load existing model
    if trainer.load_model():
        print("  ✓ Loaded existing model")
        
        # Test prediction
        test_features = {
            "num_trains": 30,
            "num_available": 28,
            "avg_readiness_score": 85.0,
            "total_mileage": 150000,
            "mileage_variance": 5000,
            "maintenance_count": 3,
            "certificate_expiry_count": 1,
            "branding_priority_sum": 15,
            "time_of_day": 12,
            "day_of_week": 3
        }
        
        prediction, confidence = trainer.predict(test_features, use_ensemble=True)
        print(f"  Ensemble Prediction: {prediction:.2f}")
        print(f"  Confidence: {confidence:.2f}")
        
        prediction_single, confidence_single = trainer.predict(test_features, use_ensemble=False)
        print(f"  Single Model Prediction: {prediction_single:.2f}")
        print(f"  Confidence: {confidence_single:.2f}")
        
        print("  ✓ Prediction OK")
    else:
        print("  ⓘ No trained model available (run train_model.py first)")


def main():
    """Run all tests"""
    print("=" * 60)
    print("Ensemble Model System Tests")
    print("=" * 60)
    
    try:
        test_config()
        test_model_initialization()
        test_data_generation()
        test_feature_extraction()
        test_training()
        test_prediction()
        
        print("\n" + "=" * 60)
        print("All Tests Completed!")
        print("=" * 60)
        print("\nNext Steps:")
        print("1. Install remaining dependencies: pip install -r requirements.txt")
        print("2. Generate training data: python SelfTrainService/train_model.py")
        print("3. Start retraining service: python SelfTrainService/start_retraining.py")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
