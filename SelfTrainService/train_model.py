"""
Manually train the ensemble model
Run this to test model training or manually trigger retraining
"""
import sys
from pathlib import Path

# Add parent directory to path
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from SelfTrainService.trainer import ModelTrainer
from SelfTrainService.data_store import ScheduleDataStore
from DataService.generators.metro_generator import MetroDataGenerator
from DataService.optimizers.schedule_optimizer import MetroScheduleOptimizer
import json


def generate_sample_data(num_schedules: int = 150):
    """Generate sample schedule data for training"""
    print(f"Generating {num_schedules} sample schedules...")
    from datetime import datetime
    
    data_store = ScheduleDataStore()
    
    for i in range(num_schedules):
        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{num_schedules}")
        
        # Generate schedule with varying parameters
        num_trains = 25 + (i % 15)  # 25-40 trains
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
        
        # Save schedule
        data_store.save_schedule(schedule.model_dump())
    
    print(f"✓ Generated {num_schedules} schedules")


def main():
    """Train the ensemble model"""
    print("=" * 60)
    print("Multi-Model Ensemble Training")
    print("=" * 60)
    
    # Check if we have enough data
    data_store = ScheduleDataStore()
    count = data_store.count_schedules()
    
    print(f"\nCurrent data: {count} schedules")
    
    if count < 100:
        print(f"Need at least 100 schedules for training")
        generate_sample_data(150)
    
    # Initialize trainer
    print("\nInitializing model trainer...")
    trainer = ModelTrainer()
    
    # Train models
    print("\nTraining ensemble models...")
    print("Models: gradient_boosting, random_forest, xgboost, lightgbm, catboost")
    print()
    
    result = trainer.train(force=True)
    
    if result["success"]:
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"\nModels trained: {', '.join(result['models_trained'])}")
        print(f"Best model: {result['best_model']}")
        print(f"Samples used: {result['samples_used']}")
        print(f"\nEnsemble Weights:")
        for model, weight in result['ensemble_weights'].items():
            print(f"  {model}: {weight:.4f}")
        
        print(f"\nModel Performance:")
        for model, metrics in result['metrics'].items():
            print(f"\n{model}:")
            print(f"  Test R²: {metrics['test_r2']:.4f}")
            print(f"  Test RMSE: {metrics['test_rmse']:.4f}")
        
        # Save summary
        summary_path = Path("models/training_summary.json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        print(f"\n✓ Training summary saved to {summary_path}")
    else:
        print(f"\n✗ Training failed: {result.get('reason', result.get('error'))}")
    
    # Show model info
    print("\n" + "=" * 60)
    print("Current Model Info")
    print("=" * 60)
    info = trainer.get_model_info()
    print(json.dumps(info, indent=2, default=str))


if __name__ == "__main__":
    main()
