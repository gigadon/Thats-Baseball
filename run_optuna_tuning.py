"""Run Optuna hyperparameter tuning for the ensemble model."""
import logging
import shutil
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

from mlb.models.pipeline import TrainingPipeline

pipeline = TrainingPipeline()
X, y, dates, sample_weight = pipeline.load_training_data()

print(f"Loaded {len(X)} samples, {X.shape[1]} features")
print(f"Home win rate: {y.mean():.3f}")

ensemble = pipeline.tune_and_train(
    X, y,
    game_dates=dates,
    sample_weight=sample_weight,
    n_trials=50,
)

# Copy tuned models to win_model/
src = Path("models")
dst = Path("win_model")
if dst.exists():
    shutil.rmtree(dst)
shutil.copytree(src, dst)
print(f"\nTuned models copied to {dst}")
print(f"\nFinal metrics: {pipeline.run_metrics}")
