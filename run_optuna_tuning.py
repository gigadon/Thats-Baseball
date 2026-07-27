"""Optuna hyperparameter tuning for the win (moneyline) ensemble.

Trains to a staging directory (default models/win_model_new) so the freshly
tuned model can be backtested before it is promoted over the served
models/win_model. Promotion is a manual copy after the backtest looks good.

Usage:
    PYTHONPATH=src python3 run_optuna_tuning.py                 # → models/win_model_new
    PYTHONPATH=src python3 run_optuna_tuning.py --out models/win_model_new --trials 50
"""
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

from mlb.models.pipeline import PipelineConfig, TrainingPipeline

parser = argparse.ArgumentParser(description="Tune + train the win ensemble")
parser.add_argument("--out", default="models/win_model_new",
                    help="Output dir for the tuned model (staging; promote after backtest)")
parser.add_argument("--trials", type=int, default=50, help="Optuna trials per base model")
args = parser.parse_args()

pipeline = TrainingPipeline(PipelineConfig(model_dir=args.out))
X, y, dates, sample_weight = pipeline.load_training_data()

print(f"Loaded {len(X)} samples, {X.shape[1]} features")
print(f"Home win rate: {y.mean():.3f}")

pipeline.tune_and_train(
    X, y,
    game_dates=dates,
    sample_weight=sample_weight,
    n_trials=args.trials,
)

print(f"\nTuned model saved to {args.out}")
print(f"Train cutoff (OOS start): {pipeline.train_cutoff_}")
print(f"\nFinal metrics: {pipeline.run_metrics}")
