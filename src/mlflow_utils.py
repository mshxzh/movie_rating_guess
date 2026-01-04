"""
Utility functions for managing MLflow models and experiments.

Usage:
    python src/mlflow_utils.py list-runs
    python src/mlflow_utils.py list-models
    python src/mlflow_utils.py compare-runs
"""

import argparse
import mlflow
import pandas as pd
from config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME


def list_experiments():
    """List all MLflow experiments."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    experiments = mlflow.search_experiments()
    
    print(f"\n{'='*70}")
    print("MLFLOW EXPERIMENTS")
    print(f"{'='*70}\n")
    
    for exp in experiments:
        print(f"Name: {exp.name}")
        print(f"  ID: {exp.experiment_id}")
        print(f"  Artifact Location: {exp.artifact_location}")
        print()


def list_runs(experiment_name=None, top_n=10):
    """List recent MLflow runs."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    if experiment_name is None:
        experiment_name = MLFLOW_EXPERIMENT_NAME
    
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        order_by=["start_time DESC"],
        max_results=top_n
    )
    
    if len(runs) == 0:
        print(f"\n✗ No runs found in experiment: {experiment_name}")
        return
    
    print(f"\n{'='*70}")
    print(f"RECENT RUNS - {experiment_name}")
    print(f"{'='*70}\n")
    
    # Select relevant columns
    columns = ['run_id', 'start_time', 'params.pipeline_version', 
               'metrics.test_rmse', 'metrics.test_mae', 'metrics.test_r2']
    
    available_cols = [col for col in columns if col in runs.columns]
    
    display_df = runs[available_cols].head(top_n)
    display_df.columns = [col.replace('params.', '').replace('metrics.', '') for col in display_df.columns]
    
    print(display_df.to_string(index=False))
    print(f"\n{'='*70}")


def list_registered_models():
    """List registered models in MLflow Model Registry."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    client = mlflow.tracking.MlflowClient()
    
    try:
        models = client.search_registered_models()
        
        if len(models) == 0:
            print("\n✗ No registered models found")
            return
        
        print(f"\n{'='*70}")
        print("REGISTERED MODELS")
        print(f"{'='*70}\n")
        
        for model in models:
            print(f"Name: {model.name}")
            print(f"  Latest Version: {model.latest_versions[-1].version if model.latest_versions else 'N/A'}")
            
            for version in model.latest_versions:
                print(f"  Version {version.version}:")
                print(f"    Stage: {version.current_stage}")
                print(f"    Run ID: {version.run_id}")
            print()
    
    except Exception as e:
        print(f"\nError listing models: {e}")


def compare_runs(run_ids):
    """Compare multiple runs side by side."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    print(f"\n{'='*70}")
    print(f"COMPARING {len(run_ids)} RUNS")
    print(f"{'='*70}\n")
    
    comparison_data = []
    
    for run_id in run_ids:
        try:
            run = mlflow.get_run(run_id)
            
            comparison_data.append({
                'run_id': run_id[:8],  # First 8 chars
                'pipeline_version': run.data.params.get('pipeline_version', 'N/A'),
                'test_rmse': run.data.metrics.get('test_rmse', None),
                'test_mae': run.data.metrics.get('test_mae', None),
                'test_r2': run.data.metrics.get('test_r2', None),
                'cv_rmse': run.data.metrics.get('cv_rmse', None),
                'n_estimators': run.data.params.get('n_estimators', 'N/A'),
                'max_depth': run.data.params.get('max_depth', 'N/A'),
                'learning_rate': run.data.params.get('learning_rate', 'N/A')
            })
        except Exception as e:
            print(f"Error loading run {run_id}: {e}")
    
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        print(df.to_string(index=False))
        print(f"\n{'='*70}")
        
        # Highlight best
        best_rmse_idx = df['test_rmse'].idxmin()
        best_run = df.loc[best_rmse_idx]
        
        print(f"\nBest Run: {best_run['run_id']} (RMSE: {best_run['test_rmse']:.4f})")


def get_best_run(experiment_name=None, metric="test_rmse"):
    """Get the best run based on a metric."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    if experiment_name is None:
        experiment_name = MLFLOW_EXPERIMENT_NAME
    
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        order_by=[f"metrics.{metric} ASC"],
        max_results=1
    )
    
    if len(runs) == 0:
        print(f"\n✗ No runs found")
        return None
    
    best_run = runs.iloc[0]
    
    print(f"\n{'='*70}")
    print(f"BEST RUN (by {metric})")
    print(f"{'='*70}\n")
    
    print(f"Run ID: {best_run['run_id']}")
    print(f"Pipeline: {best_run.get('params.pipeline_version', 'N/A')}")
    print(f"Test RMSE: {best_run.get('metrics.test_rmse', 'N/A'):.4f}")
    print(f"Test MAE: {best_run.get('metrics.test_mae', 'N/A'):.4f}")
    print(f"Test R²: {best_run.get('metrics.test_r2', 'N/A'):.4f}")
    
    print(f"To use this model:")
    print(f"   python src/predict.py --run-id {best_run['run_id']}")
    
    return best_run['run_id']


def delete_run(run_id):
    """Delete a run from MLflow."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    client = mlflow.tracking.MlflowClient()
    
    try:
        client.delete_run(run_id)
        print(f"Deleted run: {run_id}")
    except Exception as e:
        print(f"Error deleting run: {e}")


def main():
    parser = argparse.ArgumentParser(description="MLflow utility commands")
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # List runs
    list_runs_parser = subparsers.add_parser('list-runs', help='List recent runs')
    list_runs_parser.add_argument('--experiment', type=str, help='Experiment name')
    list_runs_parser.add_argument('--top', type=int, default=10, help='Number of runs to show')
    
    # List models
    subparsers.add_parser('list-models', help='List registered models')
    
    # List experiments
    subparsers.add_parser('list-experiments', help='List all experiments')
    
    # Compare runs
    compare_parser = subparsers.add_parser('compare-runs', help='Compare multiple runs')
    compare_parser.add_argument('run_ids', nargs='+', help='Run IDs to compare')
    
    # Best run
    best_parser = subparsers.add_parser('best-run', help='Get best run')
    best_parser.add_argument('--experiment', type=str, help='Experiment name')
    best_parser.add_argument('--metric', type=str, default='test_rmse', help='Metric to optimize')
    
    # Delete run
    delete_parser = subparsers.add_parser('delete-run', help='Delete a run')
    delete_parser.add_argument('run_id', help='Run ID to delete')
    
    args = parser.parse_args()
    
    if args.command == 'list-runs':
        list_runs(args.experiment, args.top)
    elif args.command == 'list-models':
        list_registered_models()
    elif args.command == 'list-experiments':
        list_experiments()
    elif args.command == 'compare-runs':
        compare_runs(args.run_ids)
    elif args.command == 'best-run':
        get_best_run(args.experiment, args.metric)
    elif args.command == 'delete-run':
        delete_run(args.run_id)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

