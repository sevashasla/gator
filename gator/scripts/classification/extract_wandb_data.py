import json
import argparse
from pathlib import Path

import wandb
import numpy as np

def to_serializable(obj):
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        try:
            json.dumps(obj)
            return obj
        except TypeError:
            return str(obj)

def extract_run(run) -> dict:
    """Extract all relevant data from a WandB run."""
    history = run.history()

    def clean(series):
        """Drop NaN and convert to list."""
        return series.dropna().tolist()

    def steps(series):
        """Return corresponding step indices."""
        return series.dropna().index.tolist()

    metrics = {}
    for col in history.columns:
        if col.startswith("_"):
            continue
        values = clean(history[col])
        if values:
            metrics[col] = {
                "values": values,
                "steps": steps(history[col]),
            }

    return {
        "id":       run.id,
        "name":     run.name,
        "project":  run.project,
        "state":    run.state,
        "config":   to_serializable(dict(run.config)),
        "summary":  to_serializable(dict(run.summary)),
        "metrics":  metrics,
        # convenient top-level fields for the web
        "val_top1": clean(history["val_top1"]) if "val_top1" in history.columns else [],
        "val_top5": clean(history["val_top5"]) if "val_top5" in history.columns else [],
        "train_loss": clean(history["train_loss"]) if "train_loss" in history.columns else [],
        "val_loss":   clean(history["val_loss"])   if "val_loss"   in history.columns else [],
        "lr":         clean(history["lr"])          if "lr"         in history.columns else [],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--entity",
        type=str,
        required=True,
        help="WandB username or team name",
    )
    parser.add_argument(
        "--projects",
        nargs="+",
        default=["gator-finetune", "mae-finetune", "jigsaw-finetune", "croco-gator-small-finetune"],
        help="WandB project names to extract",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Output directory for JSON files",
    )
    parser.add_argument(
        "--run_ids",
        nargs="+",
        default=None,
        help="Optional: specific run IDs to extract (all runs if not specified)",
    )
    args = parser.parse_args()

    api = wandb.Api()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_runs = []

    for project in args.projects:
        project_dir = output_dir / project
        project_dir.mkdir(exist_ok=True)

        try:
            runs = api.runs(f"{args.entity}/{project}")
        except Exception as e:
            print(f"Could not fetch project {project}: {e}")
            continue

        for run in runs:
            if args.run_ids and run.id not in args.run_ids:
                continue

            print(f"Extracting {project}/{run.name} ({run.id}) [{run.state}]...")

            try:
                data = extract_run(run)
            except Exception as e:
                print(f"  Error: {e}")
                continue

            # Save individual run file
            run_path = project_dir / f"{run.name}.json"
            with open(run_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"  Saved to {run_path}")

            all_runs.append(data)

    # Save combined file
    combined_path = output_dir / "all_runs.json"
    with open(combined_path, "w") as f:
        json.dump(all_runs, f, indent=2)
    print(f"\nAll runs saved to {combined_path} ({len(all_runs)} runs total)")

    # Print summary table
    print("\nSummary:")
    print(f"{'Model':<30} {'Run':<20} {'val_top1':<12} {'val_top5':<12} {'State'}")
    print("-" * 85)
    for run in all_runs:
        top1 = run["summary"].get("val_top1", "N/A")
        top5 = run["summary"].get("val_top5", "N/A")
        if isinstance(top1, float):
            top1 = f"{top1:.4f}"
        if isinstance(top5, float):
            top5 = f"{top5:.4f}"
        print(f"{run['project']:<30} {run['name']:<20} {top1:<12} {top5:<12} {run['state']}")

def combined_data_out(input_dir,output_path="all_runs.json"):

    data_dir = Path(input_dir)
    all_runs = []

    json_files = sorted(data_dir.rglob("*.json"))
    print(f"Found {len(json_files)} JSON files in {data_dir}")

    for json_file in json_files:
        if json_file.name == "all_runs.json":
            continue
        try:
            with open(json_file) as f:
                run = json.load(f)
            all_runs.append(run)
            print(f"  Loaded: {json_file}")
        except Exception as e:
            print(f"  Error loading {json_file}: {e}")
 
    output_path = Path(output_path)
    with open(output_path, "w") as f:
        json.dump(all_runs, f, indent=2)
 
    print(f"\nCombined {len(all_runs)} runs into {output_path}")



if __name__ == "__main__":
    #main()
    combined_data_out("data_out")

