"""Train MLP configurations, select using validation MAE, then evaluate held-out test data."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data import load_dataset, FEATURES, TARGET
from src.model import build_pipeline, evaluate

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

EXPERIMENTS = [
    {"name": "MLP-32-16-ReLU", "layers": (32, 16), "activation": "relu", "lr": 0.001, "alpha": 0.0001},
    {"name": "MLP-64-32-ReLU", "layers": (64, 32), "activation": "relu", "lr": 0.001, "alpha": 0.0001},
    {"name": "MLP-100-50-ReLU", "layers": (100, 50), "activation": "relu", "lr": 0.001, "alpha": 0.0001},
    {"name": "MLP-64-32-Tanh", "layers": (64, 32), "activation": "tanh", "lr": 0.001, "alpha": 0.0001},
    {"name": "MLP-64-32-ReLU-LR0005", "layers": (64, 32), "activation": "relu", "lr": 0.0005, "alpha": 0.001}
]

def main():
    df = load_dataset()
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.20, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.20, random_state=42
    )

    results, fitted = [], []
    for cfg in EXPERIMENTS:
        print(f"Training {cfg['name']} ...")
        model = build_pipeline(cfg["layers"], cfg["activation"], cfg["lr"], cfg["alpha"])
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        val_metrics = evaluate(y_val, val_pred)
        results.append({
            "model": cfg["name"], "hidden_layers": str(cfg["layers"]),
            "activation": cfg["activation"], "learning_rate": cfg["lr"],
            "alpha": cfg["alpha"], **val_metrics,
            "iterations": int(model.named_steps["mlp"].n_iter_)
        })
        fitted.append(model)

    best_idx = min(range(len(results)), key=lambda i: results[i]["mae"])
    best_cfg = EXPERIMENTS[best_idx]
    # Refit selected configuration on all training data; test set remains untouched.
    best_model = build_pipeline(best_cfg["layers"], best_cfg["activation"],
                                best_cfg["lr"], best_cfg["alpha"])
    best_model.fit(X_train_full, y_train_full)
    test_metrics = evaluate(y_test, best_model.predict(X_test))
    joblib.dump(best_model, MODEL_DIR / "mlp_crop_yield.joblib")

    metadata = {
        "project": "Crop Yield Prediction Using MLP Regression",
        "dataset": "Mendeley Data: Stacked Ensemble Model for Accurate Crop Yield Prediction Using Machine Learning Techniques",
        "dataset_doi": "10.17632/ncw2vbcgnk.2",
        "dataset_rows_after_cleaning": int(len(df)),
        "train_rows": int(len(X_train_full)),
        "test_rows": int(len(X_test)),
        "features": FEATURES,
        "target": TARGET,
        "excluded_feature": "Production (target leakage: Yield is derived from Production / Area)",
        "best_model": results[best_idx]["model"],
        "best_parameters": best_cfg,
        "test_metrics": test_metrics,
        "experiments": results,
        "missing_values_before_pipeline": {
            c: int(df[c].isna().sum()) for c in FEATURES
        },
        "dataset_note": "Real third-party dataset; verify the downloaded file matches the cited Mendeley record."
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    pd.DataFrame(results).to_csv(REPORT_DIR / "experiment_results.csv", index=False)
    print("\\nSelected:", metadata["best_model"])
    print("Held-out test metrics:", json.dumps(test_metrics, indent=2))
    print("Saved model and experiment report.")

if __name__ == "__main__":
    main()
