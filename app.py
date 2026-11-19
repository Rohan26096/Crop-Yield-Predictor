from pathlib import Path
import json
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify
from src.data import load_dataset, FEATURES, NUMERIC_FEATURES

ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / "models" / "mlp_crop_yield.joblib"
META_FILE = ROOT / "models" / "metadata.json"
DATA_FILE = ROOT / "data" / "crop_yield.csv"
app = Flask(__name__)
history = []

def load_artifacts():
    if not DATA_FILE.exists():
        raise RuntimeError("Real dataset missing. Follow data/README.md and run python train.py.")
    if not MODEL_FILE.exists() or not META_FILE.exists():
        raise RuntimeError("Trained model missing. Run python train.py first.")
    return joblib.load(MODEL_FILE), json.loads(META_FILE.read_text())

model, metadata = load_artifacts()
dataset = load_dataset()

@app.route("/")
def dashboard():
    return render_template("index.html", metadata=metadata,
                           row_count=len(dataset),
                           crop_count=dataset["Crop"].nunique(),
                           state_count=dataset["State"].nunique())

@app.route("/predict", methods=["POST"])
def predict():
    try:
        p = request.get_json(force=True)
        row = {
            "Crop_Year": int(p["Crop_Year"]),
            "Annual_Rainfall": float(p["Annual_Rainfall"]),
            "Area": float(p["Area"]),
            "Pesticide": float(p["Pesticide"]),
            "Fertilizer": float(p["Fertilizer"]),
            "Crop": str(p["Crop"]),
            "Season": str(p["Season"]),
            "State": str(p["State"])
        }
        if row["Area"] <= 0 or row["Annual_Rainfall"] < 0:
            return jsonify({"error": "Area must be positive and rainfall cannot be negative."}), 400
        pred = max(0.0, float(model.predict(pd.DataFrame([row], columns=FEATURES))[0]))
        result = {
            "yield": round(pred, 2),
            "unit": "dataset yield unit (verify source unit)",
            "crop": row["Crop"], "state": row["State"], "season": row["Season"],
            "area": row["Area"]
        }
        history.insert(0, result)
        del history[10:]
        return jsonify(result)
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

@app.route("/api/overview")
def overview():
    crop_avg = dataset.groupby("Crop")["Yield"].mean().sort_values(ascending=False).head(10)
    year_avg = dataset.groupby("Crop_Year")["Yield"].mean().sort_index()
    return jsonify({
        "crop_labels": sorted(dataset["Crop"].dropna().unique().tolist()),
        "top_crop_labels": crop_avg.index.tolist(),
        "crop_values": [round(float(x), 2) for x in crop_avg.values],
        "year_labels": [str(int(x)) for x in year_avg.index],
        "year_values": [round(float(x), 2) for x in year_avg.values],
        "history": history,
        "state_labels": dataset.groupby("State")["Yield"].mean().nlargest(8).index.tolist(),
        "state_values": [round(float(x), 2) for x in dataset.groupby("State")["Yield"].mean().nlargest(8).values]
    })

@app.route("/api/model")
def model_info():
    return jsonify(metadata)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
