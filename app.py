# app.py
import os
import json
from flask import Flask, render_template, request, jsonify
import pandas as pd
from joblib import load

DATA_PATH = "india_housing_prices.csv"
MODEL_PATH = os.path.join("model", "model.pkl")
META_PATH = os.path.join("model", "meta.json")

app = Flask(__name__)

# Load dataset for showing matching properties
df = pd.read_csv(DATA_PATH)

# Load model + metadata
model = load(MODEL_PATH)

with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

@app.route("/")
def index():
    states = meta["meta"]["states"]
    property_types = meta["meta"]["property_types"]
    return render_template("index.html", states=states, property_types=property_types)

@app.route("/cities")
def cities():
    state = request.args.get("state")
    cities = meta["city_map"].get(state, [])
    return jsonify({"cities": cities})
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Columns model was trained on
    model_columns = [
        "State", "City", "Property_Type",
        "BHK", "Size_in_SqFt", "Year_Built",
        "Furnished_Status", "Floor_No", "Total_Floors",
        "Parking_Space", "Security", "Facing", "Owner_Type"
    ]

    # Build input row
    input_row = {}
    for col in model_columns:
        input_row[col] = data.get(col) or data.get(col.lower())

    X = pd.DataFrame([input_row])

    # Find matching properties (State, City, Property_Type)
    matches = df.copy()
    for key in ["State", "City", "Property_Type"]:
        if input_row.get(key):
            matches = matches[matches[key] == input_row[key]]

    matches = matches.head(5).to_dict(orient="records")

    # Predict price
    prediction = None
    try:
        prediction = float(model.predict(X)[0])
    except Exception as e:
        prediction = None

    return jsonify({
        "prediction_lakhs": prediction,
        "matches": matches
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
