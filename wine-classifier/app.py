"""
app.py
-------

This is the Flask web application for the Wine Classifier.

What it does:
    1. Loads the already-trained model (model.pkl).
    2. Shows a home page with a form of thirteen input boxes.
    3. Reads the values the user typed.
    4. Sends those values to the model.
    5. Shows the predicted wine cultivar on the page.

Run this file with:
    python app.py
"""

import os
from flask import Flask, render_template, request
import joblib
import pandas as pd


# Create the Flask application object.
app = Flask(__name__)

# Load the trained model once when the app starts.
# This file is created by train_model.py.
model = joblib.load("model.pkl")

# Class number -> wine cultivar name mapping.
# The model gives us a number, but users want to see a name.
SPECIES_NAMES = {
    0: "Cultivar 1",
    1: "Cultivar 2",
    2: "Cultivar 3",
}

# The exact feature names the model was trained with.
# They must match the column names in data/wine.csv, otherwise
# scikit-learn will refuse to make a prediction.
FEATURE_COLUMNS = [
    "alcohol",
    "malic_acid",
    "ash",
    "alcalinity_of_ash",
    "magnesium",
    "total_phenols",
    "flavanoids",
    "nonflavanoid_phenols",
    "proanthocyanins",
    "color_intensity",
    "hue",
    "od280_od315_of_diluted_wines",
    "proline",
]

# The labels shown on the form (grouped so the HTML template can loop over them).
# order MUST match FEATURE_COLUMNS.
FORM_FIELDS = [
    ("alcohol", "Alcohol (%)", "13.0"),
    ("malic_acid", "Malic Acid (g/L)", "2.3"),
    ("ash", "Ash", "2.4"),
    ("alcalinity_of_ash", "Alcalinity of Ash", "19.5"),
    ("magnesium", "Magnesium (mg/L)", "99.7"),
    ("total_phenols", "Total Phenols", "2.3"),
    ("flavanoids", "Flavanoids", "2.0"),
    ("nonflavanoid_phenols", "Nonflavanoid Phenols", "0.4"),
    ("proanthocyanins", "Proanthocyanins", "1.6"),
    ("color_intensity", "Color Intensity", "5.1"),
    ("hue", "Hue", "0.96"),
    ("od280_od315_of_diluted_wines", "OD280/OD315 of Diluted Wines", "2.6"),
    ("proline", "Proline (mg/L)", "746.0"),
]


# ----------------------------------------------------------------------
# Home page route
# ----------------------------------------------------------------------
@app.route("/")
def home():
    # Show the form page. No prediction yet, so prediction is None.
    return render_template("index.html", fields=FORM_FIELDS, prediction=None, error=None)


# ----------------------------------------------------------------------
# Prediction route
# ----------------------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    # Read every field from the form into a list of values.
    values = []
    for field_name, _label, _placeholder in FORM_FIELDS:
        values.append(request.form.get(field_name, "").strip())

    # --- Error handling 1: empty fields ---
    if not all(values):
        return render_template(
            "index.html",
            fields=FORM_FIELDS,
            prediction=None,
            error="Please enter all thirteen values.",
        )

    # --- Error handling 2: non-numeric values ---
    # Try to convert the text into decimal numbers.
    # If the user typed letters, that raises a ValueError and we catch it.
    try:
        numbers = [float(v) for v in values]
    except ValueError:
        return render_template(
            "index.html",
            fields=FORM_FIELDS,
            prediction=None,
            error="Please enter valid numerical values.",
        )

    # Put the 13 values into a one-row table (DataFrame) using the
    # same column names as during training.
    input_data = pd.DataFrame([numbers], columns=FEATURE_COLUMNS)

    # Ask the model to predict the cultivar.
    prediction_number = model.predict(input_data)[0]

    # Convert the class number (0, 1, 2) into a readable name.
    cultivar_name = SPECIES_NAMES[prediction_number]

    # Show the result on the same page.
    return render_template(
        "index.html",
        fields=FORM_FIELDS,
        prediction=cultivar_name,
        error=None,
    )


# ----------------------------------------------------------------------
# Configuration (kept in one place so it is easy to change)
# ----------------------------------------------------------------------
PORT = int(os.environ.get("PORT", "5000"))


# Start the development server when we run "python app.py".
if __name__ == "__main__":
    app.run(debug=False, port=PORT)
