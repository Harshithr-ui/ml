"""
app.py
-------

This is the Flask web application for the Penguin Species Classifier.

What it does:
    1. Loads the already-trained model (model.pkl).
    2. Shows a home page with a form of four input boxes.
    3. Reads the values the user typed.
    4. Sends those values to the model.
    5. Shows the predicted penguin species on the page.

Run this file with:
    python app.py
"""

from flask import Flask, render_template, request
import joblib
import pandas as pd


# Create the Flask application object.
app = Flask(__name__)

# Load the trained model once when the app starts.
# This file is created by train_model.py.
model = joblib.load("model.pkl")

# Class number -> penguin species name mapping.
# The model gives us a number, but users want to see a name.
SPECIES_NAMES = {
    0: "Adelie",
    1: "Chinstrap",
    2: "Gentoo",
}

# The exact feature names the model was trained with.
# They must match the column names in data/penguins.csv, otherwise
# scikit-learn will refuse to make a prediction.
FEATURE_COLUMNS = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
]

# The labels shown on the form (grouped so the HTML template can loop over them).
FORM_FIELDS = [
    ("bill_length_mm", "Bill Length (mm)", "e.g. 39.1"),
    ("bill_depth_mm", "Bill Depth (mm)", "e.g. 18.7"),
    ("flipper_length_mm", "Flipper Length (mm)", "e.g. 181"),
    ("body_mass_g", "Body Mass (g)", "e.g. 3750"),
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
            error="Please enter all four values.",
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

    # Put the four values into a one-row table (DataFrame) using the
    # same column names as during training.
    input_data = pd.DataFrame([numbers], columns=FEATURE_COLUMNS)

    # Ask the model to predict the penguin species.
    prediction_number = model.predict(input_data)[0]

    # Convert the class number (0, 1, 2) into a readable name.
    species_name = SPECIES_NAMES[prediction_number]

    # Show the result on the same page.
    return render_template(
        "index.html",
        fields=FORM_FIELDS,
        prediction=species_name,
        error=None,
    )


# Start the development server when we run "python app.py".
if __name__ == "__main__":
    app.run(debug=True)
