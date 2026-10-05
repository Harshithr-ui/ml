"""
app.py
-------
This is the Flask web application.

What it does:
    1. Loads the already-trained model (model.pkl).
    2. Shows a home page with a form of four input boxes.
    3. Reads the values the user typed.
    4. Sends those values to the model.
    5. Shows the predicted flower name on the page.

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

# Class number -> flower name mapping.
# The model gives us a number, but users want to see a name.
FLOWER_NAMES = {
    0: "Iris Setosa",
    1: "Iris Versicolor",
    2: "Iris Virginica",
}

# The exact feature names the model was trained with.
# They must match the column names in data/Iris.csv, otherwise
# scikit-learn will refuse to make a prediction.
FEATURE_COLUMNS = [
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm",
]


# ----------------------------------------------------------------------
# Home page route
# ----------------------------------------------------------------------
@app.route("/")
def home():
    # Show the form page. No prediction yet, so prediction is None.
    return render_template("index.html", prediction=None, error=None)


# ----------------------------------------------------------------------
# Prediction route
# ----------------------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    # Get what the user typed in each of the four boxes.
    # .strip() removes extra spaces.
    sepal_length = request.form.get("sepal_length", "").strip()
    sepal_width = request.form.get("sepal_width", "").strip()
    petal_length = request.form.get("petal_length", "").strip()
    petal_width = request.form.get("petal_width", "").strip()

    # --- Error handling 1: empty fields ---
    if not sepal_length or not sepal_width or not petal_length or not petal_width:
        return render_template(
            "index.html",
            prediction=None,
            error="Please enter all four values.",
        )

    # --- Error handling 2: non-numeric values ---
    # Try to convert the text into decimal numbers.
    # If the user typed letters, that raises a ValueError and we catch it.
    try:
        values = [
            float(sepal_length),
            float(sepal_width),
            float(petal_length),
            float(petal_width),
        ]
    except ValueError:
        return render_template(
            "index.html",
            prediction=None,
            error="Please enter valid numerical values.",
        )

    # Put the four values into a one-row table (DataFrame) using the
    # same column names as during training.
    input_data = pd.DataFrame([values], columns=FEATURE_COLUMNS)

    # Ask the model to predict the flower class.
    prediction_number = model.predict(input_data)[0]

    # Convert the class number (0, 1, 2) into a readable name.
    flower_name = FLOWER_NAMES[prediction_number]

    # Show the result on the same page.
    return render_template("index.html", prediction=flower_name, error=None)


# Start the development server when we run "python app.py".
if __name__ == "__main__":
    app.run(debug=True)
