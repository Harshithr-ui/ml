# Penguin Species Classifier

A beginner-friendly **end-to-end Machine Learning web application** that classifies a
Penguin into one of three species based on four measurements.

The project covers the full pipeline:

> **Dataset → Model → Saved Model → Flask Backend → HTML Form → Prediction**

---

## Table of Contents

1. [What the project does](#what-the-project-does)
2. [Dataset](#dataset)
3. [Machine Learning model](#machine-learning-model)
4. [Project structure](#project-structure)
5. [Technology stack](#technology-stack)
6. [How to run the project](#how-to-run-the-project)
7. [How it works](#how-it-works)
8. [Results](#results)
9. [Error handling](#error-handling)
10. [Limitations & future improvements](#limitations--future-improvements)
11. [Author](#author)

---

## What the project does

A user opens the web page, enters four penguin measurements (Bill Length, Bill Depth,
Flipper Length, Body Mass), clicks **Predict Species**, and the trained machine learning
model returns the predicted penguin species:

- Adelie
- Chinstrap
- Gentoo

---

## Dataset

| Item | Details |
|------|---------|
| **Name** | Palmers Penguins |
| **Original source** | [allisonhorst/palmerpenguins (GitHub)](https://github.com/allisonhorst/palmerpenguins) |
| **File** | `data/penguins.csv` |
| **Size** | 344 rows × 7 columns (4 used for prediction) |
| **Missing values** | Yes (11 rows removed during training) |
| **License** | CC0 – Public Domain |

### Features (model input, all numeric)

| Column | Description | Unit |
|--------|-------------|------|
| `bill_length_mm` | Bill length | mm |
| `bill_depth_mm` | Bill depth | mm |
| `flipper_length_mm` | Flipper length | mm |
| `body_mass_g` | Body mass | g |

### Target (what we predict)

| Column | Values |
|--------|--------|
| `species` | `Adelie`, `Chinstrap`, `Gentoo` |

### Class distribution after cleaning

| Species | Samples | Encoded as |
|---------|---------|------------|
| Adelie | 146 | `0` |
| Chinstrap | 68 | `1` |
| Gentoo | 119 | `2` |
| **Total** | **333** | |

---

## Machine Learning model

| Item | Details |
|------|---------|
| **Algorithm** | Logistic Regression (`sklearn.linear_model.LogisticRegression`) |
| **Problem type** | Multi-class classification (3 species) |
| **Train / Test split** | 80% / 20% (266 training, 67 testing) |
| **Accuracy** | **100%** |
| **Saved model** | `model.pkl` (created with `joblib.dump`) |

**Why Logistic Regression?** It is simple, fast, easy to explain, and performs very well
on this dataset. It learns linear boundaries that separate the three species.

**Why save the model?** Training takes time. By saving the model once with Joblib, the
Flask app can load it instantly on every start instead of retraining. The model is
**not** retrained when the web app runs.

---

## Project structure

```text
penguin-classifier/
│
├── app.py                  # Flask web server (routes: / and /predict)
├── train_model.py          # Trains the model from data/penguins.csv and saves model.pkl
├── model.pkl               # The saved, already-trained model (generated)
├── requirements.txt        # The only libraries the project needs
├── README.md               # This file
├── .gitignore              # Files Git should ignore (venv, caches, etc.)
│
├── data/
│   └── penguins.csv        # The downloaded dataset (344 rows)
│
├── templates/
│   └── index.html          # The web page (form + result area)
│
└── static/
    └── style.css           # The single stylesheet for the UI
```

| File | Purpose |
|------|---------|
| `data/penguins.csv` | The raw dataset downloaded from the GitHub `palmerpenguins` repository. |
| `train_model.py` | Loads the dataset, cleans missing rows, trains the model, prints accuracy, saves `model.pkl`. |
| `model.pkl` | Stores the trained model so it can be reused without retraining. |
| `app.py` | Flask server. Loads the model, shows the form, and returns predictions. |
| `templates/index.html` | The HTML form and the area where the result appears. |
| `static/style.css` | Styles the page (clean white background, blue accent, rounded card). |
| `requirements.txt` | Lists the dependencies so anyone can install them. |

---

## Technology stack

- Python 3
- Flask
- scikit-learn
- pandas
- NumPy
- Joblib

No JavaScript frameworks, no databases, no authentication, no APIs.

---

## How to run the project

### Step 1 — Create a virtual environment

```bash
python -m venv .venv
```

### Step 2 — Activate it and install dependencies

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Step 3 — Train the model

```bash
python train_model.py
```

This reads `data/penguins.csv`, prints the accuracy, and creates `model.pkl`.

### Step 4 — Start the Flask application

```bash
python app.py
```

### Step 5 — Open the app

Visit **http://127.0.0.1:5000** in your browser.

> **Note:** `python train_model.py` and `python app.py` should be run from inside the
> `penguin-classifier` folder, because the paths to the dataset and model are relative.

---

## How it works

```text
data/penguins.csv
      ↓
Load with pandas
      ↓
Remove rows with missing measurements
      ↓
Separate features (X) and target (y)
      ↓
Train / Test split (80 / 20)
      ↓
Train Logistic Regression
      ↓
Save model.pkl (Joblib)
      ↓
Flask loads model.pkl at startup
      ↓
User enters 4 measurements in the browser
      ↓
Form sends data to /predict using POST
      ↓
Flask validates input and builds the input row
      ↓
model.predict() returns class 0, 1 or 2
      ↓
Class is mapped to a species name
      ↓
Result is rendered back into the page
```

**How the webpage talks to Flask:** The HTML form uses `method="POST"` and `action="/predict"`.
When the button is clicked, the browser sends the four values to the Flask route. Flask
reads them with `request.form`, validates them, calls the model, and returns the page with
the result.

---

## Results

- **Accuracy on the test set:** 100% (67 out of 67 test penguins classified correctly)
- All three species are cleanly separated by these four measurements, which is a well-known
  property of the Palmer's Penguins dataset.

### Example predictions

| Bill Length | Bill Depth | Flipper Length | Body Mass | Predicted |
|:---:|:---:|:---:|:---:|:---|
| 39.1 | 18.7 | 181 | 3750 | Adelie |
| 46.5 | 17.9 | 197 | 3900 | Gentoo |
| 41.8 | 18.3 | 190 | 3770 | Chinstrap |

---

## Error handling

The app never crashes because of bad input:

| Situation | Message shown |
|-----------|---------------|
| A field is left empty | *Please enter all four values.* |
| A value is not numeric | *Please enter valid numerical values.* |

Both checks live in the `/predict` route in `app.py`.

---

## Limitations & future improvements

- The dataset is small (344 rows), so accuracy can vary if the split changes.
- Only one algorithm is used. Comparing **Decision Tree**, **Random Forest** or **KNN**
  would make the analysis stronger.
- Possible improvements:
  - Show the **prediction probability** (confidence) instead of just the class.
  - Add **input range validation** for realistic measurement values.
  - Add **island** and **sex** as extra features.
  - Plot a **confusion matrix** and pairplot for the report.

---

## Author

**Student Name** — Machine Learning mini-project
Submitted for academic evaluation.
