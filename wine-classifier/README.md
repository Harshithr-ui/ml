# Wine Classifier

A beginner-friendly **end-to-end Machine Learning web application** that classifies a
bottle of wine into one of three Italian wine cultivars based on thirteen chemical
measurements.

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

A user opens the web page, enters thirteen chemical measurements of a wine, clicks
**Predict Cultivar**, and the trained machine learning model returns the predicted wine
cultivar:

- Cultivar 1
- Cultivar 2
- Cultivar 3

---

## Dataset

| Item | Details |
|------|---------|
| **Name** | Wine (UCI ML Repository – Wine dataset) |
| **Original source** | [UCI Machine Learning Repository – Wine](https://archive.ics.uci.edu/dataset/109/wine) |
| **File** | `data/wine.csv` |
| **Size** | 178 rows × 14 columns (13 used for prediction) |
| **Missing values** | None |
| **License** | Public Domain / No license |

### Features (model input, 13 chemical properties)

| Column | Description | Unit / scale |
|--------|-------------|--------------|
| `alcohol` | Alcohol content | % by volume |
| `malic_acid` | Malic acid | g/L |
| `ash` | Ash content | g/100mL |
| `alcalinity_of_ash` | Alcalinity of ash | g/100mL |
| `magnesium` | Magnesium | mg/L |
| `total_phenols` | Total phenols | g/L |
| `flavanoids` | Flavanoids | g/L |
| `nonflavanoid_phenols` | Nonflavanoid phenols | g/L |
| `proanthocyanins` | Proanthocyanins | g/L |
| `color_intensity` | Color intensity | 0–20 scale |
| `hue` | Hue | 0–1 scale |
| `od280_od315_of_diluted_wines` | OD280/OD315 of diluted wines | 0–5 scale |
| `proline` | Proline | mg/L |

### Target (what we predict)

| Column | Values |
|--------|--------|
| `Cultivar` | `1`, `2`, `3` (3 Italian wine cultivars) |

### Class distribution

| Cultivar | Samples | Encoded as |
|----------|---------|------------|
| Cultivar 1 | 59 | `0` |
| Cultivar 2 | 71 | `1` |
| Cultivar 3 | 48 | `2` |
| **Total** | **178** | |

---

## Machine Learning model

| Item | Details |
|------|---------|
| **Algorithm** | Logistic Regression (`sklearn.linear_model.LogisticRegression`) |
| **Problem type** | Multi-class classification (3 cultivars) |
| **Train / Test split** | 80% / 20% (142 training, 36 testing) |
| **Accuracy** | **100%** |
| **Saved model** | `model.pkl` (created with `joblib.dump`) |

**Why Logistic Regression?** It is simple, fast, easy to explain, and performs very well
on this dataset. It learns linear boundaries that separate the three cultivars.

**Why save the model?** Training takes time. By saving the model once with Joblib, the
Flask app can load it instantly on every start instead of retraining. The model is
**not** retrained when the web app runs.

---

## Project structure

```text
wine-classifier/
│
├── app.py                  # Flask web server (routes: / and /predict)
├── train_model.py          # Trains the model from data/wine.csv and saves model.pkl
├── model.pkl               # The saved, already-trained model (generated)
├── requirements.txt        # The only libraries the project needs
├── README.md               # This file
├── .gitignore              # Files Git should ignore (venv, caches, etc.)
│
├── data/
│   └── wine.csv            # The dataset (178 wines × 14 columns)
│
├── templates/
│   └── index.html          # The web page (form + result area)
│
└── static/
    └── style.css           # The single stylesheet for the UI
```

| File | Purpose |
|------|---------|
| `data/wine.csv` | The dataset downloaded from the UCI Machine Learning Repository. |
| `train_model.py` | Loads the dataset, trains the model, prints accuracy, saves `model.pkl`. |
| `model.pkl` | Stores the trained model so it can be reused without retraining. |
| `app.py` | Flask server. Loads the model, shows the form, and returns predictions. |
| `templates/index.html` | The HTML form and the area where the result appears. |
| `static/style.css` | Styles the page (clean white background, purple accent, rounded card). |
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

This reads `data/wine.csv`, prints the accuracy, and creates `model.pkl`.

### Step 4 — Start the Flask application

```bash
python app.py
```

### Step 5 — Open the app

Visit **http://127.0.0.1:5000** in your browser.

> **Note:** `python train_model.py` and `python app.py` should be run from inside the
> `wine-classifier` folder, because the paths to the dataset and model are relative.

---

## How it works

```text
data/wine.csv
      ↓
Load with pandas
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
User enters 13 chemical measurements in the browser
      ↓
Form sends data to /predict using POST
      ↓
Flask validates input and builds the input row
      ↓
model.predict() returns class 0, 1 or 2
      ↓
Class is mapped to a cultivar name
      ↓
Result is rendered back into the page
```

**How the webpage talks to Flask:** The HTML form uses `method="POST"` and `action="/predict"`.
When the button is clicked, the browser sends the 13 values to the Flask route. Flask
reads them with `request.form`, validates them, calls the model, and returns the page with
the result.

---

## Results

- **Accuracy on the test set:** 100% (36 out of 36 wines classified correctly)
- The three cultivars are very well separated by their chemical profiles, which is why a
  simple linear model gets perfect accuracy.

### Example predictions

| Alcohol | Malic Acid | Ash | Proline | Predicted |
|:---:|:---:|:---:|:---:|:---|
| 14.23 | 1.71 | 2.43 | 1065 | Cultivar 1 |
| 13.20 | 1.78 | 2.14 | 1050 | Cultivar 1 |
| 13.16 | 2.36 | 2.28 | 1185 | Cultivar 3 |

---

## Error handling

The app never crashes because of bad input:

| Situation | Message shown |
|-----------|---------------|
| A field is left empty | *Please enter all thirteen values.* |
| A value is not numeric | *Please enter valid numerical values.* |

Both checks live in the `/predict` route in `app.py`.

---

## Limitations & future improvements

- Only one algorithm is used. Comparing **Decision Tree**, **Random Forest** or **KNN**
  would make the analysis stronger.
- The 13 chemical properties are not useful in the real world — a winemaker cannot
  measure `ash` or `flavanoids` in a cellar without lab equipment.
- Possible improvements:
  - Show the **prediction probability** (confidence) instead of just the class.
  - Add **input range validation** for realistic measurement values.
  - Plot a **confusion matrix** for the report.

---

## Author

**Student Name** — Machine Learning mini-project
Submitted for academic evaluation.
