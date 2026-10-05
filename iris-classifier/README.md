# Iris Flower Classifier

A beginner-friendly **end-to-end Machine Learning web application** that classifies an
Iris flower into one of three species based on four measurements.

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

A user opens the web page, enters four flower measurements (Sepal Length, Sepal Width,
Petal Length, Petal Width), clicks **Predict Flower**, and the trained machine learning
model returns the predicted species:

- Iris Setosa
- Iris Versicolor
- Iris Virginica

---

## Dataset

| Item | Details |
|------|---------|
| **Name** | Iris Flower Dataset (also known as "Iris Species" / Fisher's Iris dataset) |
| **Original source** | [UCI Machine Learning Repository – Iris](https://archive.ics.uci.edu/dataset/53/iris) |
| **Mirror** | [Kaggle – Iris Species (uciml/iris)](https://www.kaggle.com/datasets/uciml/iris) — same data, identical 150 rows |
| **File** | `data/Iris.csv` |
| **Size** | 150 samples × 6 columns |
| **Missing values** | None |
| **License** | CC0 – Public Domain (UCI) |

### Features (model input)

| Column | Description | Unit |
|--------|-------------|------|
| `SepalLengthCm` | Sepal length | cm |
| `SepalWidthCm` | Sepal width | cm |
| `PetalLengthCm` | Petal length | cm |
| `PetalWidthCm` | Petal width | cm |

### Target (what we predict)

| Column | Values |
|--------|--------|
| `Species` | `Iris-setosa`, `Iris-versicolor`, `Iris-virginica` |

### Class distribution (perfectly balanced)

| Species | Samples | Encoded as |
|---------|---------|------------|
| Iris-setosa | 50 | `0` |
| Iris-versicolor | 50 | `1` |
| Iris-virginica | 50 | `2` |
| **Total** | **150** | |

The `Id` column is a simple row number and is **not** used for training.

---

## Machine Learning model

| Item | Details |
|------|---------|
| **Algorithm** | Logistic Regression (`sklearn.linear_model.LogisticRegression`) |
| **Problem type** | Multi-class classification (3 classes) |
| **Train / Test split** | 80% / 20% (120 training, 30 testing) |
| **Accuracy** | **96.67%** |
| **Saved model** | `model.pkl` (created with `joblib.dump`) |

**Why Logistic Regression?**
It is simple, fast, easy to explain, and performs very well on this dataset. It learns
linear boundaries that separate the three species.

**Why save the model?**
Training takes time. By saving the model once with Joblib, the Flask app can load it
instantly on every start instead of retraining. The model is **not** retrained when the
web app runs.

---

## Project structure

```text
iris-classifier/
│
├── app.py                  # Flask web server (routes: / and /predict)
├── train_model.py          # Trains the model from data/Iris.csv and saves model.pkl
├── model.pkl               # The saved, already-trained model (generated)
├── requirements.txt        # The only libraries the project needs
├── README.md               # This file
├── .gitignore              # Files Git should ignore (venv, caches, etc.)
│
├── data/
│   └── Iris.csv            # The downloaded dataset (150 rows)
│
├── templates/
│   └── index.html          # The web page (form + result area)
│
└── static/
    └── style.css           # The single stylesheet for the UI
```

| File | Purpose |
|------|---------|
| `data/Iris.csv` | The raw dataset downloaded from Kaggle / UCI. |
| `train_model.py` | Loads the dataset, trains the model, prints accuracy, saves `model.pkl`. |
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
- HTML5 & CSS3

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

This reads `data/Iris.csv`, prints the accuracy, and creates `model.pkl`.

### Step 4 — Start the Flask application

```bash
python app.py
```

### Step 5 — Open the app

Visit **http://127.0.0.1:5000** in your browser.

> **Note:** `python train_model.py` and `python app.py` should be run from inside the
> `iris-classifier` folder, because the paths to the dataset and model are relative.

---

## How it works

```text
data/Iris.csv
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
User enters 4 measurements in the browser
      ↓
Form sends data to /predict using POST
      ↓
Flask validates input and builds the input row
      ↓
model.predict() returns class 0, 1 or 2
      ↓
Class is mapped to a flower name
      ↓
Result is rendered back into the page
```

**How the webpage talks to Flask:**
The HTML form uses `method="POST"` and `action="/predict"`. When the button is clicked,
the browser sends the four values to the Flask route. Flask reads them with
`request.form`, validates them, calls the model, and returns the page with the result.

---

## Results

- **Accuracy on the test set:** 96.67% (29 out of 30 flowers classified correctly)
- The single misclassification is a versicolor/virginica boundary case — these two species
  overlap slightly in petal size, which is a well-known property of this dataset.

### Example predictions

| Sepal Length | Sepal Width | Petal Length | Petal Width | Predicted |
|:---:|:---:|:---:|:---:|:---|
| 5.1 | 3.5 | 1.4 | 0.2 | Iris Setosa |
| 5.9 | 3.0 | 4.2 | 1.5 | Iris Versicolor |
| 6.5 | 3.0 | 5.2 | 2.0 | Iris Virginica |

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

- The dataset is small (150 rows), so accuracy can vary slightly if the split changes.
- Only one algorithm is used. Comparing **Decision Tree**, **Random Forest** or **KNN**
  would make the analysis stronger.
- Possible improvements:
  - Show the **prediction probability** (confidence) instead of just the class.
  - Add **input range validation** for realistic measurement values.
  - Plot a **confusion matrix** and **pairplot** for the report.
  - Support **uploading a CSV** to predict many flowers at once.

---

## Author

**Student Name** — Machine Learning mini-project
Submitted for academic evaluation.
