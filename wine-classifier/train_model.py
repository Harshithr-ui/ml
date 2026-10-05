"""
train_model.py
=====================================================================
Trains a Machine Learning classifier on the Wine dataset that we
downloaded and stored in data/wine.csv.

Pipeline (the same steps a real ML project follows):
    1.  Load the dataset from the CSV file.
    2.  Explore it briefly (shape, cultivars, missing values).
    3.  Separate the features (chemical properties) from the target (cultivar).
    4.  Split the data into a training set and a testing set.
    5.  Train a Logistic Regression classifier.
    6.  Measure the accuracy on the unseen testing set.
    7.  Save the trained model to model.pkl with Joblib.

Run this file ONCE (or whenever you change the data / model):
    python train_model.py
=====================================================================
"""

import pandas as pd                                      # load and explore the CSV
from sklearn.model_selection import train_test_split    # split data into train/test
from sklearn.linear_model import LogisticRegression     # our classifier
from sklearn.metrics import accuracy_score              # measure performance
import joblib                                           # save the trained model


# ----------------------------------------------------------------------
# Configuration (kept in one place so it is easy to change)
# ----------------------------------------------------------------------
DATA_FILE = "data/wine.csv"   # the dataset we downloaded
MODEL_FILE = "model.pkl"      # where the trained model will be saved

# The 13 chemical properties the model learns from.
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

# Maps the cultivar number to a name.
# We use an explicit mapping so the names NEVER change,
# even if the row order in the CSV changes.
#   1 -> Cultivar 1
#   2 -> Cultivar 2
#   3 -> Cultivar 3
CULTIVAR_MAP = {
    1: 0,
    2: 1,
    3: 2,
}

# Maps the class number (0, 1, 2) back to a cultivar name for display.
SPECIES_NAMES = {
    0: "Cultivar 1",
    1: "Cultivar 2",
    2: "Cultivar 3",
}


# ----------------------------------------------------------------------
# Step 1: Load the dataset
# ----------------------------------------------------------------------
df = pd.read_csv(DATA_FILE)

print("=" * 55)
print("1. DATASET LOADED")
print("=" * 55)
print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])
print()

print("First 5 rows:")
print(df.head())
print()

# ----------------------------------------------------------------------
# Step 2: Explore the dataset a little
# ----------------------------------------------------------------------
print("=" * 55)
print("2. QUICK EXPLORATION")
print("=" * 55)

# Value counts shows how many wines of each cultivar we have.
print("Wines per cultivar:")
print(df["Cultivar"].value_counts().sort_index())
print()

# Missing values would break training, so we always check them.
missing = df.isnull().sum().sum()
print("Total missing values:", missing)
if missing > 0:
    df = df.dropna()
    print("Missing values were removed.")
print()

# ----------------------------------------------------------------------
# Step 3: Separate features (X) and target (y)
# ----------------------------------------------------------------------
# X = the 13 chemical properties the model learns from.
X = df[FEATURE_COLUMNS]

# y = the answer we want to predict, converted from the cultivar number to a class number.
y = df["Cultivar"].map(CULTIVAR_MAP)

# Safety check: make sure every cultivar number matched our mapping.
if y.isnull().any():
    unknown = df.loc[y.isnull(), "Cultivar"].unique()
    raise ValueError(f"Unknown cultivar found in the data: {unknown}")


# ----------------------------------------------------------------------
# Step 4: Split into training and testing data
# ----------------------------------------------------------------------
# 80% for training, 20% for testing.
# random_state=42 makes the split reproducible (same result every run).
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("=" * 55)
print("3. TRAIN / TEST SPLIT")
print("=" * 55)
print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))
print()

# ----------------------------------------------------------------------
# Step 5: Create and train the model
# ----------------------------------------------------------------------
# Logistic Regression is simple, fast and very accurate on this data.
# max_iter=2000 gives the solver more attempts to converge on this
# 13-dimensional dataset.
model = LogisticRegression(max_iter=2000)

# .fit() is the actual learning step.
model.fit(X_train, y_train)


# ----------------------------------------------------------------------
# Step 6: Evaluate the model on unseen data
# ----------------------------------------------------------------------
predictions = model.predict(X_test)                 # predict the test wines
accuracy = accuracy_score(y_test, predictions)      # compare with real answers

print("=" * 55)
print("4. MODEL EVALUATION")
print("=" * 55)
print("Model Accuracy: {:.2f}%".format(accuracy * 100))
print()

# ----------------------------------------------------------------------
# Step 7: Save the trained model
# ----------------------------------------------------------------------
# Joblib writes the model object to a file so the Flask app can load it
# instantly, without retraining every time it starts.
joblib.dump(model, MODEL_FILE)

print("=" * 55)
print("Model saved as:", MODEL_FILE)
print("You can now start the web app with:  python app.py")
print("=" * 55)
