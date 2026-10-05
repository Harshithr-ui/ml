"""
train_model.py
=====================================================================
Trains a Machine Learning classifier on the Iris flower dataset that
we downloaded and stored in data/Iris.csv.

Pipeline (the same steps a real ML project follows):
    1.  Load the dataset from the CSV file.
    2.  Explore it briefly (shape, classes, missing values).
    3.  Separate the features (measurements) from the target (species).
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
DATA_FILE = "data/Iris.csv"   # the dataset we downloaded
MODEL_FILE = "model.pkl"      # where the trained model will be saved

# The four measurement columns used as model input.
FEATURE_COLUMNS = [
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm",
]

# Maps the text species name to a number.
# We use an explicit mapping so the numbers NEVER change,
# even if the row order in the CSV changes.
#   0 -> Iris-setosa
#   1 -> Iris-versicolor
#   2 -> Iris-virginica
SPECIES_MAP = {
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2,
}


# ----------------------------------------------------------------------
# Step 1: Load the dataset
# ----------------------------------------------------------------------
df = pd.read_csv(DATA_FILE)

print("=" * 55)
print("1. DATASET LOADED")
print("=" * 55)
print("Rows    :", df.shape[0])   # 150 flowers
print("Columns :", df.shape[1])   # Id + 4 measurements + Species
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

# Value counts shows how many flowers of each species we have.
print("Flowers per species:")
print(df["Species"].value_counts())
print()

# Missing values would break training, so we always check them.
missing = df.isnull().sum().sum()
print("Total missing values:", missing)
if missing > 0:
    # Iris is a clean dataset, so this should never happen.
    df = df.dropna()
    print("Missing values were removed.")
print()


# ----------------------------------------------------------------------
# Step 3: Separate features (X) and target (y)
# ----------------------------------------------------------------------
# X = the 4 measurements the model learns from.
X = df[FEATURE_COLUMNS]

# y = the answer we want to predict, converted from text to numbers.
# .str.strip() removes any accidental spaces around the species name.
y = df["Species"].str.strip().map(SPECIES_MAP)

# Safety check: make sure every species name matched our mapping.
if y.isnull().any():
    unknown = df.loc[y.isnull(), "Species"].unique()
    raise ValueError(f"Unknown species found in the data: {unknown}")


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
# Logistic Regression is simple, fast and very accurate on Iris.
# max_iter=200 gives the solver enough attempts to converge.
model = LogisticRegression(max_iter=200)

# .fit() is the actual learning step.
model.fit(X_train, y_train)


# ----------------------------------------------------------------------
# Step 6: Evaluate the model on unseen data
# ----------------------------------------------------------------------
predictions = model.predict(X_test)                 # predict the test flowers
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
