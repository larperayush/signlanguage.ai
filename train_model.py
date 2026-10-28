import os
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, "static_signs.csv")
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "asl_classifier.joblib")

def load_dataset():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Dataset not found at: {DATA_FILE}\nCollect some data first!")

    labels = []
    features = []

    with open(DATA_FILE, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            raise ValueError("Dataset CSV file is empty.")

        for row_idx, row in enumerate(reader, start=2):
            if not row:
                continue
            if len(row) != 64:
                print(f"[Warning] Skipping malformed row {row_idx}: expected 64 columns, got {len(row)}")
                continue
            
            label = row[0].strip().upper()
            try:
                feature_vals = [float(val) for val in row[1:]]
            except ValueError:
                print(f"[Warning] Skipping row {row_idx}: invalid numeric values.")
                continue

            labels.append(label)
            features.append(feature_vals)

    return np.array(features), np.array(labels)

def main():
    print("--- ASL Model Training ---")
    print(f"Loading dataset from: {DATA_FILE}")
    
    X, y = load_dataset()
    unique_labels = np.unique(y)
    
    print(f"Total samples loaded: {len(X)}")
    print(f"Labels found in dataset ({len(unique_labels)}): {list(unique_labels)}")

    if len(X) < 10:
        print("[Error] Not enough samples to train. Collect more data first.")
        return

    # Split dataset into training (80%) and testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")

    # Initialize and train Random Forest Classifier
    print("Training Random Forest Classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\n--- Evaluation Report ---")
    print(f"Test Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Save trained model
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"Trained model successfully saved to: {MODEL_PATH}")
    print("You can re-run this script anytime as you collect more letters or samples!")

if __name__ == "__main__":
    main()