import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Machine Learning Imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Setup input/output paths
INPUT_FILE = Path("outputs/m1_labels/ds3_labeled_windows.csv")
OUT_DIR = Path("outputs/m1_model")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# The exact features our labeler generated
FEATURES =[
    "mean_speed", "max_speed", "speed_std", 
    "hard_brake_cnt", "rapid_accel_cnt", 
    "mean_rpm", "max_rpm", "high_rpm_ratio", 
    "mean_throttle", "throttle_std", "max_throttle_delta", 
    "mean_load"
]

def load_and_split_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Cannot find {INPUT_FILE}. Run label engineering first!")
        
    df = pd.read_csv(INPUT_FILE)
    df[FEATURES] = df[FEATURES].fillna(0)
    
    print(f"Total labeled windows loaded: {len(df)}")
    print("Splitting based on experimental design:")
    
    train_df = df[df["source_file"] == "exp2_19drivers_1car_1route.csv"]
    test_df = df[df["source_file"] == "exp3_4drivers_1car_1route.csv"]
    
    if len(train_df) == 0 or len(test_df) == 0:
        print("\n[WARNING] Could not find exact exp2/exp3 filenames. Using 80/20 random split instead.")
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
        
    print(f" -> Training set : {len(train_df)} windows (19 Drivers)")
    print(f" -> Testing set  : {len(test_df)} windows (4 Unseen Drivers)")

    X_train = train_df[FEATURES]
    y_train = train_df["label"]
    X_test = test_df[FEATURES]
    y_test = test_df["label"]
    
    return X_train, X_test, y_train, y_test, df["label"].unique()

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred, labels=["SLOW", "NORMAL", "AGGRESSIVE"])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["SLOW", "NORMAL", "AGGRESSIVE"],
                yticklabels=["SLOW", "NORMAL", "AGGRESSIVE"])
    plt.title(f"Confusion Matrix: {model_name}\n(Test Set: 4 Unseen Drivers)")
    plt.ylabel('Actual Behavior')
    plt.xlabel('Predicted Behavior')
    plt.tight_layout()
    
    # Clean up name for saving (e.g., "Random Forest" -> "random_forest")
    safe_name = model_name.lower().replace(" ", "_")
    plt.savefig(OUT_DIR / f"cm_{safe_name}.png", dpi=150)
    plt.close()

def plot_feature_importance(model, features, model_name):
    # Not all models have feature importances (e.g., SVM, LogReg)
    if not hasattr(model, 'feature_importances_'):
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title(f"Feature Importance: {model_name}")
    plt.bar(range(len(features)), importances[indices], align="center", color="#4CAF50")
    plt.xticks(range(len(features)), [features[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    
    safe_name = model_name.lower().replace(" ", "_")
    plt.savefig(OUT_DIR / f"feat_imp_{safe_name}.png", dpi=150)
    plt.close()

def plot_model_comparison(results):
    names = list(results.keys())
    accuracies = [results[n]['accuracy'] for n in names]
    
    plt.figure(figsize=(10, 5))
    bars = plt.bar(names, accuracies, color=['#3498db', '#e74c3c', '#2ecc71', '#9b59b6'])
    plt.title("Model Accuracy Comparison (Unseen Test Data)")
    plt.ylabel("Accuracy Score")
    plt.ylim(0, 1.0)
    
    # Add text labels above bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval*100:.1f}%", ha='center', va='bottom')
        
    plt.tight_layout()
    plt.savefig(OUT_DIR / "model_comparison.png", dpi=150)
    plt.close()

def main():
    print("=" * 65)
    print(" M1 — Driver Behavior Model Training & Comparison")
    print("=" * 65)
    
    # 1. Load Data
    X_train, X_test, y_train, y_test, classes = load_and_split_data()
    
    # 2. FEATURE SCALING (Crucial for SVM and Logistic Regression)
    print("\nScaling features (Standardization)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Define the Models to Compete
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, class_weight="balanced", random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
        "Support Vector Machine": SVC(kernel='rbf', C=1.0, class_weight="balanced", random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    }
    
    results = {}
    best_accuracy = 0
    best_model_name = ""
    best_model = None

    # 4. Train and Evaluate each model
    print("\nStarting Model Training Competition...\n" + "-"*50)
    
    for name, model in models.items():
        print(f"Training [{name}]...")
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        
        # Save results
        results[name] = {
            'accuracy': acc,
            'report': classification_report(y_test, y_pred, target_names=["AGGRESSIVE", "NORMAL", "SLOW"])
        }
        
        # Visualizations for this specific model
        plot_confusion_matrix(y_test, y_pred, name)
        plot_feature_importance(model, FEATURES, name)
        
        print(f" -> Accuracy: {acc * 100:.2f}%")
        
        # Track the best model
        if acc > best_accuracy:
            best_accuracy = acc
            best_model_name = name
            best_model = model

    print("-" * 50)
    
    # 5. Output Comparison and Winner
    plot_model_comparison(results)
    
    print("\n" + "=" * 65)
    print(f" WINNER: {best_model_name.upper()} (Accuracy: {best_accuracy * 100:.2f}%)")
    print("=" * 65)
    print("\nDetailed Classification Report for Winner:")
    print(results[best_model_name]['report'])
    
    # 6. Save the BEST Model, the Features, and the SCALER
    model_path = OUT_DIR / "m1_best_model.pkl"
    scaler_path = OUT_DIR / "m1_scaler.pkl"
    features_path = OUT_DIR / "m1_features_list.pkl"
    
    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(FEATURES, features_path)
    
    print(f"\n[export] Saved best model to: {model_path}")
    print(f"[export] Saved scaler to:     {scaler_path}")
    print(f"[export] Saved features to:   {features_path}")
    print(f"[export] Look in '{OUT_DIR}' for comparison charts and confusion matrices.")
    print("=" * 65)

if __name__ == "__main__":
    main()