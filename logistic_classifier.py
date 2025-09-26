# logistic_classifier.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np

# --- Data Loading and Preprocessing ---

print("--- Data Loading and Preprocessing ---")

# Define the file name and load the dataset from the CSV
DATASET_FILE_NAME = 'data.csv' 
try:
    df = pd.read_csv(DATASET_FILE_NAME)
except FileNotFoundError:
    print(f"Error: The file '{DATASET_FILE_NAME}' was not found.")
    exit()

# Data Cleanup and Encoding
# Drop unnecessary columns like 'id' and the final 'Unnamed' column, if they exist.
if 'id' in df.columns:
    df = df.drop(columns=['id'])
if df.columns[-1].startswith('Unnamed'): 
    df = df.iloc[:, :-1]

# Convert the text diagnosis ('M'/'B') into numerical target (1/0)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0}) 

y = df['diagnosis']
X = df.drop(columns=['diagnosis'])

print(f"Features (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")
print("Class Distribution (0=Benign, 1=Malignant):")
print(y.value_counts(normalize=True))

# Split data into training and testing sets (70/30 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("\n--- Data Splitting ---")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# Feature Standardization
# Scale the data to have a mean of 0 and standard deviation of 1.
scaler = StandardScaler()

# Fit scaler only on the training data and transform both training and test data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert scaled arrays back to DataFrames for consistency (optional)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

print("\n--- Standardization Complete. Data is ready for modeling. ---")

# --- Model Training and Initial Evaluation ---

print("\n--- Model Training ---")

# Initialize and train the Logistic Regression model
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions (class labels) and probabilities on the test data
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] # Probability for the positive class (1: Malignant)

# Evaluation (Default Threshold: 0.5)
print("\n--- Model Evaluation (Default Threshold: 0.5) ---")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
print(f"\n[[TN (True Negative), FP (False Positive)],\n [FN (False Negative), TP (True Positive)]]")

# Classification Report (Precision, Recall, F1-Score)
report = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(report)

# ROC-AUC Score
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC AUC Score: {roc_auc:.4f}")


# Plot ROC Curve (Visualization)
def plot_roc_curve(y_test, y_pred_proba, roc_auc):
    """Plots the ROC curve and highlights the AUC score."""
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {roc_auc:.4f})', color='darkorange')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR) / Recall')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('roc_curve.png') 
    plt.close() 

plot_roc_curve(y_test, y_pred_proba, roc_auc)
print("\nROC Curve saved as 'roc_curve.png'.")

# --- Threshold Tuning and Sigmoid Function Explanation ---

# Threshold Tuning
# Tuning the threshold to prioritize Recall (minimizing False Negatives) is often preferred for medical tasks.

precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)

# Find the threshold that yields a high recall, e.g., 95% recall (0.95)
target_recall = 0.95
i = np.where(recall >= target_recall)[0][-1] # Index of the lowest threshold that meets target recall
optimal_threshold = thresholds[i]

# Predict using the new threshold
y_pred_tuned = (y_pred_proba >= optimal_threshold).astype(int)

# Evaluate with the tuned threshold
print(f"\n--- Evaluation with Tuned Threshold ({optimal_threshold:.4f}) ---")

print("New Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_tuned))

tuned_report = classification_report(y_test, y_pred_tuned)
print("\nNew Classification Report:")
print(tuned_report)

print(f"Tuning Goal: Achieved Recall for class 1 is approximately {recall[i]:.4f} (at threshold {optimal_threshold:.4f}).")


# Explanation of the Sigmoid Function
# The Sigmoid function converts the linear output (logit) into a probability (0 to 1).

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Example calculation using the model's coefficients and first test sample:
sample_logit = model.decision_function(X_test_scaled.iloc[[0]])
sample_probability = sigmoid(sample_logit)[0]

print("\n--- Sigmoid Function Explanation ---")
print(f"Sigmoid function: P(y=1) = 1 / (1 + e^-(wTx + b))")
print(f"A linear score (logit) for the first test sample: {sample_logit[0]:.4f}")
print(f"Sigmoid converts this logit to a probability: {sample_probability:.4f}")
print(f"This probability is compared to the threshold ({optimal_threshold:.4f}) for final classification.")

# --- End of logistic_classifier.py ---