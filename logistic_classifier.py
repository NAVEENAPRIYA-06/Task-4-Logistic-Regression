# logistic_classifier.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("--- Data Loading and Preprocessing ---")

# 1. Define the file name and load the dataset from the CSV
DATASET_FILE_NAME = 'data.csv' 
try:
    df = pd.read_csv(DATASET_FILE_NAME)
except FileNotFoundError:
    print(f"Error: The file '{DATASET_FILE_NAME}' was not found.")
    exit()

# 2. Data Cleanup and Encoding
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

# 3. Split data into training and testing sets (70/30 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("\n--- Data Splitting ---")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# 4. Feature Standardization
# Scale the data to have a mean of 0 and standard deviation of 1.
# This prevents features with larger values from dominating the model.
scaler = StandardScaler()

# Fit scaler only on the training data and transform both training and test data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert scaled arrays back to DataFrames for consistency (optional)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

print("\n--- Standardization Complete. Data is ready for modeling. ---")

# --- Variables ready for model training: X_train_scaled, X_test_scaled, y_train, y_test ---