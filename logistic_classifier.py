# logistic_classifier.py

import pandas as pd
from sklearn.model_selection import train_test_split

print("--- Step 4: Loading and Inspecting Data (Kaggle CSV) ---")

# 1. Define the file name
DATASET_FILE_NAME = 'data.csv'  # <-- UPDATE THIS if your file is named differently

# 2. Load the dataset from the CSV
try:
    df = pd.read_csv(DATASET_FILE_NAME)
except FileNotFoundError:
    print(f"Error: The file '{DATASET_FILE_NAME}' was not found.")
    print("Please ensure you have extracted and placed the CSV file in the project folder.")
    exit()

# 3. Handle data cleanup (If necessary for the Kaggle version)
# The Kaggle breast cancer dataset often has an 'id' column and a final unnecessary column.
if 'id' in df.columns:
    df = df.drop(columns=['id'])
if df.columns[-1].startswith('Unnamed'): # Drop the final empty column if it exists
    df = df.iloc[:, :-1]

# 4. Separate Features (X) and Target (y)
# 'diagnosis' is the target column in the Kaggle dataset ('M'=Malignant, 'B'=Benign)
# We need to convert the text diagnosis (M/B) into numbers (1/0)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0}) 

y = df['diagnosis']
X = df.drop(columns=['diagnosis'])


print(f"Features (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")
print("Class Distribution (0=Benign, 1=Malignant):")
print(y.value_counts(normalize=True))

print("\nData loading complete. Proceeding to Step 5 (Split/Standardize)...")