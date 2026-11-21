import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import joblib

# Always resolve paths relative to the project root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)

# Load dataset
df = pd.read_csv("data/raw/water_quality.csv")

print("Original shape:", df.shape)

# Drop duplicates
df = df.drop_duplicates()

# Handle missing values
df = df.fillna(df.mean(numeric_only=True))

# Identify target column
TARGET_COL = "Water Quality" # Auto-detected space

# Separate features and label
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

cleaned_df = pd.DataFrame(X_scaled, columns=X.columns)
cleaned_df[TARGET_COL] = y.values

# --- STEP 1: Split dataset ONCE at preprocessing stage ---
# 80% Train Pool (for Clients), 20% Global Test Set (Untouched)
train_df, test_df = train_test_split(cleaned_df, test_size=0.2, random_state=42, stratify=y)

# Save the fitted scaler so the dashboard can scale raw user input the same way
os.makedirs("saved_models", exist_ok=True)
joblib.dump(scaler, "saved_models/scaler.pkl")

# Save files
cleaned_df.to_csv("data/processed/cleaned_data.csv", index=False) # Legacy support
train_df.to_csv("data/processed/train_data.csv", index=False)
test_df.to_csv("data/processed/test_data_global.csv", index=False)

print(f"Data Processed.")
print(f"Train Pool (for Clients): {train_df.shape}")
print(f"Global Held-Out Test Set: {test_df.shape}")
