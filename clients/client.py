import pandas as pd
import os

# Always resolve paths relative to the project root
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def split_in_two(frame):
    mid = (len(frame) + 1) // 2
    return [frame.iloc[:mid], frame.iloc[mid:]]


# Load ONLY the Train Pool
df = pd.read_csv("data/processed/train_data.csv")

TARGET_COL = "Water Quality"

# Simulate Non-IID behavior using Train Pool
# Logic:
# Industrial: High Hardness > 70th percentile
# Residential: pH centered approx (-1 to 1 after scaling)
# Agricultural: High Turbidity > 60th percentile

industrial = df[df["Hardness (mg L-1 )"] > df["Hardness (mg L-1 )"].quantile(0.7)]
residential = df[df["pH`"].between(-1.0, 1.0)]
agricultural = df[df["Turbidity (cm)"] > df["Turbidity (cm)"].quantile(0.6)]

# Split each regional dataset into 2 virtual sub-clients for Hierarchical FL
industrial_splits = split_in_two(industrial)
residential_splits = split_in_two(residential)
agricultural_splits = split_in_two(agricultural)

# Save splits
industrial_splits[0].to_csv("data/splits/client_industrial_1.csv", index=False)
industrial_splits[1].to_csv("data/splits/client_industrial_2.csv", index=False)

residential_splits[0].to_csv("data/splits/client_residential_1.csv", index=False)
residential_splits[1].to_csv("data/splits/client_residential_2.csv", index=False)

agricultural_splits[0].to_csv("data/splits/client_agricultural_1.csv", index=False)
agricultural_splits[1].to_csv("data/splits/client_agricultural_2.csv", index=False)

print("Non-IID client datasets created from Train Pool (Hierarchical splits):")
print("Industrial 1:", industrial_splits[0].shape, "| Industrial 2:", industrial_splits[1].shape)
print("Residential 1:", residential_splits[0].shape, "| Residential 2:", residential_splits[1].shape)
print("Agricultural 1:", agricultural_splits[0].shape, "| Agricultural 2:", agricultural_splits[1].shape)
