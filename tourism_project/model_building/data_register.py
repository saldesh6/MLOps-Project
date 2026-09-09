import pandas as pd
import sys
import os

RAW_PATH = "tourism.csv" if os.path.exists("tourism.csv") else "tourism_project/data/tourism.csv"

def register_data():
    print("--- Starting Automated Data Registration ---")
    try:
        df = pd.read_csv(RAW_PATH)
        print(f"Loaded source file successfully from path: {RAW_PATH}")
    except FileNotFoundError:
        print(f"Extraction Error: No database target located at {RAW_PATH}")
        sys.exit(1)
        
    expected_columns = [
        'CustomerID', 'ProdTaken', 'Age', 'TypeofContact', 'CityTier',
        'Occupation', 'Gender', 'NumberOfPersonVisiting', 'PreferredPropertyStar',
        'MaritalStatus', 'NumberOfTrips', 'Passport', 'OwnCar',
        'NumberOfChildrenVisiting', 'Designation', 'MonthlyIncome',
        'PitchSatisfactionScore', 'ProductPitched', 'NumberOfFollowups', 'DurationOfPitch'
    ]
    missing = [c for c in expected_columns if c not in df.columns]
    if missing:
        print(f"Data Schema Error: Missing expected structural vectors: {missing}")
        sys.exit(1)
    print(f"Schema structural parsing validated. Matrix dimensions: {df.shape}")
    sys.exit(0)

if __name__ == "__main__":
    register_data()
