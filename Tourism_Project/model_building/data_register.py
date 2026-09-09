import pandas as pd
import sys
import os

# Fallback path checking: Checks root first (GitHub Actions environment), then falls back to local Colab folder
if os.path.exists("tourism.csv"):
    RAW_PATH = "tourism.csv"
elif os.path.exists("Tourism_Project/tourism.csv"):
    RAW_PATH = "Tourism_Project/tourism.csv"
else:
    RAW_PATH = "tourism.csv" # Fallback default

def register_data():
    print("--- Starting Data Registration Process ---")
    
    # 1. Load the raw dataset
    try:
        df = pd.read_csv(RAW_PATH)
        print(f"Successfully loaded dataset from: {RAW_PATH}")
    except FileNotFoundError:
        print(f"Error: Could not find the file at {RAW_PATH}. Please make sure you uploaded it.")
        sys.exit(1)
        
    # 2. Validate columns (Ensure all expected project features are present)
    expected_columns = [
        'CustomerID', 'ProdTaken', 'Age', 'TypeofContact', 'CityTier', 
        'Occupation', 'Gender', 'NumberOfPersonVisiting', 'PreferredPropertyStar', 
        'MaritalStatus', 'NumberOfTrips', 'Passport', 'OwnCar', 
        'NumberOfChildrenVisiting', 'Designation', 'MonthlyIncome', 
        'PitchSatisfactionScore', 'ProductPitched', 'NumberOfFollowups', 'DurationOfPitch'
    ]
    
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        print(f"Validation Error: The dataset is missing critical columns: {missing_cols}")
        sys.exit(1)
    else:
        print("Dataset column validation passed successfully!")

    # 3. Report a data summary
    print("\n--- Dataset Summary ---")
    print(f"Total Records (Rows): {df.shape[0]}")
    print(f"Total Attributes (Columns): {df.shape[1]}")
    
    print("\nTarget Variable Class Balance ('ProdTaken'):")
    balance = df['ProdTaken'].value_counts(normalize=True) * 100
    for category, percentage in balance.items():
        label = "Purchased (1)" if category == 1 else "Not Purchased (0)"
        print(f"  - {label}: {percentage:.2f}%")
        
    print("\nData registration checklist finalized. File tracking ready for Git repository push.")

if __name__ == "__main__":
    register_data()
