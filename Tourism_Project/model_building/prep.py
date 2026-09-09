# Use the project folder name, then the folder where model building code will be stored.
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Fallback path checking: Checks root first (GitHub Actions environment), then falls back to local Colab folder
if os.path.exists("tourism.csv"):
    DATA_PATH = "tourism.csv"
elif os.path.exists("Tourism_Project/tourism.csv"):
    DATA_PATH = "Tourism_Project/tourism.csv"
else:
    DATA_PATH = "tourism.csv"

df = pd.read_csv(DATA_PATH)   
df.drop(columns=["CustomerID"], inplace=True)   # complete the code: drop the customer identifier column, it is not a predictive feature

# NOTE: categorical columns are intentionally left as raw strings.
# The training pipeline one-hot-encodes them, and the Streamlit app also sends
# raw category values. Encoding them here (e.g. LabelEncoder) would make training
# and serving use different representations, silently breaking predictions.

target = "ProdTaken"  
X = df.drop(columns=[target])
y = df[target]

# stratify keeps the (imbalanced) purchase ratio consistent across splits
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y   
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
print("ProdTaken distribution in train:")
print(ytrain.value_counts())
