# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, hf_hub_download


# Define constants for the dataset and output paths
# API client relies on HF_TOKEN being set as an environment variable
api = HfApi(token=os.getenv("HF_TOKEN"))

repo_id = "greatlearninglokeshrajoria/tourism-package-prediction"
file_in_repo = "tourism.csv"

# Download the file first
file_path_local = hf_hub_download(repo_id=repo_id, filename=file_in_repo, repo_type="dataset")

df = pd.read_csv(file_path_local)
print("Dataset loaded successfully.")

df['Gender'] = df['Gender'].replace('Fe Male', 'Female')
print(df['Gender'].value_counts())

# Encode categorical columns
# Select categorical columns
cat_cols = ['TypeofContact', 'Occupation', 'ProductPitched',
            'MaritalStatus', 'Designation', 'Gender']

label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Save for inverse_transform if needed

print(df[cat_cols].nunique())  # Verify encoding [file:1]

# Define target variable
target_col = 'ProdTaken'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id=repo_id,
        repo_type="dataset",
    )
