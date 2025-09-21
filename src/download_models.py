import gdown
import os

files = {
    "models/random_forest_model.pkl": "1tMC_rFCMH4XUkFXI_AGB6lE37kHYOksH",
    "results/random_forest_model.joblib": "1LcxrObEDOd8L-rrsUgnUKCNrxxh6kG2L"
}

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

for path, file_id in files.items():
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Downloading {path} ...")
    gdown.download(url, path, quiet=False)
