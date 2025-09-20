🚀 Network Traffic Classification with Machine Learning

This project implements and compares **Random Forest**, **MLP (Neural Network)**, and **XGBoost** models for **network traffic classification**.  
The goal is to analyze network traffic flow features and classify them into categories using different ML algorithms.

---

 📂 Project Structure
ML-Network-Traffic-Classification/
├── data/ # Dataset folder (download from Kaggle)
├── models/ # Saved models (.pkl files)
├── notebooks/ # Jupyter notebooks with training, evaluation, and comparison
│ └── training_and_evaluation.ipynb
├── src/ # Python scripts for training & evaluation
│ ├── train_rf.py
│ ├── train_mlp.py
│ ├── train_xgb.py
│ └── evaluate.py
├── requirements.txt # Dependencies
└── README.md # Project documentation

---

## 📊 Dataset
The dataset is taken from Kaggle:  
👉 [Network Traffic Classifier Dataset]
( https://www.kaggle.com/datasets/noobbcoder2/vpn-and-non-vpn-application-traffic-cic-vpn2016)  

> **Note:** Download the dataset and place it inside the `data/` folder before running the code.

---

## 🛠 Installation
Clone the repository and install dependencies:

Bash:

git clone https://github.com/your-username/network-traffic-classification-ml.git
cd network-traffic-classification-ml
pip install -r requirements.txt
________________________________________
🚀 Usage
Run Models
You can train individual models using:
python src/train_randomforest.py
python src/train_mlp.py
python src/train_xgboost.py
Run Notebook
For detailed results, visualizations, and comparison:
jupyter notebook notebooks/ network_traffic_classification.ipynb
________________________________________
📈 Results
		=== Metrics Summary ===

               precision  recall  f1-score  accuracy
MLP              0.6374   0.6294    0.6231    0.6294
XGBoost          0.9094   0.9050    0.9049    0.9050
RandomForest     0.9627   0.9575    0.9580    0.9575

Confusion matrices and heatmaps are available in the notebook.
The final comparison plot shows side-by-side performance of all models.
________________________________________
💡 Future Improvements
•	Add more ML models (SVM, Gradient Boosting, etc.)
•	Hyperparameter tuning for better accuracy
•	Extend to deep learning with CNNs/LSTMs
•	Deploy as a web application (optional)
________________________________________
👤 Author
•	Pranta Chowdhury (CSE – Premier University Chattogram)
•	GitHub: Pranta-Chy
________________________________________
⚡ License
This project is for educational purposes. You can freely use and modify it.
