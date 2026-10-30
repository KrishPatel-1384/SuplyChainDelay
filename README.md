#  Supply Chain Delay Prediction 

An end-to-end Machine Learning and data analytics solution designed to predict shipment delay risks, identify operational bottlenecks, and provide real-time risk assessments for logistics planning.

---

##  Project Directory Structure

```text
Supply_Chain_Delay_Prediction/
├── datasets/
│   └── syndelay_v1.csv                  # Historical shipment records dataset
├── models/
│   ├── preprocessor.joblib              # Saved scikit-learn preprocessing pipeline
│   ├── xgboost_delay_model.joblib       # Trained XGBoost classifier
│   ├── random_forest_delay_model.joblib # Trained Random Forest classifier
│   └── lightgbm_delay_model.joblib      # Trained LightGBM classifier
├── notebook/
│   ├── 01_data_understanding.ipynb      # Initial data inspection & profiling
│   ├── 02_eda.ipynb                     # Exploratory data analysis & visualizations
│   ├── 03_data_preprocessing.ipynb      # Data cleaning & train-test splitting
│   ├── 04_feature_engineering.ipynb     # Column transformers & feature scaling
│   ├── 05_model_training.ipynb          # Model training & artifact serialization
│   └── 06_model_evaluation.ipynb        # Model performance & accuracy comparison
├── app/
│   └── app.py                           # Interactive Streamlit Web Dashboard
├── requirements.txt                     # Project library dependencies
└── README.md                            # Project documentation