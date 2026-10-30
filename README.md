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


[] Setup & Installation Instructions
Follow these steps to set up and run the project locally on your machine:

1. Clone or Open Project Folder
Navigate to your project root directory in your terminal:

Bash
cd Supply_Chain_Delay_Prediction



2. Install Dependencies
Install all required Python packages listed in requirements.txt:

Bash
pip install -r requirements.txt
 Running the Jupyter Notebooks
Execute the notebooks sequentially in your Jupyter environment to follow the data pipeline from raw data to model evaluation:

01_data_understanding.ipynb: Inspect data dimensions, types, and target distributions.

02_eda.ipynb: Generate exploratory data visualizations (shipping modes, markets, profit distributions).

03_data_preprocessing.ipynb: Handle cleaning and data splitting.

04_feature_engineering.ipynb: Set up encoding and scaling pipelines.

05_model_training.ipynb: Train XGBoost, Random Forest, and LightGBM models, saving artifacts to the models/ folder.

06_model_evaluation.ipynb: Evaluate models using classification reports, confusion matrices, and accuracy metrics.



[] Launching the Streamlit Dashboard
The project includes an interactive web dashboard for real-time risk assessment of individual shipments (built with zero bulk-prediction processing for quick operational decisions).

To launch the app:

Bash
streamlit run app/app.py
Dashboard Features:
 Single Shipment Predictor: Configure individual order parameters (shipping mode, market, sales, discount rate, etc.) via the sidebar and instantly evaluate delay risk (0: On-Time, 1: Moderate Delay, 2: High Risk).

 EDA Visualizations: Explore historical dataset insights, market distribution trends, and profit margins.

Model Evaluation Hub: Compare overall classification performance and accuracy metrics across XGBoost, Random Forest, and LightGBM models.



[] Tech Stack
Python (Core programming language)

Pandas & NumPy (Data manipulation & cleaning)

Scikit-Learn (Preprocessing pipelines & metrics)

XGBoost, LightGBM & Scikit-Learn Ensembles (Machine learning classification models)

Matplotlib & Seaborn (Data visualization & EDA)

Streamlit ( web application framework)
