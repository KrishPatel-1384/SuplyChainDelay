import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
import joblib
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Supply Chain Delay Prediction & Analytics",
    
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
    <style>
    .main-title { font-size: 2.2rem; color: #1f77b4; font-weight: 700; margin-bottom: 0px; }
    .sub-text { font-size: 1.1rem; color: #555555; margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

# --- Header Section ---
st.markdown('<p class="main-title"> Supply Chain Delay Prediction & Analytics Hub</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Single-shipment risk assessment, exploratory analytics, and model evaluation performance.</p>', unsafe_allow_html=True)
st.markdown("---")

# --- Load Dataset, Preprocessor & Models ---
@st.cache_resource
def load_artifacts():
    df = pd.read_csv('/home/patel-krish/anaconda3/BrainyBeam/SuplyChainDelay/data/syndelay_v1.csv')
    preprocessor = joblib.load('/home/patel-krish/anaconda3/BrainyBeam/SuplyChainDelay/models/preprocessor.joblib')
    xgb_model = joblib.load('/home/patel-krish/anaconda3/BrainyBeam/SuplyChainDelay/models/xgboost_delay_model.joblib')
    rf_model = joblib.load('/home/patel-krish/anaconda3/BrainyBeam/SuplyChainDelay/models/random_forest_delay_model.joblib')
    lgb_model = joblib.load('/home/patel-krish/anaconda3/BrainyBeam/SuplyChainDelay/models/lightgbm_delay_model.joblib')
    return df, preprocessor, xgb_model, rf_model, lgb_model

try:
    df, preprocessor, xgb_model, rf_model, lgb_model = load_artifacts()
except Exception as e:
    st.error(f" Error loading artifacts. Make sure notebooks have been executed and models are saved in 'models/'.\nDetails: {e}")
    st.stop()

# --- Navigation Tabs ---
tab1, tab2, tab3 = st.tabs([" Single Shipment Predictor", " EDA Visualizations", " Model Evaluation & Comparison"])

# ==========================================
# TAB 1: SINGLE SHIPMENT PREDICTOR
# ==========================================
with tab1:
    st.subheader(" Individual Order Risk Assessment")
    st.markdown("Configure individual shipment details in the sidebar to evaluate delay risks.")

    # Sidebar Inputs
    st.sidebar.header(" Shipment Parameters")
    model_choice = st.sidebar.selectbox("Select Prediction Model", ["XGBoost", "Random Forest", "LightGBM"])

    shipping_mode = st.sidebar.selectbox("Shipping Mode", ["Standard Class", "Second Class", "First Class", "Same Day"])
    market = st.sidebar.selectbox("Market Region", ["Europe", "LATAM", "Pacific Asia", "USCA", "Africa"])
    customer_segment = st.sidebar.selectbox("Customer Segment", ["Consumer", "Corporate", "Home Office"])
    department_name = st.sidebar.selectbox("Department", ["Fan Shop", "Apparel", "Golf", "Footwear", "Outdoors", "Fitness", "Technology"])

    sales = st.sidebar.number_input("Sales Amount ($)", min_value=1.0, max_value=2000.0, value=250.0)
    profit_per_order = st.sidebar.number_input("Profit Per Order ($)", min_value=-1000.0, max_value=1000.0, value=35.0)
    order_item_quantity = st.sidebar.slider("Order Quantity", 1, 5, 2)
    order_item_discount_rate = st.sidebar.slider("Discount Rate", 0.0, 0.25, 0.1, step=0.01)
    product_price = st.sidebar.number_input("Product Price ($)", min_value=1.0, max_value=2000.0, value=120.0)
    sales_per_customer = st.sidebar.number_input("Sales Per Customer ($)", min_value=1.0, max_value=2000.0, value=250.0)
    latitude = st.sidebar.number_input("Destination Latitude", value=18.25, format="%.4f")
    longitude = st.sidebar.number_input("Destination Longitude", value=-66.37, format="%.4f")

    input_df = pd.DataFrame([{
        'sales': sales,
        'profit_per_order': profit_per_order,
        'order_item_quantity': order_item_quantity,
        'order_item_discount_rate': order_item_discount_rate,
        'product_price': product_price,
        'sales_per_customer': sales_per_customer,
        'latitude': latitude,
        'longitude': longitude,
        'shipping_mode': shipping_mode,
        'market': market,
        'customer_segment': customer_segment,
        'department_name': department_name
    }])

    col1, col2 = st.columns([1, 1], gap="medium")

    with col1:
        st.markdown("#### Input Summary")
        st.dataframe(input_df.T, use_container_width=True, height=380)

    with col2:
        st.markdown("#### Risk Evaluation Result")
        if st.button("Evaluate Shipment Risk", type="primary", use_container_width=True):
            active_model = xgb_model if model_choice == "XGBoost" else (rf_model if model_choice == "Random Forest" else lgb_model)
            
            processed_input = preprocessor.transform(input_df)
            prediction = active_model.predict(processed_input)[0]
            probabilities = active_model.predict_proba(processed_input)[0]

            risk_info = {
                0: (" Low Risk: On-Time / Early Delivery Expected", "success"),
                1: (" Moderate Risk: Potential Logistics Delay", "warning"),
                2: (" High Risk: Severe Delay Expected", "error")
            }
            label_text, status = risk_info[prediction]

            if status == "success": st.success(f"**Status:** {label_text}")
            elif status == "warning": st.warning(f"**Status:** {label_text}")
            else: st.error(f"**Status:** {label_text}")

            st.markdown("##### Probability Breakdown")
            prob_df = pd.DataFrame({
                'Risk Category': ['On-Time (0)', 'Moderate Delay (1)', 'Severe Delay (2)'],
                'Probability': [f"{p*100:.2f}%" for p in probabilities]
            })
            st.table(prob_df)

            if prediction == 2:
                st.error("- **Action:** High delay probability. Prioritize immediate carrier rerouting.")
            elif prediction == 1:
                st.warning("- **Action:** Moderate risk. Monitor shipment tracking closely.")
            else:
                st.success("- **Action:** Standard operational flow. No intervention needed.")
        else:
            st.info(" Set order parameters in the sidebar and click **'Evaluate Shipment Risk'**.")

# ==========================================
# TAB 2: EDA VISUALIZATIONS
# ==========================================
with tab2:
    st.subheader(" Exploratory Data Analysis (EDA) Graphs")
    st.markdown("Visualizing key historical trends and operational patterns from the dataset.")

    sns.set_theme(style='whitegrid')
    col_a, col_b = st.columns(2, gap="medium")

    with col_a:
        st.markdown("#### Delay Risk Distribution by Shipping Mode")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.countplot(data=df, x='shipping_mode', hue='label', palette='viridis', ax=ax)
        ax.set_title("Shipping Mode vs Delay Risk")
        ax.set_xlabel("Shipping Mode")
        ax.set_ylabel("Order Count")
        ax.legend(title='Delay Label', labels=['On-Time', 'Moderate', 'Severe'])
        st.pyplot(fig)

        st.markdown("#### Profitability Distribution by Delay Label")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.boxplot(data=df, x='label', y='profit_per_order', palette='Set2', ax=ax)
        ax.set_ylim(-200, 300)
        ax.set_title("Profit per Order vs Delay Severity")
        ax.set_xlabel("Delay Label")
        ax.set_ylabel("Profit ($)")
        st.pyplot(fig)

    with col_b:
        st.markdown("#### Delay Risk Across Global Markets")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.countplot(data=df, x='market', hue='label', palette='magma', ax=ax)
        ax.set_title("Market Region vs Delay Risk")
        ax.set_xlabel("Market")
        ax.set_ylabel("Order Count")
        plt.xticks(rotation=30)
        ax.legend(title='Delay Label', labels=['On-Time', 'Moderate', 'Severe'])
        st.pyplot(fig)

        st.markdown("#### Department Shipment Volume")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        dept_counts = df['department_name'].value_counts().reset_index()
        dept_counts.columns = ['department_name', 'count']
        sns.barplot(data=dept_counts, x='department_name', y='count', palette='crest', ax=ax)
        ax.set_title("Order Volume by Department")
        ax.set_xlabel("Department")
        ax.set_ylabel("Total Orders")
        plt.xticks(rotation=45)
        st.pyplot(fig)

# ==========================================
# TAB 3: MODEL EVALUATION & COMPARISON
# ==========================================
with tab3:
    st.subheader(" Model Evaluation & Performance Comparison")
    st.markdown("Comparing overall classification performance across trained models.")

    if st.button("Run Model Evaluation Metrics", type="primary"):
        with st.spinner("Evaluating models across the entire dataset..."):
            X = df[['sales', 'profit_per_order', 'order_item_quantity', 'order_item_discount_rate', 
                    'product_price', 'sales_per_customer', 'latitude', 'longitude', 
                    'shipping_mode', 'market', 'customer_segment', 'department_name']]
            y = df['label']
            X_proc = preprocessor.transform(X)

            y_pred_xgb = xgb_model.predict(X_proc)
            y_pred_rf = rf_model.predict(X_proc)
            y_pred_lgb = lgb_model.predict(X_proc)

            acc_xgb = accuracy_score(y, y_pred_xgb)
            acc_rf = accuracy_score(y, y_pred_rf)
            acc_lgb = accuracy_score(y, y_pred_lgb)

        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("XGBoost Accuracy", f"{acc_xgb*100:.2f}%")
        col_m2.metric("Random Forest Accuracy", f"{acc_rf*100:.2f}%")
        col_m3.metric("LightGBM Accuracy", f"{acc_lgb*100:.2f}%")

        st.markdown("#### Overall Accuracy Comparison Chart")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        models_perf = {'XGBoost': acc_xgb, 'Random Forest': acc_rf, 'LightGBM': acc_lgb}
        bars = ax.bar(models_perf.keys(), models_perf.values(), color=['#2b5c8f', '#2ca02c', '#ff7f0e'])
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Accuracy Score")
        ax.set_title("Model Performance Benchmark")

        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval - 0.05, f"{yval:.4f}", ha='center', color='white', fontweight='bold')

        st.pyplot(fig)
    else:
        st.info("Click **'Run Model Evaluation Metrics'** to compute and display model performance charts and scores.")