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
    page_title="SupplyChainWatch | Delay Prediction Engine",
    
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Theme & Styling (Matching Dark Green Sidebar Reference) ---
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-color: #0c231e;
        color: white;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    .main-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff;
    }
    .card-container {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# --- Load Dataset & Artifacts ---
@st.cache_resource
def load_project_artifacts():
    df = pd.read_csv('/home/patel-krish/anaconda3/BrainyBeam/SuplyChainDelay/data/syndelay_v1.csv')
    preprocessor = joblib.load('/home/patel-krish/anaconda3/BrainyBeam/SuplyChainDelay/models/preprocessor.joblib')
    xgb_model = joblib.load('/home/patel-krish/anaconda3/BrainyBeam/SuplyChainDelay/models/xgboost_delay_model.joblib')
    rf_model = joblib.load('/home/patel-krish/anaconda3/BrainyBeam/SuplyChainDelay/models/random_forest_delay_model.joblib')
    lgb_model = joblib.load('/home/patel-krish/anaconda3/BrainyBeam/SuplyChainDelay/models/lightgbm_delay_model.joblib')
    return df, preprocessor, xgb_model, rf_model, lgb_model

try:
    df, preprocessor, xgb_model, rf_model, lgb_model = load_project_artifacts()
except Exception as e:
    st.error(f" Error loading model artifacts. Make sure your notebooks are executed and models are saved in 'models/'. Details: {e}")
    st.stop()

# --- SIDEBAR NAVIGATION (Matching Reference Layout) ---
st.sidebar.markdown("###  SupplyChainWatch")
st.sidebar.markdown("<p style='font-size: 0.85rem; color: #a3b8b3;'>DELAY PREDICTION ENGINE</p>", unsafe_allow_html=True)
st.sidebar.markdown("---")

# Currency / Unit Selector (Inspired by reference UI)
unit_mode = st.sidebar.radio("Currency / Metric View", ["$ USD", "₹ INR"])
conversion_rate = 83.0 if unit_mode == "₹ INR" else 1.0
currency_symbol = "₹" if unit_mode == "₹ INR" else "$"

st.sidebar.markdown("---")
st.sidebar.markdown("### **NAVIGATE**")

navigation = st.sidebar.radio(
    "Go to",
    [
        "Dataset Overview",
        "Charts",
        "Model Details",
        "Single Shipment Prediction",
        "About"
    ],
    label_visibility="collapsed"
)

# ==========================================
# 1. DATASET OVERVIEW
# ==========================================
if navigation == "Dataset Overview":
    st.markdown("##  Dataset Overview")
    st.markdown("Data-driven supply chain delay risk analysis from historical shipment records.")
    
    # Top Metric Cards (Similar to reference image layout)
    total_records = len(df)
    train_records = int(total_records * 0.8)
    unique_markets = df['market'].nunique()
    avg_sales = df['sales'].mean() * conversion_rate

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("TOTAL SHIPMENTS", f"{total_records:,}")
    col2.metric("USED IN TRAINING", f"{train_records:,}")
    col3.metric("GLOBAL MARKETS", f"{unique_markets}")
    col4.metric(f"AVG SALES ({currency_symbol})", f"{currency_symbol}{avg_sales:,.2f}")

    st.markdown("---")
    st.subheader("Full cleaned dataset (preview)")
    st.dataframe(df.head(10), use_container_width=True)

# ==========================================
# 2. CHARTS (EDA VISUALIZATIONS)
# ==========================================
elif navigation == "Charts":
    st.markdown("##  Exploratory Data Analysis")
    st.markdown("Visualizing shipping patterns, profitability, and operational bottlenecks.")

    sns.set_theme(style='whitegrid')
    col_c1, col_c2 = st.columns(2, gap="medium")

    with col_c1:
        st.markdown("#### Delay Risk by Shipping Mode")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, x='shipping_mode', hue='label', palette='viridis', ax=ax)
        ax.set_title("Shipping Mode Distribution")
        ax.set_xlabel("Shipping Mode")
        ax.set_ylabel("Count")
        ax.legend(title='Delay Risk', labels=['On-Time', 'Moderate', 'Severe'])
        st.pyplot(fig)

        st.markdown("#### Profit per Order vs Delay Label")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df, x='label', y='profit_per_order', palette='Set2', ax=ax)
        ax.set_ylim(-200, 300)
        ax.set_title("Profitability Distribution")
        ax.set_xlabel("Delay Risk Label")
        ax.set_ylabel("Profit ($)")
        st.pyplot(fig)

    with col_c2:
        st.markdown("#### Market Region Delay Breakdown")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, x='market', hue='label', palette='magma', ax=ax)
        ax.set_title("Market Region Breakdown")
        ax.set_xlabel("Market")
        ax.set_ylabel("Count")
        plt.xticks(rotation=30)
        ax.legend(title='Delay Risk', labels=['On-Time', 'Moderate', 'Severe'])
        st.pyplot(fig)

        st.markdown("#### Department Shipment Volume")
        fig, ax = plt.subplots(figsize=(6, 4))
        dept_counts = df['department_name'].value_counts().reset_index()
        dept_counts.columns = ['department_name', 'count']
        sns.barplot(data=dept_counts, x='department_name', y='count', palette='crest', ax=ax)
        ax.set_title("Volume by Department")
        ax.set_xlabel("Department")
        ax.set_ylabel("Orders")
        plt.xticks(rotation=45)
        st.pyplot(fig)

# ==========================================
# 3. MODEL DETAILS
# ==========================================
elif navigation == "Model Details":
    st.markdown("##  Model Details & Benchmarks")
    st.markdown("Performance comparison across trained classification algorithms.")

    X = df[['sales', 'profit_per_order', 'order_item_quantity', 'order_item_discount_rate', 
            'product_price', 'sales_per_customer', 'latitude', 'longitude', 
            'shipping_mode', 'market', 'customer_segment', 'department_name']]
    y = df['label']
    X_proc = preprocessor.transform(X)

    acc_xgb = accuracy_score(y, xgb_model.predict(X_proc))
    acc_rf = accuracy_score(y, rf_model.predict(X_proc))
    acc_lgb = accuracy_score(y, lgb_model.predict(X_proc))

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("XGBoost Accuracy", f"{acc_xgb*100:.2f}%")
    col_m2.metric("Random Forest Accuracy", f"{acc_rf*100:.2f}%")
    col_m3.metric("LightGBM Accuracy", f"{acc_lgb*100:.2f}%")

    st.markdown("---")
    st.subheader("Model Accuracy Comparison Chart")
    fig, ax = plt.subplots(figsize=(7, 4))
    perf = {'XGBoost': acc_xgb, 'Random Forest': acc_rf, 'LightGBM': acc_lgb}
    bars = ax.bar(perf.keys(), perf.values(), color=['#0c231e', '#2ca02c', '#ff7f0e'])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy Score")
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval - 0.05, f"{yval:.4f}", ha='center', color='white', fontweight='bold')
    st.pyplot(fig)

# ==========================================
# 4. SINGLE SHIPMENT PREDICTION
# ==========================================
elif navigation == "Single Shipment Prediction":
    st.markdown("## 🔍 Predict Delay Risk for One Shipment")
    st.markdown("Configure order characteristics to calculate real-time delay probability.")

    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        model_choice = st.selectbox("Select Prediction Model", ["XGBoost", "Random Forest", "LightGBM"])
    with col_sel2:
        shipping_mode = st.selectbox("Shipping Mode", ["Standard Class", "Second Class", "First Class", "Same Day"])

    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        market = st.selectbox("Market Region", ["Europe", "LATAM", "Pacific Asia", "USCA", "Africa"])
        customer_segment = st.selectbox("Customer Segment", ["Consumer", "Corporate", "Home Office"])
        department_name = st.selectbox("Department", ["Fan Shop", "Apparel", "Golf", "Footwear", "Outdoors", "Fitness"])
    with col_p2:
        sales_input = st.number_input(f"Sales Amount ({currency_symbol})", min_value=1.0, max_value=5000.0, value=250.0)
        sales = sales_input / conversion_rate
        profit_input = st.number_input(f"Profit Per Order ({currency_symbol})", min_value=-1000.0, max_value=2000.0, value=35.0)
        profit_per_order = profit_input / conversion_rate
        order_item_quantity = st.slider("Order Item Quantity", 1, 5, 2)
    with col_p3:
        order_item_discount_rate = st.slider("Discount Rate", 0.0, 0.25, 0.1, step=0.01)
        product_price_input = st.number_input(f"Product Price ({currency_symbol})", min_value=1.0, max_value=5000.0, value=120.0)
        product_price = product_price_input / conversion_rate
        sales_per_customer = sales

    latitude = st.number_input("Destination Latitude", value=18.25, format="%.4f")
    longitude = st.number_input("Destination Longitude", value=-66.37, format="%.4f")

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

    st.markdown("---")
    if st.button("Evaluate Shipment Risk", type="primary", use_container_width=True):
        active_model = xgb_model if model_choice == "XGBoost" else (rf_model if model_choice == "Random Forest" else lgb_model)
        
        proc_input = preprocessor.transform(input_df)
        prediction = active_model.predict(proc_input)[0]
        probs = active_model.predict_proba(proc_input)[0]

        risk_map = {
            0: (" Low Risk: On-Time / Early Delivery Expected", "success"),
            1: (" Moderate Risk: Potential Logistics Delay", "warning"),
            2: (" High Risk: Severe Delay Expected", "error")
        }
        text, status = risk_map[prediction]

        if status == "success": st.success(f"**Status:** {text}")
        elif status == "warning": st.warning(f"**Status:** {text}")
        else: st.error(f"**Status:** {text}")

        st.markdown("#### Probability Breakdown")
        prob_df = pd.DataFrame({
            'Risk Category': ['On-Time (0)', 'Moderate Delay (1)', 'Severe Delay (2)'],
            'Probability': [f"{p*100:.2f}%" for p in probs]
        })
        st.table(prob_df)

        if prediction == 2:
            st.error("- **Action Required:** High likelihood of delay. Prioritize immediate carrier rerouting.")
        elif prediction == 1:
            st.warning("- **Action Required:** Moderate delay risk. Monitor tracking updates closely.")
        else:
            st.success("- **Action Required:** Standard operational flow. No intervention needed.")

# ==========================================
# 5. ABOUT
# ==========================================
elif navigation == "About":
    st.markdown("##  About SupplyChainWatch")
    st.markdown("""
    **SupplyChainWatch** is an enterprise-grade analytics and machine learning system engineered to predict potential operational delays in logistics.
    
    * **Data Source:** Historical shipment records containing geographic routes, shipping modes, customer segments, and discount metrics.
    * **Models Implemented:** XGBoost, Random Forest, and LightGBM classifiers.
    * **Objective:** Provide logistics teams with real-time early warnings to prioritize high-risk shipments and optimize delivery schedules.
    """)
