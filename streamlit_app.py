import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Page configuration
st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 42px;
        color: #1f77b4;
        text-align: center;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .sub-header {
        font-size: 20px;
        color: #666;
        text-align: center;
        margin-bottom: 30px;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        text-align: center;
    }
    .churn-yes {
        background-color: #ffebee;
        border: 2px solid #ef5350;
    }
    .churn-no {
        background-color: #e8f5e9;
        border: 2px solid #66bb6a;
    }
</style>
""", unsafe_allow_html=True)

# Load the trained model and preprocessors
@st.cache_resource
def load_model():
    try:
        with open('best_churn_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        return model, scaler, encoders
    except FileNotFoundError:
        st.error("Model files not found! Please ensure the model has been trained.")
        return None, None, None

model, scaler, encoders = load_model()

# Title and description
st.markdown('<p class="main-header">🏦 Bank Customer Churn Prediction System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predict whether a customer will leave the bank using Machine Learning</p>', unsafe_allow_html=True)

# Sidebar for user inputs
st.sidebar.header("📋 Customer Information")
st.sidebar.markdown("Enter the customer details below:")

# Input fields
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Demographic Information")
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650, step=1)
    age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
    gender = st.selectbox("Gender", ["Male", "Female"])
    country = st.selectbox("Country", ["France", "Germany", "Spain"])

with col2:
    st.subheader("💼 Account Information")
    tenure = st.slider("Tenure (years with bank)", min_value=0, max_value=10, value=5)
    balance = st.number_input("Account Balance", min_value=0.0, value=50000.0, step=1000.0)
    products_number = st.selectbox("Number of Products", [1, 2, 3, 4])
    estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0, step=1000.0)

col3, col4 = st.columns(2)

with col3:
    credit_card = st.radio("Has Credit Card?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

with col4:
    active_member = st.radio("Is Active Member?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

# Prediction button
if st.button("🔮 Predict Churn", type="primary"):
    if model is not None:
        # Encode categorical variables
        gender_encoded = encoders['gender'].transform([gender])[0]
        country_encoded = encoders['country'].transform([country])[0]

        # Feature engineering
        age_group = 0 if age <= 30 else 1 if age <= 40 else 2 if age <= 50 else 3
        has_balance = 1 if balance > 0 else 0
        balance_category = 0 if balance <= 0 else 1 if balance <= 50000 else 2 if balance <= 100000 else 3
        tenure_category = 0 if tenure <= 2 else 1 if tenure <= 5 else 2 if tenure <= 7 else 3
        credit_score_category = 0 if credit_score <= 400 else 1 if credit_score <= 600 else 2 if credit_score <= 700 else 3
        balance_to_salary_ratio = balance / (estimated_salary + 1)
        products_per_tenure = products_number / (tenure + 1)
        active_products_interaction = active_member * products_number

        # Create feature array
        features = np.array([[
            credit_score, age, tenure, balance, products_number,
            credit_card, active_member, estimated_salary,
            gender_encoded, country_encoded, age_group, has_balance,
            balance_category, tenure_category, credit_score_category,
            balance_to_salary_ratio, products_per_tenure, active_products_interaction
        ]])

        # Scale features
        features_scaled = scaler.transform(features)

        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]

        # Display results
        st.markdown("---")
        st.subheader("📊 Prediction Results")

        if prediction == 1:
            st.markdown(f"""
            <div class="prediction-box churn-yes">
                <h2>⚠️ HIGH RISK - Customer Likely to Churn</h2>
                <h3>Probability of Churn: {probability[1]*100:.2f}%</h3>
                <p>This customer shows strong indicators of potential churn. Immediate retention actions recommended.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-box churn-no">
                <h2>✅ LOW RISK - Customer Likely to Stay</h2>
                <h3>Probability of Staying: {probability[0]*100:.2f}%</h3>
                <p>This customer shows strong loyalty indicators. Continue providing excellent service.</p>
            </div>
            """, unsafe_allow_html=True)

        # Display feature values
        with st.expander("📈 View Customer Profile Summary"):
            summary_df = pd.DataFrame({
                'Feature': ['Credit Score', 'Age', 'Gender', 'Country', 'Tenure', 
                           'Balance', 'Products', 'Credit Card', 'Active Member', 'Salary'],
                'Value': [credit_score, age, gender, country, f"{tenure} years", 
                         f"${balance:,.2f}", products_number, 
                         "Yes" if credit_card == 1 else "No",
                         "Yes" if active_member == 1 else "No",
                         f"${estimated_salary:,.2f}"]
            })
            st.table(summary_df)

# Information section
st.markdown("---")
st.markdown("### 📚 About This Application")
col5, col6, col7 = st.columns(3)

with col5:
    st.markdown("""
    **🎯 Purpose**
    - Predict customer churn
    - Enable proactive retention
    - Reduce customer attrition
    """)

with col6:
    st.markdown("""
    **🤖 Model Information**
    - Algorithm: Gradient Boosting
    - Accuracy: 86.95%
    - F1-Score: 60.39%
    """)

with col7:
    st.markdown("""
    **📊 Key Features**
    - Age & Product Usage
    - Account Activity
    - Geographic Location
    """)

# Footer
st.markdown("---")
st.markdown("*Developed with Streamlit | Machine Learning for Banking | © 2025*")
