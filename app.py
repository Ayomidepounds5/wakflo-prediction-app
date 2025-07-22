import streamlit as st
import joblib
import pandas as pd
import numpy as np
import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from PIL import Image # Import the Image library to handle local images
import shap # Import the SHAP library for model explanation
import matplotlib.pyplot as plt # Import for plotting

# --- Page Configuration ---
st.set_page_config(
    page_title="WAKFLO | Predicting Repeat Customer Behavior",
    page_icon="🌊",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- Define The Custom Transformer ---
class DateAndTenureTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X_copy = X.copy()
        reg_date = pd.to_datetime(X_copy['RegistrationDate'], errors='coerce')
        X_copy['Reg_Year'] = reg_date.dt.year
        X_copy['Reg_Month'] = reg_date.dt.month
        X_copy['Reg_DayOfWeek'] = reg_date.dt.dayofweek
        today = pd.to_datetime('2024-01-01', errors='coerce')
        X_copy['Tenure_Days'] = (today - reg_date).dt.days
        X_copy = X_copy.drop('RegistrationDate', axis=1)
        return X_copy

# --- Load ONLY The Saved Model ---
try:
    model = joblib.load("xgboost_model.joblib")
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'xgboost_model.joblib' is in the same directory.")
    st.stop()

# --- UI & Styling ---
logo_col, title_col = st.columns([1, 4])
with logo_col:
    try:
        logo = Image.open("Asset 8@4x-8.png")
        st.image(logo, width=100)
    except FileNotFoundError:
        st.error("Logo file not found.")

with title_col:
    st.subheader("WAKFLO")
    st.title("Predicting Repeat Customer Behavior")

st.markdown("---")

# --- Quick Scenarios Section ---
st.subheader("Quick Scenarios")
st.markdown("Click a button to load a pre-defined customer profile and see the prediction.")

def set_scenario(age, income, purchases, avg_value, satisfaction, premium):
    st.session_state.age = age
    st.session_state.income = income
    st.session_state.total_purchases = purchases
    st.session_state.avg_order_value = avg_value
    st.session_state.avg_satisfaction = satisfaction
    st.session_state.premium_member = premium

scenario_col1, scenario_col2 = st.columns(2)
with scenario_col1:
    st.button("High-Potential New Customer", on_click=set_scenario, args=(24, "High", 1, 150.0, 8.0, "No"))
with scenario_col2:
    st.button("At-Risk Loyal Customer", on_click=set_scenario, args=(55, "Medium", 25, 75.0, 2.5, "Yes"))

st.markdown("---")


# --- Create The Input Form ---
st.subheader("Manual Input")
col1, col2 = st.columns(2)
with col1:
    reg_date = st.date_input("Registration Date", datetime.date(2023, 1, 1), key='reg_date')
    gender = st.selectbox("Gender", ["Female", "Male", "Other"], key='gender')
    income = st.selectbox("Income Level", ["Low", "Medium", "High"], key='income')
    country = st.selectbox("Country", ["USA", "Canada", "UK", "Australia", "Germany", "Japan", "Brazil", "Other"], key='country')
    city = st.selectbox("City", ["New York", "London", "Tokyo", "Sydney", "Paris", "Berlin", "Other"], key='city')
    fav_category = st.selectbox("Favorite Category", ["Clothing", "Electronics", "Home Goods", "Books", "Toys", "Food"], key='fav_category')
    sec_fav_category = st.selectbox("Second Favorite Category", ["Clothing", "Electronics", "Home Goods", "Books", "Toys", "Food"], key='sec_fav_category')
with col2:
    age = st.number_input("Age", 10, 100, 30, key='age')
    total_purchases = st.number_input("Total Purchases", 0, 100, 5, key='total_purchases')
    avg_order_value = st.number_input("Average Order Value ($)", 0.0, 1000.0, 50.0, key='avg_order_value')
    clv = st.number_input("Customer Lifetime Value ($)", 0.0, 10000.0, 500.0, key='clv')
    email_engage = st.slider("Email Engagement Rate", 0.0, 1.0, 0.5, key='email_engage')
    social_engage = st.slider("Social Media Engagement Rate", 0.0, 1.0, 0.3, key='social_engage')
    cs_interactions = st.number_input("Customer Service Interactions", 0, 50, 1, key='cs_interactions')

st.markdown("---")
mobile_usage = st.select_slider("Mobile App Usage", ["Low", "Medium", "High"], key='mobile_usage')
premium_member = st.radio("Premium Member?", ["Yes", "No"], horizontal=True, key='premium_member')
has_returned = st.radio("Has Returned Items Previously?", ["Yes", "No"], horizontal=True, key='has_returned')
email_conv = st.slider("Email Conversion Rate", 0.0, 1.0, 0.1, key='email_conv')
social_conv = st.slider("Social Media Conversion Rate", 0.0, 1.0, 0.05, key='social_conv')
search_conv = st.slider("Search Engine Conversion Rate", 0.0, 1.0, 0.02, key='search_conv')
avg_satisfaction = st.slider("Average Satisfaction Score", 1.0, 10.0, 6.0, key='avg_satisfaction')

# NEW: Add a hypothetical feature for browsing behavior
st.number_input("Number of Browsing Sessions (Future Feature)", 0, 200, 10, key='browsing_sessions')
st.caption("Note: 'Browsing Sessions' is a hypothetical feature for a future model and does not affect the current prediction.")


# --- Prediction Logic ---
if st.button("Predict for WAKFLO", type="primary"):
    input_data = pd.DataFrame({
        'RegistrationDate': [pd.to_datetime(st.session_state.reg_date)], 'Age': [st.session_state.age], 'Gender': [st.session_state.gender],
        'IncomeLevel': [st.session_state.income], 'Country': [st.session_state.country], 'City': [st.session_state.city], 'TotalPurchases': [st.session_state.total_purchases],
        'AverageOrderValue': [st.session_state.avg_order_value], 'CustomerLifetimeValue': [st.session_state.clv],
        'FavoriteCategory': [st.session_state.fav_category], 'SecondFavoriteCategory': [st.session_state.sec_fav_category],
        'EmailEngagementRate': [st.session_state.email_engage], 'SocialMediaEngagementRate': [st.session_state.social_engage],
        'MobileAppUsage': [st.session_state.mobile_usage], 'CustomerServiceInteractions': [st.session_state.cs_interactions],
        'AverageSatisfactionScore': [st.session_state.avg_satisfaction], 'EmailConversionRate': [st.session_state.email_conv],
        'SocialMediaConversionRate': [st.session_state.social_conv], 'SearchEngineConversionRate': [st.session_state.search_conv],
        'PremiumMember': [st.session_state.premium_member], 'HasReturnedItems': [st.session_state.has_returned]
    })

    date_transformer = DateAndTenureTransformer()
    data_with_tenure = date_transformer.transform(input_data)
    numeric_features = data_with_tenure.select_dtypes(include=np.number).columns.tolist()
    categorical_features = data_with_tenure.select_dtypes(exclude=np.number).columns.tolist()
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    preprocessor_def = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    processed_data = preprocessor_def.fit_transform(data_with_tenure)
    final_columns = preprocessor_def.get_feature_names_out()
    processed_df = pd.DataFrame(processed_data, columns=final_columns)
    training_columns = model.feature_names_in_
    for col in training_columns:
        if col not in processed_df.columns:
            processed_df[col] = 0
    processed_df = processed_df[training_columns]
    
    prediction = model.predict(processed_df)
    prediction_proba = model.predict_proba(processed_df)

    st.markdown("---")
    st.subheader("Prediction Result")
    probability_of_repeat = prediction_proba[0][1]
    if prediction[0] == 1:
        st.success(f"**YES**, this customer is likely to be a repeat customer.")
        st.progress(float(probability_of_repeat))
    else:
        st.error(f"**NO**, this customer is unlikely to be a repeat customer.")
        st.progress(float(probability_of_repeat))
    st.metric(label="Probability of Repeat Purchase", value=f"{probability_of_repeat:.2%}")
    st.info("This prediction is based on historical WAKFLO customer data.", icon="💡")

    # --- NEW: Prediction Explanation Section ---
    st.markdown("---")
    st.subheader("Why did the model decide this?")
    
    # Use SHAP to explain the model's prediction
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(processed_df)
    
    # Create a SHAP waterfall plot
    fig, ax = plt.subplots()
    # We use shap_values[0] because we are explaining a single prediction
    shap.waterfall_plot(shap_values[0], max_display=10, show=False)
    plt.tight_layout() # Adjust layout to prevent labels overlapping
    st.pyplot(fig)
    
    st.markdown("""
    **How to read this chart:**
    * The **base value** `E[f(X)]` at the bottom is the model's average prediction across all training data.
    * Each bar shows how a feature's value **pushed the prediction** higher (red) or lower (blue) from the base value.
    * The **final prediction** `f(x)` at the top is the result of all these pushes combined.
    """)

