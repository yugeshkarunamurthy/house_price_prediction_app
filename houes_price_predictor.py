import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# --- Load model ---
@st.cache_resource
def load_model():
    with open("house_price_predictor.pkl", "rb") as f:
        return joblib.load(f)

model = load_model()

# --- Constants ---
FEATURE_NAMES = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
                 'waterfront', 'view', 'condition', 'grade', 'yr_built', 'yr_renovated']

# --- Page setup ---
st.title("🏡 House Price Prediction App")
st.markdown("Enter house features to estimate its selling price 💰")
# --- Inputs ---
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        bedrooms = st.number_input("🛏️ Bedrooms", min_value=0, step=1, value=3)
        bathrooms = st.number_input("🛁 Bathrooms", min_value=0.0, step=0.25, value=2.0)
        sqft_living = st.number_input("📏 Living Area (sqft)", min_value=100, step=10, value=1800)
        sqft_lot = st.number_input("🌳 Lot Area (sqft)", min_value=500, step=10, value=5000)
        floors = st.number_input("🏢 Number of Floors", min_value=1.0, step=0.5, value=1.0)

    with col2:
        waterfront = st.selectbox("🌊 Waterfront View", [0, 1], format_func=lambda x: "Yes" if x else "No", index=0)
        view = st.slider("👁️‍🗨️ View Rating", min_value=0, max_value=4, value=0)
        condition = st.slider("🧱 Condition (1-5)", min_value=1, max_value=5, value=3)
        grade = st.slider("🏗️ Grade (1-13)", min_value=1, max_value=13, value=7)
        yr_built = st.number_input("📅 Year Built", min_value=1800, max_value=2025, step=1, value=2000)
        yr_renovated = st.number_input("🔧 Year Renovated (0 if never)", min_value=0, max_value=2025, step=1, value=0)

    submitted = st.form_submit_button("🔮 Predict Price")

# --- Validation ---
if submitted:
    if sqft_living <= 0 or sqft_lot <= 0:
        st.error("❌ Square footage values must be greater than zero.")
    elif yr_renovated != 0 and yr_renovated < yr_built:
        st.error("❌ Renovation year cannot be before the year built.")
    else:
        # --- Prepare data for prediction ---
        input_data = pd.DataFrame([{
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'sqft_living': sqft_living,
            'sqft_lot': sqft_lot,
            'floors': floors,
            'waterfront': waterfront,
            'view': view,
            'condition': condition,
            'grade': grade,
            'yr_built': yr_built,
            'yr_renovated': yr_renovated
        }])

        prediction = model.predict(input_data)[0]
        st.success(f"🏷️ Estimated House Price: **${prediction:,.2f}**")

        # --- Feature importance (if supported) ---
        if hasattr(model, "feature_importances_"):
            st.markdown("### 📊 Feature Importance")
            importance_df = pd.DataFrame({
                "Feature": FEATURE_NAMES,
                "Importance": model.feature_importances_
            }).sort_values(by="Importance", ascending=False)

            st.bar_chart(importance_df.set_index("Feature"))

        else:
            st.info("ℹ️ Feature importance not supported for this model.")
