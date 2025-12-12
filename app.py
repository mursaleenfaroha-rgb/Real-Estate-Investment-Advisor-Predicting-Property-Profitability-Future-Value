import streamlit as st
import pandas as pd
import joblib
import datetime

# -------------------------
# Load trained ML pipelines
# -------------------------
@st.cache_resource
def load_models():
    clf = joblib.load("models/classification_model.joblib")
    reg = joblib.load("models/regression_model.joblib")
    return clf, reg

clf_model, reg_model = load_models()

# -------------------------
# App title and description
# -------------------------
st.title("🏡 Real Estate Investment Advisor")
st.write(
    "This app predicts whether a property is a **Good Investment** and "
    "estimates its **Future Price after 5 years** using machine learning."
)

# -------------------------
# User input form
# -------------------------
st.header("📋 Enter Property Details")

city = st.selectbox("City", ["Mumbai", "Delhi", "Bengaluru", "Chennai", "Kolkata"])
locality = st.text_input("Locality / Area", "Some Locality")

bhk = st.number_input("BHK", min_value=1, max_value=10, value=3)
size = st.number_input("Size (SqFt)", min_value=200, max_value=10000, value=1200)
price_lakhs = st.number_input("Current Price (Lakhs)", min_value=5.0, max_value=10000.0, value=75.0)

furnished_status = st.selectbox("Furnished Status", ["Unfurnished", "Semi-Furnished", "Fully-Furnished"])
availability_status = st.selectbox("Availability Status", ["Ready_to_move", "Under_Construction"])

nearby_schools = st.number_input("Nearby Schools", min_value=0, max_value=50, value=3)
nearby_hospitals = st.number_input("Nearby Hospitals", min_value=0, max_value=50, value=2)
public_transport = st.number_input("Public Transport Accessibility (0-10)", min_value=0, max_value=10, value=7)
parking_space = st.number_input("Parking Space (0-3)", min_value=0, max_value=3, value=1)
amenities = st.number_input("Amenities Score (0-10)", min_value=0, max_value=10, value=6)

year_built = st.number_input("Year Built", min_value=1950, max_value=2025, value=2005)

# -------------------------
# Feature engineering for input
# -------------------------
current_year = datetime.datetime.now().year
price_per_sqft = (price_lakhs * 100000) / max(size, 1)
age_of_property = current_year - year_built
school_density_score = nearby_schools / 10  # approximate normalization

input_data = {
    "City": city,
    "Locality": locality,
    "BHK": bhk,
    "Size_in_SqFt": size,
    "Price_in_Lakhs": price_lakhs,
    "Furnished_Status": furnished_status,
    "Availability_Status": availability_status,
    "Nearby_Schools": nearby_schools,
    "Nearby_Hospitals": nearby_hospitals,
    "Public_Transport_Accessibility": public_transport,
    "Parking_Space": parking_space,
    "Amenities": amenities,
    "Year_Built": year_built,
    "Price_per_SqFt": price_per_sqft,
    "Age_of_Property": age_of_property,
    "School_Density_Score": school_density_score,
}

input_df = pd.DataFrame([input_data])

st.subheader("🔍 Model Input Preview")
st.dataframe(input_df)

# -------------------------
# Run predictions
# -------------------------
if st.button("Predict"):
    # Classification
    class_pred = clf_model.predict(input_df)[0]
    class_prob = clf_model.predict_proba(input_df)[0, 1]

    # Regression
    future_price_pred = reg_model.predict(input_df)[0]

    st.subheader("📌 Results")

    if class_pred == 1:
        st.success(f"✅ Good Investment (Confidence: {class_prob:.2f})")
    else:
        st.error(f"⚠️ Not a Good Investment (Confidence: {class_prob:.2f})")

    st.info(f"💰 Estimated Price After 5 Years: **{future_price_pred:.2f} Lakhs**")
