import os
import joblib
import pandas as pd
import streamlit as st

# Set up clean app headers
st.set_page_config(page_title="Visit with Us - Wellness Tourism Predictor", layout="centered")
st.title("Visit with Us 🗺️")
st.subheader("Wellness Tourism Package Prediction Dashboard")
st.write("Input customer metrics below to evaluate purchase probability.")

# 1. Dynamically locate and safely load the trained model asset sitting next to this script
model_path = os.path.join(os.path.dirname(__file__), "best_model.joblib")

@st.cache_resource
def load_model():
    return joblib.load(model_path)

try:
    model = load_model()
except FileNotFoundError:
    st.error(f"Could not locate the model file at {model_path}. Please verify your pipeline artifact files.")
    st.stop()

# 2. Construct the layout forms for data entry matching your data description
st.write("### Customer Demographics & Profile")
col1, col2 = st.columns(2)

with col1:
    Age = st.number_input("Age", min_value=18, max_value=100, value=35)
    CityTier = st.selectbox("City Tier", [1, 2, 3], help="Tier 1 > Tier 2 > Tier 3 development standards")
    Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Freelancer"])
    Gender = st.selectbox("Gender", ["Male", "Female"])

with col2:
    MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
    Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    MonthlyIncome = st.number_input("Monthly Income (Gross)", min_value=0, value=25000)
    OwnCar = st.selectbox("Owns a Car?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

st.write("### Trip Details & Historical Interactions")
col3, col4 = st.columns(2)

with col3:
    TypeofContact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
    NumberOfPersonVisiting = st.slider("Number of Persons Visiting", 1, 10, 2)
    NumberOfChildrenVisiting = st.slider("Number of Children Visiting (Age < 5)", 0, 5, 0)
    NumberOfTrips = st.number_input("Average Annual Trips Taken", min_value=0, value=3)

with col4:
    Passport = st.selectbox("Has Valid Passport?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    PreferredPropertyStar = st.selectbox("Preferred Hotel Property Star Rating", [3, 4, 5], index=0)
    DurationOfPitch = st.number_input("Duration of Sales Pitch (Minutes)", min_value=0, value=15)
    PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", 1, 5, 3)

ProductPitched = st.selectbox("Product Pitched", ["Wellness", "Basic", "Deluxe", "Standard", "King"])
NumberOfFollowups = st.slider("Number of Follow-ups Conducted", 0, 10, 3)

# 3. Process inputs into a dataframe for prediction on form submission
if st.button("Evaluate Potential Purchase", type="primary"):
    # Must explicitly match the feature tracking columns used during training
    input_data = pd.DataFrame([{
        "Age": Age,
        "TypeofContact": TypeofContact,
        "CityTier": CityTier,
        "DurationOfPitch": DurationOfPitch,
        "Occupation": Occupation,
        "Gender": Gender,
        "NumberOfPersonVisiting": NumberOfPersonVisiting,
        "NumberOfFollowups": NumberOfFollowups,
        "ProductPitched": ProductPitched,
        "PreferredPropertyStar": PreferredPropertyStar,
        "MaritalStatus": MaritalStatus,
        "NumberOfTrips": NumberOfTrips,
        "Passport": Passport,
        "OwnCar": OwnCar,
        "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
        "Designation": Designation,
        "MonthlyIncome": MonthlyIncome,
        "PitchSatisfactionScore": PitchSatisfactionScore
    }])
    
    # 4. Generate predictions using the pipeline
    prediction_proba = model.predict_proba(input_data)[0][1]
    prediction = int(prediction_proba >= 0.5)
    
    st.write("---")
    if prediction == 1:
        st.success(f"### 🎉 High Potential Buyer! Probability score: {prediction_proba*100:.1f}%")
        st.balloons()
        st.write("The marketing team should **actively prioritize** contacting this customer for the Wellness Tourism Package.")
    else:
        st.warning(f"### 🛑 Low Potential Buyer. Probability score: {prediction_proba*100:.1f}%")
        st.write("This customer is unlikely to purchase the package at this time. Standard campaign follow-ups recommended.")
