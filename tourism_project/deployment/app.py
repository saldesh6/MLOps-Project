import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Visit with Us - Tourism Predictor", layout="centered")
st.title("Visit with Us 🗺️")
st.subheader("Tourism Package Prediction Dashboard")
st.write("Input customer metrics below to evaluate purchase probability.")

model_path = os.path.join(os.path.dirname(__file__), "best_model.joblib")

@st.cache_resource
def load_model():
    return joblib.load(model_path)

try:
    model = load_model()
except FileNotFoundError:
    st.error("Could not locate the model file binaries. Please verify your pipeline runs.")
    st.stop()

# FIXED: Standardized path lookup mechanism to find tourism.csv flawlessly on Streamlit Cloud
@st.cache_data
def load_baseline_data():
    possible_paths = [
        "tourism.csv",
        "../tourism.csv",
        "../../tourism.csv",
        os.path.join(os.getcwd(), "tourism.csv")
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return pd.read_csv(path)
    return None

df_base = load_baseline_data()

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

if st.button("Evaluate Potential Purchase", type="primary"):
    numeric_features = [
        "Age", "CityTier", "NumberOfPersonVisiting", "PreferredPropertyStar",
        "NumberOfTrips", "Passport", "OwnCar", "NumberOfChildrenVisiting",
        "MonthlyIncome", "PitchSatisfactionScore", "NumberOfFollowups", "DurationOfPitch"
    ]
    
    input_data = pd.DataFrame([{
        "Age": Age, "TypeofContact": TypeofContact, "CityTier": CityTier,
        "DurationOfPitch": DurationOfPitch, "Occupation": Occupation, "Gender": Gender,
        "NumberOfPersonVisiting": NumberOfPersonVisiting, "NumberOfFollowups": NumberOfFollowups,
        "ProductPitched": ProductPitched, "PreferredPropertyStar": PreferredPropertyStar,
        "MaritalStatus": MaritalStatus, "NumberOfTrips": NumberOfTrips, "Passport": Passport,
        "OwnCar": OwnCar, "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
        "Designation": Designation, "MonthlyIncome": MonthlyIncome, "PitchSatisfactionScore": PitchSatisfactionScore
    }])
    
    prediction_proba = model.predict_proba(input_data)[0][1]
    prediction = 1 if prediction_proba >= 0.5 else 0
    
    st.write("---")
    if prediction == 1:
        st.success(f"### 🎉 High Potential Buyer! Conversion Probability: {prediction_proba*100:.1f}%")
        st.balloons()
    else:
        st.warning(f"### 🛑 Low Potential Buyer. Conversion Probability: {prediction_proba*100:.1f}%")
        
    # Explainable AI: Extract and Render Top Features Drivers
    try:
        st.write("### 🔑 Top Predictive Feature Drivers (Model Explainability)")
        xgb_step = model.named_steps['xgbclassifier']
        ct_step = model.named_steps['columntransformer']
        
        encoded_cats = ct_step.named_transformers_['onehotencoder'].get_feature_names_out()
        all_features = list(numeric_features) + list(encoded_cats)
        
        importance_df = pd.DataFrame({
            'Feature': all_features,
            'Weight': xgb_step.feature_importances_
        }).sort_values(by='Weight', ascending=False).head(5)
        
        st.bar_chart(data=importance_df, x='Feature', y='Weight', horizontal=True)
    except Exception as e:
        st.info(f"Feature rendering skipped: {str(e)}")
        
    if df_base is not None:
        st.write("### 📊 Demographic Context & Purchase Trends")
        
        st.write(f"#### Monthly Income Breakdown by Occupation for **{Designation}s**")
        filtered_df = df_base[df_base['Designation'] == Designation]
        if not filtered_df.empty:
            income_chart = filtered_df.groupby('Occupation')['MonthlyIncome'].mean().reset_index()
            st.bar_chart(data=income_chart, x='Occupation', y='MonthlyIncome')
        
        st.write("#### Package Purchase Conversion Ratios Across Designations")
        conversion_chart = df_base.groupby('Designation')['ProdTaken'].mean().reset_index()
        conversion_chart['Conversion %'] = conversion_chart['ProdTaken'] * 100
        st.line_chart(data=conversion_chart, x='Designation', y='Conversion %')
    else:
        st.info("Baseline data parsing pending. Upload tourism.csv to enable trend analytics graphs.")
