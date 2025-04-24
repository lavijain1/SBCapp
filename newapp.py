import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load('sbc_model.pkl')
scaler = joblib.load('scaler.pkl')

# Load precomputed median values (saved during training)
feature_medians = joblib.load('feature_medians.pkl')  # <-- You need to save this when training

# Set page configuration
st.set_page_config(page_title="SBC AI Tool", layout="centered")

st.title("Safe Bearing Capacity (SBC) AI Tool")
st.markdown("### Select Conditions and Enter Relevant Parameters")

# Define sheet-specific features
sheet_features = {
    "nps_shear": ['depth', 'water_table', 'density', 'angle','cohesion_kg_cm2','general_shear','local_shear','void_ratio','soil_type', 'problem_type'],
    "nps_settlement": ['depth','length', 'width','n_value', 'settlement', 'depth_factor','water_table','requried_settlement', 'soil_type', 'problem_type'],
    "clay_shear": ['depth',  'water_table', 'density', 'angle','cohesion_kg_cm2', 'general_shear','local_shear','void_ratio','recommended_sbc_t_m2', 'soil_type', 'problem_type'],
    "clay_settlement": ['depth', 'length', 'width','settlement', 'depth_factor','void_ratio','cc', 'density','requried_settlement', 'soil_type', 'problem_type']
}

# Select soil and problem type
soil_type = st.selectbox("Soil Type", options=["Non-plastic (NPS)", "Clay"])
problem_type = st.selectbox("Problem Type", options=["Shear", "Settlement"])

# Determine feature group
sheet_key = f"{'nps' if soil_type == 'Non-plastic (NPS)' else 'clay'}_{problem_type.lower()}"
required_features = sheet_features[sheet_key]

# Dictionary to store input values
input_values = {}

# Input widgets based on selected features
st.markdown("### Input Parameters:")
for feature in required_features:
    if feature == 'soil_type':
        input_values[feature] = 0 if soil_type == "Non-plastic (NPS)" else 1
    elif feature == 'problem_type':
        input_values[feature] = 0 if problem_type == "Shear" else 1
    else:
        label = feature.replace('_', ' ').title()
        input_values[feature] = st.number_input(f"{label}", format="%.4f")
# Prediction
if st.button("Predict SBC"):
    # Fill in missing features not shown in UI with 0
    feature_names = joblib.load("feature_names.pkl")  # <-- You must have saved this earlier during training

    full_input = {feat: input_values.get(feat, feature_medians[feat]) for feat in feature_names}
    input_df = pd.DataFrame([full_input])
    
    # Scale and predict
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    
    st.success(f"Predicted Safe Bearing Capacity (SBC): {prediction:.2f} t/m²")
