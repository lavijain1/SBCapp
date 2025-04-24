import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load('sbc_model.pkl')
scaler = joblib.load('scaler.pkl')

# Load precomputed median values (saved during training)
feature_medians = joblib.load('feature_medians.pkl')  # <-- You need to save this when training

# Set page configuration
st.set_page_config(page_title="SBC Predictor", layout="centered")

st.title("Safe Bearing Capacity (SBC) Prediction Tool")
st.markdown("### Select Conditions and Enter Relevant Parameters")

# Define sheet-specific features
sheet_features = {
    "nps_shear": ['depth', 'water_table', 'density', 'angle', 'length', 'width', 'n_value', 'depth_factor', 'soil_type', 'problem_type'],
    "nps_settlement": ['depth', 'density', 'length', 'width', 'settlement', 'requried_settlement', 'soil_type', 'problem_type'],
    "clay_shear": ['depth', 'cohesion_kg_cm2', 'angle', 'void_ratio', 'length', 'width', 'soil_type', 'problem_type'],
    "clay_settlement": ['depth', 'cc', 'length', 'width', 'settlement', 'requried_settlement', 'soil_type', 'problem_type']
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
    elif feature in ['n_value']:
        input_values[feature] = st.number_input("SPT-N Value", min_value=0)
    elif feature in ['settlement', 'requried_settlement']:
        input_values[feature] = st.number_input(f"{feature.replace('_', ' ').title()} (mm)", min_value=0.0)
    elif feature in ['cohesion_kg_cm2']:
        input_values[feature] = st.number_input("Cohesion (kg/cm²)", min_value=0.0)
    elif feature in ['angle']:
        input_values[feature] = st.number_input("Angle of Internal Friction (°)", min_value=0.0)
    elif feature in ['density']:
        input_values[feature] = st.number_input("Density (t/m³)", min_value=0.0)
    elif feature in ['depth']:
        input_values[feature] = st.number_input("Depth (m)", min_value=0.0)
    elif feature in ['water_table']:
        input_values[feature] = st.number_input("Water Table Depth (m)", min_value=0.0)
    elif feature in ['void_ratio']:
        input_values[feature] = st.number_input("Void Ratio", min_value=0.0)
    elif feature in ['depth_factor']:
        input_values[feature] = st.number_input("Depth Factor", min_value=0.0)
    elif feature in ['cc']:
        input_values[feature] = st.number_input("Compression Index (Cc)", min_value=0.0)
    elif feature in ['length', 'width']:
        input_values[feature] = st.number_input(f"{feature.capitalize()} (m)", min_value=0.0)

# Prediction
if st.button("Predict SBC"):
    # Fill in missing features not shown in UI with 0
    all_model_features = list(model.feature_names_in_)
    full_input = {feat: input_values.get(feat, feature_medians[feat]) for feat in all_model_features}
    input_df = pd.DataFrame([full_input])
    
    # Scale and predict
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    
    st.success(f"Predicted Safe Bearing Capacity (SBC): {prediction:.2f} t/m²")
