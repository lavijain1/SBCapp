import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load('sbc_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_medians = joblib.load('feature_medians.pkl')
feature_names = joblib.load("feature_names.pkl")  # Ensure this is saved during training

st.set_page_config(page_title="SBC Predictor", layout="centered")
st.title("Safe Bearing Capacity (SBC) Prediction Tool")
st.markdown("### Select Conditions and Enter Relevant Parameters")

# Define features by sheet type
sheet_features = {
    "nps_shear": ['depth', 'water_table', 'density', 'angle','cohesion_kg_cm2','general_shear','local_shear','void_ratio','soil_type', 'problem_type'],
    "nps_settlement": ['depth','length', 'width','n_value', 'settlement', 'depth_factor','water_table','requried_settlement', 'soil_type', 'problem_type'],
    "clay_shear": ['depth',  'water_table', 'density', 'angle','cohesion_kg_cm2', 'general_shear','local_shear','void_ratio','recommended_sbc_t_m2', 'soil_type', 'problem_type'],
    "clay_settlement": ['depth', 'length', 'width','settlement', 'depth_factor','void_ratio','cc', 'density','requried_settlement', 'soil_type', 'problem_type']
}

# User choices
soil_type = st.selectbox("Soil Type", options=["Non-plastic (NPS)", "Clay"])
problem_type = st.selectbox("Problem Type", options=["Shear", "Settlement"])

# Determine which features to show
sheet_key = f"{'nps' if soil_type == 'Non-plastic (NPS)' else 'clay'}_{problem_type.lower()}"
required_features = sheet_features[sheet_key]

# Input values storage
input_values = {}
st.markdown("### Input Parameters:")

# Input widgets
for feature in required_features:
    if feature == 'soil_type':
        input_values[feature] = 0 if soil_type == "Non-plastic (NPS)" else 1
    elif feature == 'problem_type':
        input_values[feature] = 0 if problem_type == "Shear" else 1
    else:
        label = feature.replace('_', ' ').title()
        val = st.number_input(f"{label}", format="%.4f", key=feature)
        input_values[feature] = val

# Check for missing (0.0) inputs excluding categorical
missing_fields = [feat for feat in required_features if feat not in ['soil_type', 'problem_type'] and input_values.get(feat, 0.0) == 0.0]

# Use session state to track confirmation
if 'confirmed_prediction' not in st.session_state:
    st.session_state.confirmed_prediction = False

if st.button("Predict SBC"):
    if missing_fields and not st.session_state.confirmed_prediction:
        with st.expander("⚠️ Missing Fields Detected - Click to Confirm"):
            st.warning(f"The following fields are 0.0 and might not be intended:\n\n- " + "\n- ".join(missing_fields))
            if st.button("Yes, predict anyway with 0.0 values"):
                st.session_state.confirmed_prediction = True
                st.experimental_rerun()
            elif st.button("No, I'll enter the values"):
                st.stop()
    else:
        # Prepare full input with fallback to medians for missing features
        full_input = {}
        for feat in feature_names:
            val = input_values.get(feat, feature_medians.get(feat, 0.0))
            if feat not in ['soil_type', 'problem_type'] and val == 0.0 and not st.session_state.confirmed_prediction:
                val = feature_medians.get(feat, 0.0)
            full_input[feat] = val

        input_df = pd.DataFrame([full_input])
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]

        st.success(f"Predicted Safe Bearing Capacity (SBC): {prediction:.2f} t/m²")

        # Reset confirmation for next prediction
        st.session_state.confirmed_prediction = False
