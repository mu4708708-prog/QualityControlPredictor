import streamlit as st
import pickle
import pandas as pd

# -------------------------
# 1. Load Model & Scaler
# -------------------------
with open("Pass_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("scaler_model.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# -------------------------
# 2. Streamlit UI Setup
# -------------------------
st.set_page_config(page_title="Quality Prediction", page_icon="*", layout="centered")
st.title("Machine Quality Prediction System")
st.markdown("""
This app predicts whether a machine's product will *Pass* or *Fail* based on five input features.  
The input data is scaled before being fed into the trained ML model.
""")

# -------------------------
# 3. User Input
# -------------------------
st.sidebar.header("Input Features")

temperature = st.sidebar.number_input("Temperature (°C)", min_value=55.0, max_value=120.0,  value=55.0, step=0.1)

vibration = st.sidebar.number_input("Vibration (mm/s)", min_value=0.01, max_value=20.0, value=1.0, step=0.1)

pressure = st.sidebar.number_input("Pressure (bar)", min_value=1.5, max_value=3.50,value=1.5, step=0.1)

flow_rate = st.sidebar.number_input("Flow Rate (L/min)", min_value=1.0, max_value=500.0, value=10.0, step=0.1)

efficiency = st.sidebar.number_input("Efficiency (%)", min_value=5.0, max_value=100.0,value=90.0, step=0.1)

# Collect inputs
input_df = pd.DataFrame({
    "Temperature (°C)": [temperature],
    "Vibration (mm/s)": [vibration],
    "Pressure (bar)": [pressure],
    "Flow Rate (L/min)": [flow_rate],
    "Efficiency (%)": [efficiency]
})

st.subheader("Entered Input Values")
st.table(input_df)

# -------------------------
# 4. Prediction
# -------------------------
if st.button("Predict"):
    try:
        # Scale the input
        scaled_data = scaler.transform(input_df)

        # Predict
        prediction = model.predict(scaled_data)
        label = prediction[0]   # "Pass" or "Fail"

        # Correct output mapping
        if label == "Pass":
            result_text = "Pass"
        else:
            result_text = "Fail"

        st.subheader("Prediction Result")
        st.success(f"Predicted Quality: {result_text}")

    except Exception as e:
        st.error(f"Error: {str(e)}")
