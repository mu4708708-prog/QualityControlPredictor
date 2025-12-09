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
st.set_page_config(page_title="Quality Prediction", page_icon="⚙️", layout="wide")

# Custom CSS for better UI
st.markdown("""
    <style>
        .main-title {
            text-align: center;
            font-size: 40px;
            color: #2A8CE0;
            font-weight: 700;
        }
        .subtitle {
            text-align: center;
            font-size: 18px;
            color: #666;
            margin-bottom: 30px;
        }
        .card {
            background: #FFFFFF;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.12);
        }
        .predict-btn button {
            width: 100%;
            font-size: 20px;
            font-weight: 600;
            border-radius: 10px;
            padding: 12px;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='main-title'>Machine Quality Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Enter machine parameters and get instant Pass/Fail prediction.</p>", unsafe_allow_html=True)

# -------------------------
# 3. Centered Input Card
# -------------------------
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.subheader("🔧 Input Features")

    temperature = st.number_input("🌡 Temperature (°C)", min_value=55.0, max_value=120.0, value=55.0, step=0.1)
    vibration = st.number_input("📳 Vibration (mm/s)", min_value=0.01, max_value=20.0, value=1.0, step=0.1)
    pressure = st.number_input("⏱ Pressure (bar)", min_value=1.5, max_value=3.50, value=1.5, step=0.1)
    flow_rate = st.number_input("💧 Flow Rate (L/min)", min_value=1.0, max_value=500.0, value=10.0, step=0.1)
    efficiency = st.number_input("⚡ Efficiency (%)", min_value=5.0, max_value=100.0, value=90.0, step=0.1)

    input_df = pd.DataFrame({
        "Temperature (°C)": [temperature],
        "Vibration (mm/s)": [vibration],
        "Pressure (bar)": [pressure],
        "Flow Rate (L/min)": [flow_rate],
        "Efficiency (%)": [efficiency]
    })

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# 4. Prediction Button
# -------------------------
with col2:
    st.markdown("<div class='predict-btn'>", unsafe_allow_html=True)
    predict = st.button("🔍 Predict Quality")
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# 5. Result & Table
# -------------------------
if predict:
    try:
        scaled_data = scaler.transform(input_df)
        prediction = model.predict(scaled_data)
        label = prediction[0]

        result = "Pass ✅" if label == "Pass" else "Fail ❌"

        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("📌 Input Summary")
            st.table(input_df)

            st.subheader("📊 Prediction Result")
            if label == "Pass":
                st.success(f"Predicted Quality: {result}")
            else:
                st.error(f"Predicted Quality: {result}")

            st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {str(e)}")
