# Import required libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the saved model and scaler
model = joblib.load('heart_disease_model.joblib')
scaler = joblib.load('scaler.joblib')

# Set page configuration
st.set_page_config(page_title="Heart Disease Prediction", layout="wide", page_icon="❤️")

# CSS for improved alignment of tooltips and general styling
st.markdown(
    """
    <style>
    .tooltip-inline {
        display: inline-block;
        vertical-align: middle;
        font-size: 14px;
        color: #6C757D;
        margin-left: 8px;
        cursor: pointer;
    }
    .tooltip-inline:hover {
        color: #007BFF;
    }
    .magic-box {
        background-color: #F7F9FC;
        border: 1px solid #E0E0E0;
        padding: 10px;
        border-radius: 6px;
        margin-top: 5px;
    }
    .dataframe-container {
        border: 1px solid #E0E0E0;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 20px;
        background-color: #FFF;
    }
    </style>
    """, unsafe_allow_html=True
)

# Main container
with st.container():
    st.image("heart_banner.jpg", use_container_width=True)
    st.markdown(
        '<div style="text-align: center; padding-top: 10px;">'
        '<h1 style="color: #2C2D3A;">Heart Disease Prediction</h1>'
        '<p style="color: #6C757D; font-size: 16px;">Enter patient details below to predict the likelihood of heart disease</p>'
        '</div>',
        unsafe_allow_html=True
    )

# Sidebar input fields
with st.sidebar:
    st.header("Patient Details")

    # Input fields with collapsible magic info boxes
    
    # age
    age = st.number_input("Age (years):", min_value=29, max_value=80, value=50, step=1)

    st.sidebar.markdown("------------------")
    
    # sex
    sex = st.radio("Sex:", options=["Male", "Female"])

    st.sidebar.markdown("------------------")

    # chest pain
    col1, col2 = st.columns([10, 1])
    col1.write("### Chest Pain Type:")
    if col2.button("ℹ️", key="chest_pain_info"):
        st.session_state["show_chest_info"] = not st.session_state.get("show_chest_info", False)

    chest_pain = st.radio("", options=["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])

    if st.session_state.get("show_chest_info", False):
        st.markdown('''
            <div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px;">
                <ul>
                    <li><strong>Typical Angina:</strong> Pain from reduced blood flow, triggered by physical activity.</li>
                    <li><strong>Atypical Angina:</strong> Pain not related to exertion.</li>
                    <li><strong>Non-anginal Pain:</strong> Pain unrelated to heart problems.</li>
                    <li><strong>Asymptomatic:</strong> No noticeable symptoms of chest pain.</li>
                    </ul>
            </div>
        ''', unsafe_allow_html=True)

    st.sidebar.markdown("------------------")

    # resting bp
    col1, col2 = st.columns([10, 1])
    col1.write("### Resting Blood Pressure (mmHg):")
    resting_bp_s = col1.slider("", min_value=50.0, max_value=200.0, value=120.0)
    if col2.button("ℹ️", key="resting_bp_info"):
        st.session_state["show_bp_info"] = not st.session_state.get("show_bp_info", False)

    if st.session_state.get("show_bp_info", False):
        st.markdown(
            '<div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px;">'
            'Blood pressure when resting; high levels are linked to cardiovascular risk.'
            '</div>',
            unsafe_allow_html=True
        )

    st.sidebar.markdown("------------------")
    
    # cholesterol
    col1, col2 = st.columns([10, 1])
    col1.write("### Cholesterol Level (mg/dL):")
    cholesterol = col1.slider("", min_value=50, max_value=400, value=200, step=10)
    if col2.button("ℹ️", key="cholesterol_info"):
        st.session_state["show_cholesterol_info"] = not st.session_state.get("show_cholesterol_info", False)

    if st.session_state.get("show_cholesterol_info", False):
        st.markdown(
            '<div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px;">'
            'Serum cholesterol in mg/dL. High levels increase cardiovascular risk.'
            '</div>',
            unsafe_allow_html=True
        )

    st.sidebar.markdown("------------------")

    # fasting bs
    col1, col2= st.columns([10, 1])
    col1.write("### Fasting Blood Sugar (>120mg/dL):")
    fasting_bs = col1.radio("", options=["Yes ", "No "])
    if col2.button("ℹ️", key="fasting_bs_info"):
        st.session_state["show_fbs_info"] = not st.session_state.get("show_fbs_info", False)

    if st.session_state.get("show_fbs_info", False):
        st.markdown(
            '<div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px;">'
            'Whether fasting glucose exceeds 120 mg/dL, indicating possible diabetes.'
            '</div>',
            unsafe_allow_html=True
        )
    
    st.sidebar.markdown("------------------") 
    
    # resting ecg
    col1, col2 = st.columns([10, 1])
    col1.write("### Resting ECG:")
    resting_ecg = col1.radio("", options=["Normal", "Abnormal", "Left Ventricular Hypertrophy"])
    if col2.button("ℹ️", key="resting_ecg_info"):
        st.session_state["show_ecg_info"] = not st.session_state.get("show_ecg_info", False)

    if st.session_state.get("show_ecg_info", False):
        st.markdown(
            '<div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px;">'
            'Electrocardiogram results at rest: normal, abnormal, or hypertrophic signs.'
            '</div>',
            unsafe_allow_html=True
        )

    st.sidebar.markdown("------------------")  
    
    # max heart rate
    col1, col2 = st.columns([10, 1])
    col1.write("### Maximum Heart Rate Achieved:")
    max_heart_rate = col1.slider("", min_value=60, max_value=220, value=150)
    if col2.button("ℹ️", key="max_heart_rate_info"):
        st.session_state["show_hr_info"] = not st.session_state.get("show_hr_info", False)

    if st.session_state.get("show_hr_info", False):
        st.markdown(
            '<div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px;">'
            'Highest heart rate reached during physical activity; indicates cardiac fitness.'
            '</div>',
            unsafe_allow_html=True
        )

    st.sidebar.markdown("------------------")
    
    # exercise angina
    col1, col2 = st.columns([10, 1])
    col1.write("### Exercise-Induced Angina:")
    exercise_angina = col1.radio("", options=["Yes", "No"])
    if col2.button("ℹ️", key="exercise_angina_info"):
        st.session_state["show_angina_info"] = not st.session_state.get("show_angina_info", False)

    if st.session_state.get("show_angina_info", False):
        st.markdown(
            '<div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px;">'
            'Chest pain experienced due to exercise, a strong indicator of heart disease.'
            '</div>',
            unsafe_allow_html=True
        )

    st.sidebar.markdown("------------------")
    
    # oldpeak
    col1, col2 = st.columns([10, 1])
    col1.write("### ST Depression (Oldpeak):")
    oldpeak = col1.slider("", min_value=0.0, max_value=6.0, value=1.0, step=0.1)
    if col2.button("ℹ️", key="oldpeak_info"):
        st.session_state["show_oldpeak_info"] = not st.session_state.get("show_oldpeak_info", False)

    if st.session_state.get("show_oldpeak_info", False):
        st.markdown(
            '<div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px;">'
            'ST depression, indicating ischemia (reduced blood flow to the heart).'
            '</div>',
            unsafe_allow_html=True
        )
    
    st.sidebar.markdown("------------------")
        
    # ST slope
    col1, col2 = st.columns([10, 1])
    col1.write("### ST Slope:")
    st_slope = col1.radio("", options=["Normal", "Upsloping", "Flat", "Downsloping"])
    if col2.button("ℹ️", key="st_slope_info"):
        st.session_state["show_st_slope_info"] = not st.session_state.get("show_st_slope_info", False)

    if st.session_state.get("show_st_slope_info", False):
        st.markdown('''
        <div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px;">
        <ul>
            <li><strong>Normal:</strong> A healthy ST segment.</li>
            <li><strong>Upsloping:</strong> Generally normal but less typical.</li>
            <li><strong>Flat:</strong> May indicate cardiac risk.</li>
            <li><strong>Downsloping:</strong> Often linked to ischemia or severe heart disease.</li>
        </ul>
        </div>
        ''', unsafe_allow_html=True)




# Dynamic input table
st.subheader("Current Input Data")
user_data_dict = {
    "Age": age, "Sex": "Male" if sex == "Male" else "Female", "Chest Pain Type": chest_pain,
    "Resting BP (mmHg)": resting_bp_s, "Cholesterol": cholesterol, "Fasting BS": "Yes" if fasting_bs == "Yes" else "No",
    "Resting ECG": resting_ecg, "Max Heart Rate": max_heart_rate, "Exercise Angina": exercise_angina,
    "Oldpeak": oldpeak, "ST Slope": st_slope
}
user_df = pd.DataFrame([user_data_dict])
st.table(user_df)

# Convert user inputs to model-compatible format and make prediction
sex = 1 if sex == "Male" else 0
chest_pain = {"Typical Angina": 1, "Atypical Angina": 2, "Non-anginal Pain": 3, "Asymptomatic": 4}[chest_pain]
fasting_bs = 1 if fasting_bs == "Yes" else 0
resting_ecg = {"Normal": 0, "Abnormal": 1, "Left Ventricular Hypertrophy": 2}[resting_ecg]
exercise_angina = 1 if exercise_angina == "Yes" else 0
st_slope = {"Normal": 0, "Upsloping": 1, "Flat": 2, "Downsloping": 3}[st_slope]

resting_bp_s_transformed = np.log1p(resting_bp_s - 43)
cholesterol_transformed = np.log1p(cholesterol + 18)
oldpeak_transformed = np.log(oldpeak + 0.01)

user_data = np.array([[age, sex, chest_pain, resting_bp_s_transformed, cholesterol_transformed,
                       fasting_bs, resting_ecg, max_heart_rate, exercise_angina, oldpeak_transformed, st_slope]])
scaled_input = scaler.transform(user_data)

# Prediction
st.subheader("Prediction")
if st.button("Predict"):
    prediction = model.predict(scaled_input)
    prediction_proba = model.predict_proba(scaled_input)[0]

    if prediction[0] == 1:
        probability = prediction_proba[1] * 100  # Calculate probability
        st.markdown(
            f'''
            <div style="text-align:center; background-color:#ffcccc; padding: 20px; border-radius: 10px; border: 2px solid red; box-shadow: 0px 0px 10px rgba(255, 0, 0, 0.5);">
                <h3 style="color: red; font-family: 'Arial', sans-serif;">💔 Heart Disease Detected</h3>
                <p style="font-size: 18px; color: #a50000; font-weight: bold;"> 
                    <span style="font-size: 22px; color: red;">Probability:</span> {probability:.2f}%
                </p>
                <p style="font-size: 16px; color: #a50000;">It's crucial to take immediate steps for your heart health! 🔬</p>
            </div>
            ''',
            unsafe_allow_html=True
        )
    else:
        probability = prediction_proba[0] * 100  # Calculate probability
        st.markdown(
            f'''
            <div style="text-align:center; background-color:#e0ffcc; padding: 20px; border-radius: 10px; border: 2px solid green; box-shadow: 0px 0px 10px rgba(0, 255, 0, 0.5);">
                <h3 style="color: green; font-family: 'Arial', sans-serif;">❤️ No Heart Disease Detected</h3>
                <p style="font-size: 18px; color: #006600; font-weight: bold;"> 
                    <span style="font-size: 22px; color: green;">Probability:</span> {probability:.2f}%
                </p>
                <p style="font-size: 16px; color: #006600;">You're in the clear! Keep up the healthy habits! 💪</p>
            </div>
            ''',
            unsafe_allow_html=True
        )

