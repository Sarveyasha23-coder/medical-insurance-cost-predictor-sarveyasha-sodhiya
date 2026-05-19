import streamlit as st
import pickle
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Medical Insurance Cost Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model (10).pkl", "rb"))
scaler = pickle.load(open("scaler (20).pkl", "rb"))

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background:  linear-gradient(135deg, #0b0f19, #1e293b, #334155);
    background-size: cover;
    color: white;
}

.main-title {
    text-align: center;
    font-size: 3.2rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    font-size: 1.15rem;
    color: #dbeafe;
    margin-bottom: 30px;
}

.glass-card {
    background: rgba(255, 255, 255, 0.12);
    border-radius: 20px;
    padding: 25px;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
    margin-bottom: 20px;
}

.result-box {
    background: linear-gradient(135deg, #10b981, #059669);
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    font-size: 1.8rem;
    font-weight: bold;
    color: white;
    box-shadow: 0 8px 25px rgba(0,0,0,0.25);
    margin-top: 25px;
}

.footer {
    text-align: center;
    margin-top: 50px;
    color: #e0f2fe;
    font-size: 14px;
}

div.stButton > button {
    width: 100%;
    height: 3.2em;
    font-size: 18px;
    font-weight: 600;
    border-radius: 15px;
    background: linear-gradient(90deg, #06b6d4, #3b82f6);
    color: white;
    border: none;
    transition: 0.3s;
}

div.stButton > button:hover {
    transform: scale(1.03);
    background: linear-gradient(90deg, #2563eb, #0891b2);
}

[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] {
    border-radius: 12px;
}

/* White font for Age, BMI, Children, Gender only */
.stNumberInput label,
.stNumberInput input,
.stSelectbox label {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="main-title">🏥 AI Medical Insurance Cost Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Predict healthcare insurance charges using machine learning regression models</div>',
    unsafe_allow_html=True
)

# ---------------- HERO IMAGE ----------------
st.image(
    "https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?q=80&w=1600&auto=format&fit=crop",
    use_container_width=True
)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------- INPUT SECTION ----------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("📋 Enter Patient Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=25)
    sex = st.selectbox("Gender", ["Male", "Female"])

with col2:
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=24.5)
    children = st.number_input("Children", min_value=0, max_value=10, value=0)

with col3:
    smoker = st.selectbox("Smoker", ["Yes", "No"])
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- ENCODING ----------------
sex_val = 1 if sex == "Male" else 0
smoker_val = 1 if smoker == "Yes" else 0

region_map = {
    "northeast": 0,
    "northwest": 1,
    "southeast": 2,
    "southwest": 3
}
region_val = region_map[region]

# ---------------- PREDICTION ----------------
if st.button("🚀 Predict Insurance Cost"):
    input_data = np.array([[age, sex_val, bmi, children, smoker_val, region_val]])
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0]

    st.markdown(
        f"""
        <div class="result-box">
            Estimated Medical Insurance Cost<br><br>
            💰 ${prediction:,.2f}
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------- SIDEBAR INFO ----------------
with st.sidebar:
    st.title("Project Info")
    st.write("Model: Best Performing Regressor")
    st.write("Pipeline: Data Preprocessing + Prediction")
    st.write("Deployment: Streamlit Cloud")
    st.write("Type: End-to-End ML Application")

# ---------------- FOOTER ----------------
st.markdown(
    """
    <div class="footer">
        Developed by <b>Sarveyasha Sodhiya</b> | AI & ML Engineering Project | Industry-Ready Healthcare Analytics
    </div>
    """,
    unsafe_allow_html=True
)
