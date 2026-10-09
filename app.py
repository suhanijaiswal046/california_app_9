import streamlit as st
import numpy as np
import joblib

# Page
st.set_page_config(
    page_title="California House Price",
    page_icon="🏡",
    layout="centered"
)

# CSS
st.markdown("""
<style>
.stApp {
    background: #f5f7fb;
}

.title {
    text-align: center;
    font-size: 38px;
    font-weight: 800;
    color: #16325c;
}

.subtitle {
    text-align: center;
    color: #6b7280;
    margin-bottom: 30px;
}

.result {
    background: white;
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    box-shadow: 0 5px 20px rgba(0,0,0,0.08);
    margin-top: 25px;
}

.price {
    font-size: 42px;
    font-weight: 800;
    color: #2563eb;
}
</style>
""", unsafe_allow_html=True)


# Load model
obj = joblib.load("california.joblib")
model = obj["Model"]


# Header
st.markdown(
    '<div class="title">🏡 California House Price Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Enter property details to estimate its value</div>',
    unsafe_allow_html=True
)


# Inputs
col1, col2 = st.columns(2)

with col1:
    MedInc = st.number_input(
        "💰 Median Income",
        min_value=0.0,
        value=3.5,
        step=0.1
    )

    AveRooms = st.number_input(
        "🚪 Average Rooms",
        min_value=1.0,
        value=5.0,
        step=0.1
    )

    AveOccup = st.number_input(
        "👨‍👩‍👧 Average Occupancy",
        min_value=0.1,
        value=3.0,
        step=0.1
    )

with col2:
    HouseAge = st.number_input(
        "🏠 House Age",
        min_value=0.0,
        value=20.0,
        step=1.0
    )

    Population = st.number_input(
        "👥 Population",
        min_value=1.0,
        value=1000.0,
        step=100.0
    )

    Latitude = st.number_input(
        "📍 Latitude",
        min_value=32.0,
        max_value=42.0,
        value=35.0,
        step=0.01
    )


# Predict
st.markdown("")

if st.button("🔮 Predict House Price", use_container_width=True):

    data = np.array([
        MedInc,
        HouseAge,
        AveRooms,
        Population,
        AveOccup,
        Latitude
    ]).reshape(1, -1)

    prediction = model.predict(data)[0]

    price = prediction * 100000

    st.markdown(
        f"""
        <div class="result">
            <div>Estimated House Value</div>
            <div class="price">${price:,.0f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )