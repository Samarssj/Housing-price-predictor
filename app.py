import joblib
import pandas as pd
import streamlit as st

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏡",
    layout="wide"
)

# -------------------- LOAD MODEL --------------------
MODEL_PATH = "model/house_price_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# -------------------- TITLE --------------------
st.title("🏡 House Price Predictor")
st.markdown(
    "Predict house prices using a **Machine Learning model (XGBoost)**."
)

st.divider()

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.header("📌 About")

    st.success("Model: XGBoost")

    st.write("""
Dataset:
- California Housing Dataset

Algorithm:
- XGBoost Regressor

Built using:
- Scikit-Learn
- XGBoost
- Streamlit
""")

    st.markdown("---")

    preset = st.selectbox(
        "Choose Sample Data",
        [
            "Custom",
            "Affordable Home",
            "Large Family Home",
            "Coastal Property"
        ]
    )

# -------------------- PRESETS --------------------
longitude = -122.23
latitude = 37.88
housing_median_age = 41
total_rooms = 880
total_bedrooms = 129
population = 322
households = 126
median_income = 8.3252
ocean_proximity = "NEAR BAY"

if preset == "Affordable Home":
    longitude = -122.23
    latitude = 37.88
    housing_median_age = 30
    total_rooms = 1200
    total_bedrooms = 220
    population = 500
    households = 220
    median_income = 4.2
    ocean_proximity = "INLAND"

elif preset == "Large Family Home":
    longitude = -121.95
    latitude = 37.75
    housing_median_age = 45
    total_rooms = 2600
    total_bedrooms = 500
    population = 900
    households = 400
    median_income = 6.8
    ocean_proximity = "<1H OCEAN"

elif preset == "Coastal Property":
    longitude = -122.40
    latitude = 37.95
    housing_median_age = 35
    total_rooms = 1800
    total_bedrooms = 300
    population = 650
    households = 280
    median_income = 7.5
    ocean_proximity = "NEAR BAY"

# -------------------- INPUTS --------------------
col1, col2 = st.columns(2)

with col1:
    longitude = st.number_input(
        "Longitude",
        value=float(longitude),
        step=0.01
    )

    latitude = st.number_input(
        "Latitude",
        value=float(latitude),
        step=0.01
    )

    housing_median_age = st.number_input(
        "Housing Median Age",
        value=int(housing_median_age),
        step=1
    )

    total_rooms = st.number_input(
        "Total Rooms",
        value=int(total_rooms),
        step=1
    )

    total_bedrooms = st.number_input(
        "Total Bedrooms",
        value=int(total_bedrooms),
        step=1
    )

with col2:

    population = st.number_input(
        "Population",
        value=int(population),
        step=1
    )

    households = st.number_input(
        "Households",
        value=int(households),
        step=1
    )

    median_income = st.number_input(
        "Median Income",
        value=float(median_income),
        step=0.1
    )

    ocean_proximity = st.selectbox(
        "Ocean Proximity",
        [
            "NEAR BAY",
            "<1H OCEAN",
            "INLAND",
            "NEAR OCEAN",
            "ISLAND"
        ],
        index=[
            "NEAR BAY",
            "<1H OCEAN",
            "INLAND",
            "NEAR OCEAN",
            "ISLAND"
        ].index(ocean_proximity)
    )

st.divider()

# -------------------- PREDICTION --------------------
if st.button("🔮 Predict House Price", use_container_width=True):

    input_df = pd.DataFrame([{
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "ocean_proximity": ocean_proximity
    }])

    prediction = model.predict(input_df)[0]

    st.success("Prediction Complete!")

    st.metric(
        label="🏠 Estimated House Price",
        value=f"${prediction:,.2f}"
    )

st.info(
    "💡 Try changing values like Median Income, Total Rooms, and Ocean Proximity to see how the prediction changes."
)