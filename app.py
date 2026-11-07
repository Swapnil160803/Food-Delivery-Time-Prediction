
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="Food Delivery Time Prediction",
    page_icon="🚴",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

# --------------------------------
# LOAD TRAINED MODEL AND SCALER
# --------------------------------

@st.cache_resource
def load_artifacts():
    model = joblib.load(BASE_DIR / "xgb_best.pkl")
    scaler = joblib.load(BASE_DIR / "standard_scaler.pkl")
    return model, scaler

try:
    model, scaler = load_artifacts()
except Exception as e:
    st.error(f"Unable to load model or scaler: {e}")
    st.stop()


# --------------------------------
# EXACT 21 MODEL FEATURES
# --------------------------------

FEATURE_COLUMNS = [
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "distance_km",
    "order_hour",
    "Road_traffic_density",
    "Vehicle_condition",
    "multiple_deliveries",
    "Type_of_order_Drinks",
    "Type_of_order_Meal",
    "Type_of_order_Snack",
    "Type_of_vehicle_electric_scooter",
    "Type_of_vehicle_motorcycle",
    "Type_of_vehicle_scooter",
    "Festival_Yes",
    "City_Semi-Urban",
    "City_Urban",
    "Weatherconditions_Fog",
    "Weatherconditions_Sandstorms",
    "Weatherconditions_Stormy",
    "Weatherconditions_Sunny",
    "Weatherconditions_Windy"
]


# --------------------------------
# APPLICATION TITLE
# --------------------------------

st.title("🚴 Food Delivery Time Prediction")

st.write(
    "Predict the estimated food delivery time "
    "using Machine Learning Model."
)

st.info(
    "Enter the 12 delivery input features below "
    "to predict the delivery time in minutes."
)

# --------------------------------
# INPUT FORM - 12 FEATURES
# --------------------------------

with st.form("delivery_prediction_form"):

    st.subheader("1. Delivery Person Information")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Delivery Person Age",
            min_value=15,
            max_value=50,
            value=30
        )

        rating = st.number_input(
            "Delivery Person Ratings",
            min_value=1.0,
            max_value=5.0,
            value=4.7,
            step=0.1
        )

        vehicle_condition = st.selectbox(
            "Vehicle Condition",
            [0, 1, 2]
        )

    with col2:
        multiple_deliveries = st.selectbox(
            "Multiple Deliveries",
            [0, 1, 2, 3]
        )

        vehicle_type = st.selectbox(
            "Type of Vehicle",
            [
                "bicycle",
                "electric_scooter",
                "motorcycle",
                "scooter"
            ]
        )

    st.divider()

    st.subheader("2. Order Information")

    col3, col4 = st.columns(2)

    with col3:
        order_type = st.selectbox(
            "Type of Order",
            ["Buffet", "Drinks", "Meal", "Snack"]
        )

        festival = st.selectbox(
            "Festival",
            ["No", "Yes"]
        )

        order_time = st.time_input(
            "Order Time",
            value=pd.to_datetime("18:00").time()
        )

    with col4:
        city = st.selectbox(
            "City",
            ["Metropolitian", "Semi-Urban", "Urban"]
        )

        weather = st.selectbox(
            "Weather Conditions",
            [
                "Cloudy",
                "Fog",
                "Sandstorms",
                "Stormy",
                "Sunny",
                "Windy"
            ]
        )

        traffic = st.selectbox(
            "Road Traffic Density",
            ["Low", "Medium", "High", "Jam"]
        )

    st.divider()

    st.subheader("3. Delivery Route Information")

    distance_km = st.number_input(
        "Delivery Distance (KM)",
        min_value=0.1,
        max_value=100.0,
        value=5.0,
        step=0.1
    )

    st.caption(
        "Enter the restaurant-to-customer distance in kilometres."
    )

    submitted = st.form_submit_button(
        "Predict Delivery Time 🚀",
        use_container_width=True
    )


# --------------------------------
# PREDICTION
# --------------------------------

if submitted:

    # Convert order time into hour
    order_hour = order_time.hour

    # Traffic ordinal encoding
    traffic_mapping = {
        "Low": 0,
        "Medium": 1,
        "High": 2,
        "Jam": 3
    }

    # Create all 21 model features
    input_data = {

        "Delivery_person_Age": age,
        "Delivery_person_Ratings": rating,
        "distance_km": distance_km,
        "order_hour": order_hour,

        "Road_traffic_density": traffic_mapping[traffic],
        "Vehicle_condition": vehicle_condition,
        "multiple_deliveries": multiple_deliveries,

        # Order encoding
        "Type_of_order_Drinks": int(order_type == "Drinks"),
        "Type_of_order_Meal": int(order_type == "Meal"),
        "Type_of_order_Snack": int(order_type == "Snack"),

        # Vehicle encoding
        "Type_of_vehicle_electric_scooter":
            int(vehicle_type == "electric_scooter"),

        "Type_of_vehicle_motorcycle":
            int(vehicle_type == "motorcycle"),

        "Type_of_vehicle_scooter":
            int(vehicle_type == "scooter"),

        # Festival encoding
        "Festival_Yes": int(festival == "Yes"),

        # City encoding
        "City_Semi-Urban": int(city == "Semi-Urban"),
        "City_Urban": int(city == "Urban"),

        # Weather encoding
        "Weatherconditions_Fog": int(weather == "Fog"),

        "Weatherconditions_Sandstorms":
            int(weather == "Sandstorms"),

        "Weatherconditions_Stormy":
            int(weather == "Stormy"),

        "Weatherconditions_Sunny":
            int(weather == "Sunny"),

        "Weatherconditions_Windy":
            int(weather == "Windy")
    }

    # Convert input into DataFrame
    input_df = pd.DataFrame(
        [input_data],
        columns=FEATURE_COLUMNS
    )

    try:

        # Scale input using the original trained scaler
        scaled_input = scaler.transform(input_df)

        # Predict using trained XGBoost model
        prediction = model.predict(scaled_input)[0]

        prediction = max(0, float(prediction))

        st.success("Prediction completed successfully!")

        st.subheader("Predicted Delivery Time")

        st.metric(
            label="Estimated Delivery Time",
            value=f"{prediction:.2f} Minutes"
        )

        # Additional information
        st.write("### Delivery Summary")

        c1, c2, c3 = st.columns(3)

        c1.metric("Distance", f"{distance_km} KM")
        c2.metric("Traffic", traffic)
        c3.metric("Weather", weather)

        st.info(
            f"Estimated delivery time is "
            f"{prediction:.2f} minutes."
        )

        st.caption(
            "Note: This prediction is an estimate. "
            "Actual delivery time may vary."
        )

    except Exception as e:

        st.error(f"Prediction failed: {e}")


# --------------------------------
# ABOUT PROJECT
# --------------------------------

with st.expander("About this Project"):

    st.write("""
    ### Food Delivery Time Prediction

    This project uses Machine Learning to estimate
    food delivery time.

    **Technologies Used:**
    - Python
    - Pandas
    - NumPy
    - Scikit-learn
    - XGBoost
    - Streamlit

    **Machine Learning Algorithm:**
    XGBoost Regressor

    **Model Type:**
    Regression

    **Target Variable:**
    Time_taken(min)
    """)
