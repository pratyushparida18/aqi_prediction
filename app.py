import streamlit as st
import hopsworks
import joblib
import numpy as np

st.title("AQI Prediction App")

# Input fields
pm25 = st.number_input("pm25")
pm10 = st.number_input("pm10")
o3 = st.number_input("o3")
no2 = st.number_input("no2")
so2 = st.number_input("so2")
co = st.number_input("co")

# Submit button
if st.button("Submit"):
    project = hopsworks.login(api_key_value="1NYjeRL5YLXIyVT2.k0sa3VLtHI2Wg9gDSptQ67wIAA9FvG3QuKpJWMmtl2j5zPjxLLLiMdnmRtIkT4Oe")
    mr = project.get_model_registry()
    model = mr.get_model(
        name="aqi_prediction_model",
        version=1
    )
    saved_model_dir = model.download()

    retrieved_xgboost_model = joblib.load(saved_model_dir + "/xgboost_aqi_prediction_model.pkl")
    # Make prediction
    new_data = np.array([[pm25, pm10, o3, no2, so2, co]])
    prediction = retrieved_xgboost_model.predict(new_data)

    # Display prediction
    st.subheader("Prediction")
    st.write(prediction)