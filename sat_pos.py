import pandas as pd
from pycaret.regression import load_model, predict_model
import streamlit as st
import plotly.express as px
import numpy as np

# Set page configurations
st.set_page_config(page_title="Satellite Position Predictor", layout="centered")

# 1. Load the trained PyCaret model (cached so it only loads once)
@st.cache_resource
def get_model():
    # Make sure model is in the same directory as this script
    return load_model("sat_pos")

model = get_model()

st.title("🏥 Satellite Position Prediction Dashboard")
st.write("Fill out the product,client and worker realted details below to check the Satellite Position prediction.")

# 2. Build the Form Interface
with st.form("prediction_form"):
    st.subheader("Decimal number Inputs")
    col1,col2 = st.columns(2)
    
    with col1:
        X_Position = st.number_input("X_Position", min_value=0.0, value=98.58666575005762)
        Velocity = st.number_input("Velocity", min_value=0.0, value=7.5511267713932995)
        Altitude = st.number_input("Altitude", min_value=-100.0, value=-28.0)
        Fuel_Level = st.number_input("Fuel_Level", min_value=-100.0, value=-20.15839350086838)
     
    with col2:        
        Signal_Strength = st.number_input("Signal_Strength", min_value=0.0, value=14.817947157134617)
        Battery_Temp = st.number_input("Battery_Temp", min_value=0.0, value=1.0384929611720055)
        Solar_Exposure = st.number_input("Solar_Exposure", min_value=-100.0, value=-5.760837892713907)

    # Submit button for the form
    submit_button = st.form_submit_button("Predict Satellite Position")

# 3. Handle Prediction Logic upon form submission
if submit_button:
    # Compile the form inputs into a dictionary matching your PyCaret model's features
    input_data = {'X_Position': X_Position,'Velocity': Velocity,'Altitude': Altitude,'Fuel_Level': Fuel_Level,'Signal_Strength': Signal_Strength,'Battery_Temp': Battery_Temp,'Solar_Exposure': Solar_Exposure}    
    
    # Convert input dict to DataFrame
    df = pd.DataFrame([input_data])
    
    with st.spinner("Calculating risk..."):
        # Make the prediction using PyCaret
        #predictions = predict_model(model, data=df,round=2,raw_score=True)
        predictions = predict_model(model, data=df,round=2)
        prediction_label = predictions["prediction_label"].iloc[0]
        # prediction_score_1 = predictions["prediction_score_1"].iloc[0]
        # prediction_score_0 = predictions["prediction_score_0"].iloc[0]
        
        # Display the result to the user
        st.success("### Prediction Complete!")
        st.metric(label="Satellite Position Result", value=f"Position: {prediction_label}")
        # st.metric(label="Risk Status Result", value=f"Class: {prediction_label}")
        # st.metric(label="Prediction Confidence Scores", value=f"Class 0 Score:{prediction_score_0}, Class 1 Score:{prediction_score_1}")



# 3. Create a clean two-column layout
control_panel, visual_panel = st.columns([1, 2], gap="large")

# Left Column: Input Widgets (Sliders, Dropdowns, Toggles)
with control_panel:
    st.subheader("🎛️ Adjust Variables")
    
    # Define variables interactively
    X_Position = st.slider("X_Position", min_value=0.0, value=98.58666575005762, step=0.1)
    Velocity = st.slider("Velocity", min_value=0.0, value=7.5511267713932995, step=0.1)
    Altitude = st.slider("Altitude", min_value=-100.0, value=-28.0, step=0.1)
    Fuel_Level = st.slider("Fuel_Level", min_value=-100.0, value=-20.15839350086838, step=0.1)
    Signal_Strength = st.slider("Signal_Strength", min_value=0.0, value=14.817947157134617, step=0.1)
    Battery_Temp = st.slider("Battery_Temp", min_value=0.0, value=1.0384929611720055, step=0.1)
    Solar_Exposure = st.slider("Solar_Exposure", min_value=-100.0, value=-5.760837892713907, step=0.1)    
    
    # Dropdown or selection inputs are also fully supported
    #category_option = st.selectbox("Location Zone Type", options=["Urban", "Suburban", "Rural"])

# Right Column: Instant Live Output & Charts
with visual_panel:
    st.subheader("🔮 Real-Time Prediction")
    
    # Map inputs into the exact shape/features your pre-trained model expects
    # (Fill in additional static feature values if your model uses more columns)
    input_features = np.array([[X_Position, Velocity, Altitude, Fuel_Level,Signal_Strength,Battery_Temp,Solar_Exposure, 98.58, 7.55,-28.0,-20.15,14.81,1.03,-5.76]])

    # Compile the form inputs into a dictionary matching your PyCaret model's features
    input_data = {'X_Position': X_Position,'Velocity': Velocity,'Altitude': Altitude,'Fuel_Level': Fuel_Level,'Signal_Strength': Signal_Strength,'Battery_Temp': Battery_Temp,'Solar_Exposure': Solar_Exposure}    
    
    # Convert input dict to DataFrame
    df = pd.DataFrame([input_data])     
    
    # Generate live inference
    predictions = predict_model(model, data=df,round=2)
    live_prediction = predictions["prediction_label"].iloc[0]
    
    # Display the result prominently using a Metric widget
    st.metric(
        label="Predicted Target Value", 
        value=f"${live_prediction:,.2f}" if isinstance(live_prediction, (int, float)) else str(live_prediction)
    )
    
    # 4. Generate Interactive Visualization based on active values
    st.write("### Current Feature Input Values")
    chart_data = pd.DataFrame({
        "Variable": ['X_Position', 'Velocity', 'Altitude', 'Fuel_Level','Signal_Strength','Battery_Temp','Solar_Exposure'],
        "Selected Value": [X_Position, Velocity, Altitude, Fuel_Level,Signal_Strength,Battery_Temp,Solar_Exposure]
    })
    
    # # Compile the form inputs into a dictionary matching your PyCaret model's features
    # input_data = {'X_Position': X_Position,'Velocity': Velocity,'Altitude': Altitude,'Fuel_Level': Fuel_Level,'Signal_Strength': Signal_Strength,'Battery_Temp': Battery_Temp,'Solar_Exposure': Solar_Exposure}    
    
    # # Convert input dict to DataFrame
    # chart_data = pd.DataFrame([input_data])    
    
    # Plotly bar chart that updates on every slider adjustment
    fig = px.bar(
        chart_data, 
        x="Variable", 
        y="Selected Value", 
        color="Red",
        text="Selected Value",
        title="Active Simulation Inputs"
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

