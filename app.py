import random
import streamlit as st
import pickle
import numpy as np
from facts_carbon import base_facts
from fpdf import FPDF
import base64
import pandas as pd

# Load model and scalers
model = pickle.load(open('model.pkl', 'rb'))
scaler_x = pickle.load(open('scaler_X.pkl', 'rb'))
scaler_y = pickle.load(open('scaler_y.pkl', 'rb'))

# Page Config
st.set_page_config(page_title="Carbon Emission Predictor", layout="centered")

# Custom CSS for Styling
st.markdown("""
    <style>
        body {
            background-color: #e8f5e9;
        }
        .main {
            background-color: #e8f5e9;
        }
        [data-testid="stHeader"] {
            background-color: #328E6E;
        }
        h1, h2, h3, h4 {
            color: #1b5e20;
        }
        b{
            color: #1b5e20;
        }
        a{
            color: #1b5e20;
        }
        p, label, .st-bc, .st-c7, .st-ce,i {
            color: #1a237e;
        }
        .stButton>button {
            background-color: #66bb6a;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 10px;
        }
        .stButton>button:hover {
            background-color: #388e3c;
        }
        .card {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 0px 15px #a5d6a7;
            margin-bottom: 20px;
        }
        
        div[data-baseweb="select"] > div {
            background-color: #d1e7dd;  /* Light green background */
            color: #0f5132;             /* Dark green text */
            border-radius: 5px;
        }
        div[data-baseweb="input"] input:focus {
            background-color: #000000 !important; /* Light blue background */
            color: #A4B465 !important;            /* Text color */
            border: 2px solid #00796b !important; /* Teal border */
            border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center;'>🌿 <br>CarbonChakra: <br>A Carbon Emission Prediction App</h1>", unsafe_allow_html=True)

# Tabs for Navigation
tab1, tab2, tab3 = st.tabs(["📌 About", "🎉 Fun Facts", "📈 Prediction"])

# ----------------------- TAB 1: ABOUT -----------------------
with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("🧠 About the Application")
    st.markdown("""
    This application predicts **Carbon Emission** based on user inputs using a **Support Vector Machine (SVM)** model.  
    Built with 💡 **scikit-learn** for ML and **Streamlit** for the interactive UI. 

    <br>
    <b>Carbon Footprint:</b>
    <br>
    <p>A <b>carbon footprint</b> refers to the total amount of greenhouse gases (GHGs), primarily carbon dioxide (CO₂), emitted directly or indirectly by human activities. It is measured in terms of equivalent tons of CO₂, encompassing activities like transportation, energy consumption, and food production. The carbon footprint is an essential indicator of an individual’s, organization’s, or product’s contribution to climate change. <b>Reducing the carbon footprint is crucial for mitigating global warming and minimizing environmental harm.</b></p>
    
    <b>Carbon Emissions:</b>
    <br>
    Carbon emissions specifically refer to the release of carbon dioxide and other GHGs into the atmosphere. These emissions result from the **burning of fossil fuels (coal, oil, and natural gas)**, **deforestation**, and **industrial processes**. Carbon emissions are the primary cause of the greenhouse effect, which leads to rising global temperatures and climate instability. Major sources include **vehicles**, **power plants**, **agriculture**, and **deforestation**.
    
    <b>Impact and Importance:</b>
    <br>
    Carbon footprint and carbon emissions are key to understanding **climate change**. Human activities release **greenhouse gases** that warm the Earth, causing rising sea levels, extreme weather, and ecosystem disruption. Global efforts like the **Paris Agreement** aim to reduce these emissions and limit temperature rise to below **1.5°C**.    
    
    <b>Reducing Carbon Footprint and Emissions:</b>
    <br>
    To mitigate climate change, individuals and organizations can reduce their carbon footprint by adopting sustainable practices:
    
    <b>Energy Efficiency:</b> 
    Switching to renewable energy sources like solar and wind power.
    
    <b>Transportation:</b> 
    Opting for electric vehicles, public transport, or biking.
    
    <b>Sustainable Diet:</b> 
    Reducing meat consumption and focusing on plant-based diets.
    
    <b>Waste Reduction:</b> Recycling and reducing waste production.
    
    Conservation: Planting trees to absorb CO₂ and protect forests.
    <br>
    <b>Technologies:</b> <b> Python, Streamlit, SVM </b> 
    <br>
    <b>Developer:</b> <b> Ajitesh Channa 🚀 </b>  
    <b>Contact:</b> <b> ac.ajiteshchanna@gmail.com ✉️ </b>
    <br>
    <i>Let's calculate your carbon footprint and raise awareness! 🌏</i>
    """, unsafe_allow_html=True)
    df = pd.read_csv("Carbon Emission.csv")
    st.write("Original Dataset Preview:")
    st.dataframe(df)

    df_pure = pd.read_csv("Updated_Dataset.csv")
    st.write("Modified Dataset (According to Problem Statement) Preview:")
    st.dataframe(df_pure)
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------- TAB 2: FUN FACTS -----------------------
with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("💡 Fun Facts about Carbon Emission")
    current_facts = []
    while len(current_facts)<5:
        c_f = random.choice(base_facts)
        if c_f not in current_facts:
            current_facts.append(c_f)

    for fact in current_facts:
        st.success(fact)
    st.button("New Facts!")
    st.markdown('</div>', unsafe_allow_html=True)


# ----------------------- TAB 3: PREDICTION -----------------------
with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("📈 Predict Your Carbon Emission")

    name = st.text_input("👤 What is your name?")

    # Categorical Inputs
    body_type = st.selectbox("💪 Select your Body Type", ["Underweight", "Normal", "Overweight", "Obese"])
    gender = st.selectbox("🚻 Select your Gender", ["Male", "Female"])
    diet = st.selectbox("🥗 Select your Diet", ["Vegan", "Vegetarian", "Pescatarian", "Omnivore"])
    shower = st.selectbox("🚿 How often do you shower?", ["Less Frequently", "Daily", "Twice a Day", "More Frequently"])
    heating_source = st.selectbox("🔥 Heating Source?", ["Coal", "Wood", "Natural Gas", "Electricity"])
    transport = st.selectbox("🚗 Transportation Mode",["Public","Bicycle","Private"])
    social_activity = st.selectbox("🎭 Social Activity", ["Never", "Often", "Sometimes"])
    air_travel = st.selectbox("🛫 Air Travel Frequency", ["Very Frequently", "Frequently", "Rarely","Never"])
    waste_bag_size = st.selectbox("🗑️ Waste Bag Size", ["Extra Large", "Large", "Medium","Small"])
    energy_efficiency = st.selectbox("💡 Use Energy Efficient Practices?",["No","Sometimes","Yes"])
    cooking = st.selectbox("🍳 Cooking Appliance Used", ["Oven", "Microwave", "Grill","Airfryer","Stove"])
    recycle = st.selectbox("♻️ Recycled Waste Type",["Metal","Paper","Glass","Plastic"])

    # Numeric Inputs
    monthly_grocery_bill = st.number_input("🛒 Monthly Grocery Bill (in ₹)",min_value=0)
    vehicle_monthly = st.number_input("🚙 Vehicle Monthly Distance (km)",min_value=0)
    waste_bag_count = st.number_input("♻️ Waste Bags Used Weekly",min_value=0)
    tv_hour = st.number_input("📺 Daily Screen Time (TV/PC) in hours",min_value=0)
    new_clothes = st.number_input("👕 New Clothes Bought per Month",min_value=0)
    internet_hour = st.number_input("🌐 Internet Usage Daily (in hours)",min_value=0)

    # Encodings
    gender_male = int(gender == "Male")
    gender_female = int(gender == "Female")

    body = ["Underweight", "Normal", "Overweight", "Obese"].index(body_type)
    diet_type = ["Vegan", "Vegetarian", "Pescatarian", "Omnivore"].index(diet)
    shower_cat = ["Less Frequently", "Daily", "Twice a Day", "More Frequently"].index(shower)
    heating = ["Electricity", "Natural Gas", "Wood", "Coal"].index(heating_source)
    transport_cat = ["Bicycle", "Public", "Private"].index(transport)
    social = ["Never", "Sometimes", "Often"].index(social_activity)
    air = ["Never", "Rarely", "Frequently", "Very Frequently"].index(air_travel)
    bag_size = ["Small", "Medium", "Large", "Extra Large"].index(waste_bag_size)

    energy_efficiency_no = int(energy_efficiency == "No")
    energy_efficiency_sometimes = int(energy_efficiency == "Sometimes")
    energy_efficiency_yes = int(energy_efficiency == "Yes")

    cooking_oven = int(cooking == "Oven")
    cooking_stove = int(cooking == "Stove")
    cooking_grill = int(cooking == "Grill")
    cooking_microwave = int(cooking == "Microwave")
    cooking_airfryer = int(cooking == "Airfryer")

    recycle_paper = int(recycle == "Paper")
    recycle_plastic = int(recycle == "Plastic")
    recycle_metal = int(recycle == "Metal")
    recycle_glass = int(recycle == "Glass")

    input_data = np.array([[body,diet_type,shower_cat,heating,transport_cat,social,
                            monthly_grocery_bill,air,vehicle_monthly,bag_size,waste_bag_count,
                            tv_hour,new_clothes,internet_hour,
                            gender_female,gender_male,
                            energy_efficiency_no,energy_efficiency_sometimes,energy_efficiency_yes,
                            cooking_oven,cooking_microwave,cooking_grill,cooking_airfryer,cooking_stove,
                            recycle_metal,recycle_paper,recycle_glass,recycle_plastic]])


    def generate_pdf(name, input_data, prediction,message):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.set_text_color(34, 139, 34)  # Forest Green Title
        pdf.cell(200, 10, txt="Carbon Emission Report", ln=True, align='C')

        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Arial", '', 12)
        pdf.ln(10)
        pdf.cell(200, 10, txt=f"Name: {name}", ln=True)
        pdf.cell(200, 10, txt=f"Estimated Carbon Emission: {prediction:.2f} grams/km", ln=True)
        pdf.cell(200, 10, txt=f"Depicts: {message}", ln=True)
        pdf.ln(10)
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, txt="Input Summary:", ln=True)
        pdf.set_font("Arial", '', 12)

        fields = ["Body Type", "Diet", "Shower Habit", "Heating Source", "Transport", "Social Activity",
                  "Monthly Grocery Bill", "Air Travel", "Vehicle Monthly Distance", "Waste Bag Size",
                  "Waste Bag Count", "TV Hours", "New Clothes per Month", "Internet Hours", "Gender",
                  "Energy Efficiency", "Cooking Method", "Recycling Type"]

        values = [body_type, diet, shower, heating_source, transport, social_activity,
                  monthly_grocery_bill, air_travel, vehicle_monthly, waste_bag_size,
                  waste_bag_count, tv_hour, new_clothes, internet_hour, gender,
                  energy_efficiency, cooking, recycle]

        for f, v in zip(fields, values):
            pdf.cell(200, 10, txt=f"{f}: {v}", ln=True)

        return pdf

    if st.button("🔍 Predict My Emission"):
        if name == "":
            st.error("Kindly enter your name first!")
        else:
            scaled_input = scaler_x.transform(input_data)
            prediction_scaled = model.predict(scaled_input)
            prediction = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1))
            result = prediction[0][0]
            message = ""
            if result<1000:
                message = f"**Label**: Excellent\n**Remark**: Sustainable choices (e.g., shared EV, train)"
            elif result<2000:
                message = f"**Label**: Moderate\n**Remark**: Carpool, fuel-efficient vehicles"
            elif result<3000:
                message = f"**Label**: High Impact\n**Remark**: Personal car travel, short-haul flights"
            elif result<5000:
                message = f"**Label**: Very High Impact\n**Remark**: SUV use, long drives with poor mileage"
            else:
                message =f"**Label**: Severely Unsustainable\n**Remark**: Flights, luxury vehicles, solo long-distance travel"
            st.success(f"**{name}**, your 🌱 Estimated Carbon Emission: **{result:.2f} grams/km**")
            st.success(message)
            message = message.replace("**","")
            pdf = generate_pdf(name, input_data, prediction[0][0],message)
            pdf_file_path = "emission_report.pdf"
            pdf.output(pdf_file_path)

            with open("emission_report.pdf", "rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode('utf-8')
                pdf_display = f'<a href="data:application/pdf;base64,{base64_pdf}" download="carbon_emission_report.pdf"><button style="background-color:#4CAF50;border:none;color:white;padding:10px 24px;text-align:center;text-decoration:none;display:inline-block;font-size:16px;border-radius:12px;font-weight:bold;">📄 Download Report (PDF)</button></a>'
                st.markdown(pdf_display, unsafe_allow_html=True)


    st.markdown('</div>', unsafe_allow_html=True)


