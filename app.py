import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st
import datetime

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Car Price Predictor",
    layout="wide",
    page_icon="🚗"
)

# ---------------- LOAD MODEL ----------------
model = pk.load(open("model.pkl", "rb"))
cars_data = pd.read_csv("Cardetails.csv")

# ---------------- BRAND CLEAN ----------------
def get_brand(name):
    return name.split(' ')[0]

cars_data['name'] = cars_data['name'].apply(get_brand)

# ---------------- UI STYLE ----------------
st.markdown("""
    <style>
        .main {
            background-color: #f5f7fb;
        }
        .block-container {
            padding-top: 2rem;
        }
        .title {
            font-size: 40px;
            font-weight: 700;
            color: #1f4e79;
        }
        .sub {
            font-size: 18px;
            color: gray;
        }
        .card {
            background: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🚗 Car Price Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">AI-powered intelligent car valuation</div>', unsafe_allow_html=True)

st.write("")

# ---------------- INPUT UI ----------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🚘 Basic Info")
    name = st.selectbox("Brand", cars_data['name'].unique())
    fuel = st.selectbox("Fuel Type", cars_data['fuel'].unique())
    seller_type = st.selectbox("Seller Type", cars_data['seller_type'].unique())

with col2:
    st.markdown("### ⚙️ Vehicle Specs")
    transmission = st.selectbox("Transmission", cars_data['transmission'].unique())
    owner = st.selectbox("Owner Type", cars_data['owner'].unique())
    engine = st.slider("Engine CC", 700, 5000, 1200)

with col3:
    st.markdown("### 📊 Usage Info")
    year = st.slider("Manufactured Year", 1994, 2024, 2015)
    km_driven = st.slider("KM Driven", 0, 200000, 50000)
    mileage = st.slider("Mileage", 10, 40, 18)
    max_power = st.slider("Max Power", 0, 200, 80)
    seats = st.slider("Seats", 5, 10, 5)

st.write("---")

# ---------------- PREDICT BUTTON ----------------
if st.button("🔮 Predict Price"):

    input_df = pd.DataFrame(
        [[name, year, km_driven, fuel, seller_type,
          transmission, owner, mileage, engine, max_power, seats]],
        columns=['name','year','km_driven','fuel','seller_type',
                 'transmission','owner','mileage','engine','max_power','seats']
    )

    # ---------------- ENCODING ----------------
    input_df['owner'] = input_df['owner'].replace({
        'First Owner':1,
        'Second Owner':2,
        'Third Owner':3,
        'Fourth & Above Owner':4,
        'Test Drive Car':5
    })

    input_df['fuel'] = input_df['fuel'].replace({
        'Diesel':1, 'Petrol':2, 'LPG':3, 'CNG':4
    })

    input_df['seller_type'] = input_df['seller_type'].replace({
        'Individual':1, 'Dealer':2, 'Trustmark Dealer':3
    })

    input_df['transmission'] = input_df['transmission'].replace({
        'Manual':1, 'Automatic':2
    })

    input_df['name'] = input_df['name'].replace({
        'Maruti':1,'Skoda':2,'Honda':3,'Hyundai':4,'Toyota':5,
        'Ford':6,'Renault':7,'Mahindra':8,'Tata':9,'Chevrolet':10,
        'Datsun':11,'Jeep':12,'Mercedes-Benz':13,'Mitsubishi':14,
        'Audi':15,'Volkswagen':16,'BMW':17,'Nissan':18,'Lexus':19,
        'Jaguar':20,'Land':21,'MG':22,'Volvo':23,'Daewoo':24,
        'Kia':25,'Fiat':26,'Force':27,'Ambassador':28,
        'Ashok':29,'Isuzu':30,'Opel':31
    })

    # ---------------- PREDICTION ----------------
    prediction = model.predict(input_df)[0]

    st.success("🎯 Prediction Completed Successfully!")

    # ---------------- RESULT CARD ----------------
    st.markdown("""
        <div class="card">
            <h2>💰 Estimated Car Price</h2>
        </div>
    """, unsafe_allow_html=True)

    st.metric(label="Predicted Price", value=f"₹ {prediction:,.0f}")

    st.write("")

    colA, colB = st.columns(2)

    with colA:
        st.info(f"📉 Lower Range: ₹ {prediction*0.9:,.0f}")

    with colB:
        st.info(f"📈 Upper Range: ₹ {prediction*1.1:,.0f}")

    # ---------------- HISTORY ----------------
    input_df["prediction"] = prediction
    input_df["time"] = datetime.datetime.now()

    try:
        old = pd.read_csv("history.csv")
        input_df = pd.concat([old, input_df])
    except:
        pass

    input_df.to_csv("history.csv", index=False)

    st.toast("Saved to history ✔")