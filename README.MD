


import pandas as pd
import numpy as np
import pickle as pk 
import streamlit as st 

model = pk.load(open('model.pkl','rb'))

st.header('Car Price Prediction ML Model') 

cars_data = pd.read_csv('Cardetails.csv')

def get_brand_name(car_name):
    car_name = car_name.split(' ')[0]
    return car_name.strip(' ') 

cars_data['name'] = cars_data['name'].apply(get_brand_name) 

name = st.selectbox('Select Car Brand', cars_data['name'].unique())  
year = st.slider('Car Manufactured Year', 1994,2024)
km_driven = st.slider('No of kms Driven', 11,200000)

fuel = st.selectbox('Fuel type', cars_data['fuel'].unique())  
seller_type = st.selectbox('Seller type', cars_data['seller_type'].unique())  
transmission = st.selectbox('Transmission type', cars_data['transmission'].unique())  

owner = st.selectbox('Seller type', cars_data['owner'].unique())  

mileage = st.slider('Car Mileage', 10,40)

engine = st.slider('Engine CC', 700, 5000)

max_power = st.slider('Max Power', 0 ,200)

seats = st.slider('No of Seats', 5, 10 )

if st.button("Predict"):

    input_data_model = pd.DataFrame(
        [[name, year, km_driven, fuel, seller_type,
          transmission, owner, mileage, engine, max_power, seats]],
        
        columns=['name','year','km_driven','fuel','seller_type',
                 'transmission','owner','mileage','engine','max_power','seats']
    )

   

    input_data_model['owner'] = cars_data['owner'].replace(
    ['First Owner', 'Second Owner', 'Third Owner',
     'Fourth & Above Owner', 'Test Drive Car'],
    
    [1, 2, 3, 4, 5]
).astype(int)    
    

    # FUEL
input_data_model['fuel'] = cars_data['fuel'].replace({
    'Diesel': 1,
    'Petrol': 2,
    'LPG': 3,
    'CNG': 4
})

# SELLER TYPE
input_data_model['seller_type'] = cars_data['seller_type'].replace({
    'Individual': 1,
    'Dealer': 2,
    'Trustmark Dealer': 3
})

# TRANSMISSION
input_data_model['transmission'] = cars_data['transmission'].replace({
    'Manual': 1,
    'Automatic': 2
})

# NAME (safe way)
input_data_model['name'] = cars_data['name'].str.split().str[0]

input_data_model['name'] = cars_data['name'].replace({
    'Maruti': 1, 'Skoda': 2, 'Honda': 3, 'Hyundai': 4, 'Toyota': 5,
    'Ford': 6, 'Renault': 7, 'Mahindra': 8, 'Tata': 9, 'Chevrolet': 10,
    'Datsun': 11, 'Jeep': 12, 'Mercedes-Benz': 13, 'Mitsubishi': 14,
    'Audi': 15, 'Volkswagen': 16, 'BMW': 17, 'Nissan': 18, 'Lexus': 19,
    'Jaguar': 20, 'Land': 21, 'MG': 22, 'Volvo': 23, 'Daewoo': 24,
    'Kia': 25, 'Fiat': 26, 'Force': 27, 'Ambassador': 28,
    'Ashok': 29, 'Isuzu': 30, 'Opel': 31
})







car_price = model.predict(input_data_model)

st.markdown('Car Price is going to be ' + str(car_price[0]))

