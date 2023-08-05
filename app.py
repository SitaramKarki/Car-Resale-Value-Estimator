import streamlit as st
import pandas as pd
import pickle

# Load the pre-trained model
with open('LinearRegressionModel.pkl', 'rb') as file:
    model = pickle.load(file)

# Load your car dataset here (assuming you have a CSV file with columns: company, name, year, fuel_type, kms_driven, price)
df = pd.read_csv('cleaned_Car_Data.csv')

companies = sorted(df['company'].unique())
years = sorted(df['year'].unique(), reverse=True)
fuel_types = df['fuel_type'].unique()


def predict_car_price(selected_company, selected_car_model, selected_year, selected_fuel_type,
                      selected_number_of_km_driven):
    # Create a DataFrame with the user inputs
    data = pd.DataFrame(
        [[selected_car_model, selected_company, selected_year, selected_number_of_km_driven, selected_fuel_type]],
        columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])

    # Make predictions using the pre-trained model
    car_price_prediction = model.predict(data)

    return car_price_prediction[0]


def main():
    st.title("Car Resale Value Estimator")
    st.write("An Initiative By Situ Mobility Solutions Limited")

    selected_company = st.selectbox("Select the Company", companies)
    selected_car_model = st.selectbox("Select the Car Model", df[df['company'] == selected_company]['name'].unique())
    selected_year = st.selectbox("Select the Year of Purchase", years)
    selected_fuel_type = st.selectbox("Select the Fuel Type", df[df['name'] == selected_car_model]['fuel_type'].unique())
    selected_number_of_km_driven = st.number_input("Enter the Number of Kilometers Driven")

    if st.button("Estimate Car Price"):
        # Call the predict_car_price function to get the prediction
        car_price_prediction = predict_car_price(selected_company, selected_car_model, selected_year,
                                                 selected_fuel_type, selected_number_of_km_driven)

        # Display the prediction
        st.write(f"Estimated Car Price: Rs. {car_price_prediction:,.2f}")


if __name__ == "__main__":
    main()
