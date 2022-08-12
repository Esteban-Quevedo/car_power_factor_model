from sklearn.linear_model import LinearRegression
import pandas as pd
import bentoml


def manufacturer_replacement(dtf):
    manu_vals = {"Manufacturer": {'Acura': 1, 'Audi': 2, 'BMW': 3, 'Buick': 4, 'Cadillac': 5, 'Chevrolet': 6,
                                  'Chrysler': 7, 'Dodge': 8, 'Ford': 9, 'Honda': 10, 'Hyundai': 11, 'Infiniti': 12,
                                  'Jaguar': 13, 'Jeep': 14, 'Lexus': 15, 'Lincoln': 16, 'Mitsubishi': 17, 'Mercury': 18,
                                  'Mercedes-B': 19, 'Nissan': 20, 'Oldsmobile': 21, 'Plymouth': 22, 'Pontiac': 23,
                                  'Porsche': 24, 'Saab': 25, 'Saturn': 26, 'Subaru': 27, 'Toyota': 28, 'Volkswagen': 29,
                                  'Volvo': 30}}
    dtf.replace(manu_vals, inplace=True)
    return dtf


def vehicle_type_replacement(dtf):
    type_vals = {"Vehicle_type": {'Car': 1, 'Passenger': 2}}
    dtf.replace(type_vals, inplace=True)
    return dtf


def get_dataset():
    dtf = pd.read_csv('data/car_dataset.csv')
    return dtf


def process_data(dtf):
    dtf = manufacturer_replacement(dtf)
    dtf = vehicle_type_replacement(dtf)
    dtf = dtf.query('Price_in_thousands>0 & Price_in_thousands<50')
    dtf = dtf.query('Engine_size>0 or Engine_size<8')
    dtf = dtf.query('Horsepower>100')
    dtf = dtf.query('Curb_weight>0.5')
    dtf = dtf.query('Fuel_efficiency>10 & Fuel_efficiency<40')
    dtf = dtf.query('Power_perf_factor>25 & Power_perf_factor<150')
    dtf = dtf.drop('Sales_in_thousands', axis=1)
    dtf = dtf.drop('year_resale_value', axis=1)
    return dtf


def create_model():
    model = LinearRegression()
    return model


if __name__ == "__main__":
    df = get_dataset()
    training_data = process_data(df)
    # Defining Training Data
    trn_target = training_data.pop('Power_perf_factor')
    trn_data = training_data
    # Creating the model
    mod = create_model()
    # Fit the model
    mod.fit(X=trn_data, y=trn_target)
    # Save model to the BentoML local model store
    saved_model = bentoml.sklearn.save_model("car_power_factor_model", mod)
    print(f"Model saved: {saved_model}")
