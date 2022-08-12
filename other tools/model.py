from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
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
    dtf = pd.read_csv('car_dataset.csv')
    return dtf


def process_data(dtf):
    dtf = manufacturer_replacement(dtf)
    dtf = vehicle_type_replacement(dtf)
    dtf = dtf.query('Sales_in_thousands<500')
    dtf = dtf.query('year_resale_value>0 & year_resale_value<50')
    dtf = dtf.query('Price_in_thousands>0 & Price_in_thousands<50')
    dtf = dtf.query('Engine_size>0 or Engine_size<8')
    dtf = dtf.query('Horsepower>100')
    dtf = dtf.query('Curb_weight>0.5')
    dtf = dtf.query('Fuel_efficiency>10 & Fuel_efficiency<40')
    dtf = dtf.query('Power_perf_factor>25 & Power_perf_factor<150')
    dtf = dtf.drop('Sales_in_thousands', axis=1)
    dtf = dtf.drop('year_resale_value', axis=1)
    training_data = dtf.sample(frac=0.9, random_state=0)
    testing_data = dtf.drop(training_data.index)
    return training_data, testing_data


def create_model():
    model = LinearRegression()
    return model


def test(model, target, data, qty):
    predictions = model.predict(data)
    for y, y_pred in list(zip(target, predictions))[:qty]:
        print("Real Value: {:.3f} Estimate Value: {:.5f}".format(y, y_pred))
    error = np.sqrt(mean_squared_error(test_target, predictions))
    print(f"% Error: {error * 100}")


def predict(model, data):
    prediction = model.predict(data)
    return prediction


if __name__ == "__main__":
    df = get_dataset()
    trn_dts, test_dts = process_data(df)
    # Defining Training Data
    trn_target = trn_dts.pop('Power_perf_factor')
    trn_data = trn_dts

    # Creating the model
    mod = create_model()
    # Fit the model
    mod.fit(X=trn_data, y=trn_target)

    option = 0

    # Make some Predictions to compare
    if option == 0:
        # Defining Testing Data
        test_target = test_dts.pop('Power_perf_factor')
        test_data = test_dts
        test(mod, test_target, test_data, 5)

    # Save de model into local Bento
    elif option == 1:
        saved_model = bentoml.sklearn.save_model("car_mod", mod)
        print(f"Model saved: {saved_model}")

    # Make a prediction
    elif option == 2:
        new_car = pd.DataFrame(
            np.array([[1, 2, 21.5, 1.8, 140, 101.2, 67.3, 172.4, 2.639, 13.2, 28]]),
            columns=['Manufacturer',
                     'Vehicle_type', 'Price_in_thousands', 'Engine_size', 'Horsepower',
                     'Wheelbase', 'Width', 'Length', 'Curb_weight', 'Fuel_capacity',
                     'Fuel_efficiency'])
        result = predict(mod, new_car)
        print(f"The Power Perf Factor predicted to the car is: {result[0]}")
    else:
        raise ValueError("Please put a valid value to the option variable")
