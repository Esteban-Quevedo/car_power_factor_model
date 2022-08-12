from bentoml.io import NumpyNdarray, JSON
from pydantic import BaseModel
import pandas as pd
import numpy as np
import bentoml

car_mod_runner = bentoml.sklearn.get("car_power_factor_model:latest").to_runner()

svc = bentoml.Service("car_power", runners=[car_mod_runner])


class Customer(BaseModel):
    Manufacturer: int = 1
    Vehicle_type: int = 2
    Price_in_thousands: float = 21.5
    Engine_size: float = 1.8
    Horsepower: int = 140
    Wheelbase: float = 101.2
    Width: float = 67.3
    Length: float = 172.4
    Curb_weight: float = 2.639
    Fuel_capacity: float = 13.2
    Fuel_efficiency: int = 28


@svc.api(input=JSON(pydantic_model=Customer), output=NumpyNdarray())
def predict(data: Customer) -> np.array:
    df = pd.DataFrame(data.dict(), index=[0])
    result = car_mod_runner.predict.run(df)
    return np.array(result)
