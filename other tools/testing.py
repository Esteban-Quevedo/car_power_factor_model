import bentoml

iris_clf_runner = bentoml.sklearn.get("car_power_factor_model:latest").to_runner()
iris_clf_runner.init_local()
result = iris_clf_runner.predict.run([[1, 2, 21.5, 1.8, 140, 101.2, 67.3, 172.4, 2.639, 13.2, 28]])
print(f"The prediction for the values that you entered is: {result[0]}")
