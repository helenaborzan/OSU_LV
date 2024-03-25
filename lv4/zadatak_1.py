from sklearn . model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import sklearn . linear_model as lm
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error


data = pd.read_csv("data_C02_emission.csv")
print(data.dtypes)
X = data[['Engine Size (L)', 'Cylinders', 'Fuel Consumption City (L/100km)','Fuel Consumption Hwy (L/100km)', 'Fuel Consumption Comb (L/100km)', 'Fuel Consumption Comb (mpg)']]
y = data['CO2 Emissions (g/km)']

X_train , X_test , y_train , y_test = train_test_split (X , y , test_size = 0.2 , random_state =5)

plt.scatter(X_train['Fuel Consumption City (L/100km)'], y_train, color="blue", label="Training data")
plt.scatter(X_test['Fuel Consumption City (L/100km)'], y_test, color="red", label="Testing data")
plt.xlabel("Fuel Consumption City (L/100km)")
plt.ylabel("C02 Emissions(g/km)")
plt.legend()
plt.show()


plt.hist(X_train['Fuel Consumption City (L/100km)'], color="blue", alpha=0.5)
plt.xlabel("Fuel Consumption City (L/100km)")
plt.title("Data before scaling")
plt.show()

sc = MinMaxScaler ()
X_train_scaled = sc.fit_transform(X_train)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = sc.transform(X_test)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

plt.hist(X_train_scaled['Fuel Consumption City (L/100km)'], color="red", alpha=0.5)
plt.xlabel("Fuel Consumption City (L/100km)")
plt.title("Data after scaling")
plt.show()

linearModel = lm.LinearRegression()
linearModel.fit(X_train_scaled , y_train)
print(f"Koeficijenti linearnog modela: {linearModel.coef_}")


y_test_prediction = linearModel.predict(X_test_scaled)
plt.scatter(X_test_scaled['Fuel Consumption City (L/100km)'], y_test, color="blue", label="Real values")
plt.scatter(X_test_scaled["Fuel Consumption City (L/100km)"], y_test_prediction, color="red", label="Predictions")
plt.xlabek("Fuel Consumption City (L/100km)")
plt.ylabel("C02 Emissions(g/km)")
plt.legend()
plt.show()


MSE = mean_squared_error(y_test, y_test_prediction)
print(f"Mean squared error(MSE): {MSE}")
MAE = mean_absolute_error(y_test, y_test_prediction)
print(f"Mean absolute error(MAE): {MAE}")
MAPE = mean_absolute_percentage_error(y_test, y_test_prediction)
print(f"Mean absolute percentage error(MAPE): {MAPE}")
