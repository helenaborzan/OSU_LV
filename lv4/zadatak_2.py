from sklearn . model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import sklearn . linear_model as lm
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn . preprocessing import OneHotEncoder


data = pd.read_csv("data_C02_emission.csv")
ohe = OneHotEncoder ()
X_encoded = pd.DataFrame(ohe.fit_transform(data[['Fuel Type']]).toarray())
data = data.join(X_encoded)
data.columns = ['Make', 'Model', 'Vehicle Class', 'Engine Size (L)', 'Cylinders', 'Transmission','Fuel Type','Fuel Consumption City (L/100km)','Fuel Consumption Hwy (L/100km)', 'Fuel Consumption Comb (L/100km)', 'Fuel Consumption Comb (mpg)', 'CO2 Emissions (g/km)', 'Fuel Type 0', 'Fuel Type 1', 'Fuel Type 2', 'Fuel Type 3']

X = data.drop('CO2 Emissions (g/km)', axis=1)
print(data.head(10))
y = data['CO2 Emissions (g/km)'].copy()


X_train , X_test , y_train , y_test = train_test_split (X , y , test_size = 0.2 , random_state =5)
X_train = X_train[['Engine Size (L)', 'Cylinders', 'Fuel Consumption City (L/100km)','Fuel Consumption Hwy (L/100km)', 'Fuel Consumption Comb (L/100km)', 'Fuel Consumption Comb (mpg)', 'Fuel Type 0', 'Fuel Type 1', 'Fuel Type 2', 'Fuel Type 3']]
X_test = X_test[['Engine Size (L)', 'Cylinders', 'Fuel Consumption City (L/100km)','Fuel Consumption Hwy (L/100km)', 'Fuel Consumption Comb (L/100km)', 'Fuel Consumption Comb (mpg)', 'Fuel Type 0', 'Fuel Type 1', 'Fuel Type 2', 'Fuel Type 3']]

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


plt.hist(X_train['Fuel Consumption City (L/100km)'], color="red", alpha=0.5)
plt.xlabel("Fuel Consumption City (L/100km)")
plt.title("Data after scaling")
plt.show()

linearModel = lm.LinearRegression()
linearModel.fit(X_train, y_train)
print(f"Koeficijenti linearnog modela: {linearModel.coef_}")


y_test_prediction = linearModel.predict(X_test)
plt.scatter(X_test['Fuel Consumption City (L/100km)'], y_test, color="blue", label="Real values")
plt.scatter(X_test["Fuel Consumption City (L/100km)"], y_test_prediction, color="red", label="Predictions")
plt.xlabel("Fuel Consumption City (L/100km)")
plt.ylabel("C02 Emissions(g/km)")
plt.legend()
plt.show()


MSE = mean_squared_error(y_test, y_test_prediction)
print(f"Mean squared error(MSE): {MSE}")
MAE = mean_absolute_error(y_test, y_test_prediction)
print(f"Mean absolute error(MAE): {MAE}")
MAPE = mean_absolute_percentage_error(y_test, y_test_prediction)
print(f"Mean absolute percentage error(MAPE): {MAPE}")