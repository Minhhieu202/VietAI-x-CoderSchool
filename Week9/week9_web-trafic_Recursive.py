import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,root_mean_squared_error

data = pd.read_csv("Time-series-datasets\web-traffic.csv")
data['date'] = pd.to_datetime(data['date'],dayfirst=True)

def create_ts_data(data,window_size):
    i = 1
    while i < window_size:
        data["users_{}".format(i)] = data["users"].shift(-i)
        i += 1
    data["target"] = data["users"].shift(-window_size)
    data = data.dropna(axis = 0)
    return data

# plot figure
# fig, ax = plt.subplots()
# ax.plot(data["date"], data["users"])
# ax.set_xlabel("Date")
# ax.set_ylabel("Users")
# plt.show()

window_size = 5
train_ratio = 0.8
data = create_ts_data(data, window_size)
x = data.drop(["date","target"],axis = 1)
y = data["target"]
sample = len(data)
x_train = x[:int(sample*train_ratio)]
y_train = y[:int(sample*train_ratio)]
x_test = x[int(sample*train_ratio):]
y_test = y[int(sample*train_ratio):]

model = Pipeline(steps=[
    ("standar", StandardScaler()),
    ("model", RandomForestRegressor())
])

model.fit(x_train,y_train)
y_predict = model.predict(x_test)

print("MAE = {}".format(mean_absolute_error(y_test,y_predict)))
# print("MSE = {}".format(mean_squared_error(y_test,y_predict)))
print("RMSE = {}".format(root_mean_squared_error(y_test,y_predict)))
print("r2 = {}".format(r2_score(y_test,y_predict)))