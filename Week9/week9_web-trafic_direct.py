import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import pickle

def create_ts_data(data, window_size, target_size):
    i = 1
    while i < window_size:
        data["users_{}".format(i)] = data["users"].shift(-i)
        i += 1
    i = 0
    while i < target_size:
        data["target_{}".format(i)] = data["users"].shift(-i-window_size)
        i += 1
    data = data.dropna(axis = 0)
    return data

data = pd.read_csv("Time-series-datasets\web-traffic.csv")

window_size = 5
target_size = 3
train_ratio = 0.8
data = create_ts_data(data,window_size,target_size)

targets = ["target_{}".format(i) for i in range(target_size)]
x = data.drop(["date"]+targets, axis=1)
y = data[targets]

num_sample = len(data)
x_train = x[:int(num_sample*train_ratio)]
y_train = y[:int(num_sample*train_ratio)]
x_test = x[int(num_sample*train_ratio):]
y_test = y[int(num_sample*train_ratio):]

models = [LinearRegression() for _ in range(target_size)]
for index, model in enumerate(models):
    model.fit(x_train,y_train["target_{}".format(index)])
mse = []
mae = []
r2 = []
for index, model in enumerate(models):
    y_predict = model.predict(x_test)
    mse.append(mean_squared_error(y_test["target_{}".format(index)],y_predict))
    mae.append(mean_absolute_error(y_test["target_{}".format(index)],y_predict))
    r2.append(r2_score(y_test["target_{}".format(index)],y_predict))

print(mse)
print(mae)
print(r2)