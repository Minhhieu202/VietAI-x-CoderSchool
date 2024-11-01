import pandas as pd

df = pd.read_csv("Iris.csv")
print(df)

print("---Iris-setosa'r---")
vd = df[df['Species'].isin(['Iris-setosa'])]
print(vd)
print("----------------")

print("---Iris-versicolor---")
vd2 = df[df['Species'].isin(['Iris-versicolor'])]
print(vd2)
print("----------------")

print("---Iris-virginica---")
vd3 = df[df['Species'].isin(['Iris-virginica'])]
print(vd3)
print("----------------")

task1_a = vd.nlargest(10,'SepalLengthCm')
print(task1_a)
print("-----------------")

task1_b = vd2.nlargest(10,'SepalLengthCm')
print(task1_b)
print("-----------------")

task1_c = vd3.nlargest(10,'SepalLengthCm')
print(task1_c)
print("-----------------")



