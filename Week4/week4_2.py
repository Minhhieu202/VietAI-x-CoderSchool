import pandas as pd

df = pd.read_csv("iris.csv")

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
"""
task1_a = vd[['SepalLengthCm','SepalWidthCm']].mean()
print(task1_a)
print("----------------")

task1_b = vd2[['SepalLengthCm','SepalWidthCm']].mean()
print(task1_b)
print("----------------")

task1_c = vd3[['SepalLengthCm','SepalWidthCm']].mean()
print(task1_c)


task2_a = vd.nlargest(1,['SepalLengthCm','SepalWidthCm'])
print(task2_a)
print("----------------")

task2_b = vd2.nlargest(1,['SepalLengthCm','SepalWidthCm'])
print(task2_b)
print("----------------")


task2_c = vd3.nlargest(1,['SepalLengthCm','SepalWidthCm'])
print(task2_c)
print("----------------")
"""





