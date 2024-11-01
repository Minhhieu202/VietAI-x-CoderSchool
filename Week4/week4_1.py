import pandas as pd

df = pd.read_csv("Week4_1_data.csv")

task1 = df.groupby(['educational_level']).size()
print(task1)
print ("----------------------------")

task2 = df.nlargest(20,'annual_income')
print(task2)
print ("----------------------------")

task3 = df[(df['year_of_birth']> 1960) & (df['annual_income'] > 50000)]
print(task3)
print ("----------------------------")

task4 = df[(df['year_of_birth']> 1960) & (df['annual_income'] > 50000)].nlargest(20,'annual_income')
print(task4)
print ("----------------------------")

task5 = df[df['marital_status'].isin(['Married', 'Divorced'])]
print(task5)
print ("----------------------------")

task6 = df.groupby(['educational_level'])['annual_income'].mean()
print(task6)
print ("----------------------------")

task7 = df.groupby(['educational_level','marital_status'])['annual_income'].mean()
print(task7)