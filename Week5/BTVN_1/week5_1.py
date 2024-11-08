import pandas as pd 
import numpy as np

data = pd.read_csv('D:\AI\VietAI-x-CoderSchool\Week5\BTVN_1\dataset_1.csv')

#task1
"""
data['Population'] = data['Population'].str.replace(',', '').astype(int)
top_10_city_population = data.nlargest(10, 'Population')[['City','Population']]
bottom_10_city = data.nsmallest(10,'Population')[['City','Population']]

print(top_10_city_population)
print("----------------")
print(bottom_10_city)
"""
#task2
"""
top_cn_have3_city = data['Country'].value_counts()
top_cn_have3_city = top_cn_have3_city[top_cn_have3_city>3]
print(top_cn_have3_city)
"""
#task3
"""
top_5_country = data['Country'].value_counts()
print(top_5_country)
"""
#task4
"""
data['Population'] = data['Population'].str.replace(',', '').astype(int)
data['Area KM2'] = pd.to_numeric(data['Area KM2'],errors='coerce')
top20_Population = data.nlargest(20,'Population')[['City','Population']]
top20_Areakm2 = data.nlargest(20,'Area KM2')[['City','Area KM2']]

sort_population = top20_Population.sort_values(by='Population')
sort_area = top20_Areakm2.sort_values(by='Area KM2')

print(sort_area)
print("-------------------")
print(sort_population)

top_20_population_area = set(top20_Population).intersection(set(top20_Areakm2))
print(top_20_population_area)
#khong co quoc gia nao ca

"""
#task5
"""
data['Density  M2'] = data['Density  M2'].str.replace(',','').astype(int)
Population_Country = data.groupby('Country')['Density  M2'].mean()
print(Population_Country)
"""
#task6
data['Population'] = data['Population'].str.replace(',', '').astype(int)
countries_with_min_2_cities = data['Country'].value_counts()
countries_with_min_2_cities = countries_with_min_2_cities[countries_with_min_2_cities >= 2].index

filtered_df = data[data['Country'].isin(countries_with_min_2_cities)]
largest_city_per_country = filtered_df.loc[filtered_df.groupby('Country')['Population'].idxmax(), ['Country', 'City', 'Population']]

print("Thành phố có dân số lớn nhất của từng quốc gia (có ít nhất 2 thành phố):")
print(largest_city_per_country)



