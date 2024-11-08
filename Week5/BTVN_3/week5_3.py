import pandas as pd

df = pd.read_csv('BTVN_3\dataset_3.csv')
#task1
"""
df['revenue'] = df['price'] * df['units_sold']
revenue = df[df['revenue']>100000]
print(revenue)
"""
#task2
top_unit_sold = df.nlargest(100,'units_sold')
top_unit_sold_10_rating = df[df['rating']>=df['rating'].quantile(0.9)]
task2 = top_unit_sold[top_unit_sold['product_id'].isin(top_unit_sold_10_rating['product_id'])]
print(task2)