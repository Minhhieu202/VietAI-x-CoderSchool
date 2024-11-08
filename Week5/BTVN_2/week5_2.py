import pandas as pd

df = pd.read_csv('BTVN_2\dataset_2.csv')

#task1
"""
top100_mathscore = df.nlargest(100,'math score')
top100_readingscore = df.nlargest(100,'reading score')
top100_readingandmath = df[df.index.isin(top100_mathscore.index) & df.index.isin(top100_readingscore.index)]

print(top100_readingandmath.index[:10])
"""
#task2
"""
top20_writingscore = df.nlargest(20,'writing score')[['race/ethnicity','writing score']]
task2 = top20_writingscore.index[:20]
print(task2)
"""
#task3
"""
studentinA = df[df['race/ethnicity'] == 'group A']['parental level of education'].value_counts()
print(studentinA)
"""
#task4
"""
df['average_score'] = df[['math score','reading score','writing score']].mean(axis=1)
average_by_parental = df.groupby('parental level of education')['average_score'].mean()
print(average_by_parental)
"""
#task5
"""
df['average_score'] = df[['math score','reading score','writing score']].mean(axis=1)
average_by_lunch = df.groupby('lunch')['average_score'].mean()
print(average_by_lunch)
"""
#task6
"""
top10_mathscore = df.groupby('race/ethnicity')['math score'].nlargest(10)
print(top10_mathscore)
"""
#task7
df['average_score'] = df[['math score','reading score','writing score']].mean(axis=1)
average_test_preparation_course = df.groupby(['race/ethnicity','parental level of education','lunch','test preparation course'])['average_score'].mean()
print(average_test_preparation_course)