
import pandas as pd

df = pd.read_csv('data-night5-i.csv')

new_rows = []

for index, row in df.iterrows():
    for col in df.columns[0:]: 
        new_rows.append([row[col]])
df_new = pd.DataFrame(new_rows, columns=['value'])

df_new.to_csv('data-night5-i_v2.csv', index=False)
