
import pandas as pd

df = pd.read_csv('solar2.csv')

new_rows = []

for index, row in df.iterrows():
    date = row['date']
    for col in df.columns[1:]: 
        new_rows.append([date, row[col]])
df_new = pd.DataFrame(new_rows, columns=['date', 'value'])

df_new.to_csv('solar2_v2.csv', index=False)
