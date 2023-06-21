import pandas as pd

df = pd.read_csv('Original/DY13.csv')


df = df.groupby(['Prscrbr_Geo_Desc','Gnrc_Name']).sum()['Tot_Clms']


df.to_csv('Summed/DY13_f.csv')