
import pandas as pd
import random
import matplotlib.pyplot as plt

df = pd.read_csv('augment.csv')


v = pd.DataFrame(columns=["date","val"])

for x,y in zip(df.values[0][3::],df.columns[3::]):
    v.loc[len(v.index)]=[y,x]

v.to_csv('single.csv')