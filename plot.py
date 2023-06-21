import pandas as pd
import random
import matplotlib.pyplot as plt

df = pd.read_csv('all_year.csv')[0:100]

plt.plot(df.columns[2::],[int(str(x).replace(",","")) for x in df.values[0][2::]])
plt.show()