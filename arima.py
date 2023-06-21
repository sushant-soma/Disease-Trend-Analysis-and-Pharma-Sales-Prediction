import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

with open('augment.csv', newline='') as f:
  csv_reader = csv.reader(f)
  csv_headings = next(csv_reader)
  first_line = next(csv_reader)



df=pd.read_csv('augment.csv')
print(first_line)

# Updating the header
#df.columns=["Month","Sales"]


# from statsmodels.tsa.stattools import adfuller

# test_result=adfuller(df['Sales'])

# def adfuller_test(sales):
#     result=adfuller(sales)
#     labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations']
#     for value,label in zip(result,labels):
#         print(label+' : '+str(value) )

# if result[1] <= 0.05:
#     print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data is stationary")
# else:
#     print("weak evidence against null hypothesis,indicating it is non-stationary ")

# adfuller_test(df['Sales'])






##########################################################



# For non-seasonal data
#p=1, d=1, q=0 or 1

from statsmodels.tsa.arima_model import ARIMA
model=ARIMA(df['Sales'],order=(1,1,1))
model_fit=model.fit()
print(model_fit.summary())


df['forecast']=model_fit.predict(start=90,end=103,dynamic=True)
df[['Sales','forecast']].plot(figsize=(12,8))


import statsmodels.api as sm
model=sm.tsa.statespace.SARIMAX(df['Sales'],order=(1, 1, 1),seasonal_order=(1,1,1,12))
results=model.fit()
df['forecast']=results.predict(start=90,end=103,dynamic=True)
df[['Sales','forecast']].plot(figsize=(12,8))

from pandas.tseries.offsets import DateOffset
future_dates=[df.index[-1]+ DateOffset(months=x)for x in range(0,24)]
future_datest_df=pd.DataFrame(index=future_dates[1:],columns=df.columns)

future_datest_df.tail()

future_df=pd.concat([df,future_datest_df])

future_df['forecast'] = results.predict(start = 104, end = 120, dynamic= True)
future_df[['Sales', 'forecast']].plot(figsize=(12, 8))