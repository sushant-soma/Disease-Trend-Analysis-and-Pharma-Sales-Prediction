import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import time

# Importing the dataset
datas = pd.read_csv('all_year.csv')

years = ['DY13', 'DY14', 'DY15', 'DY16', 'DY17', 'DY18', 'DY19', 'DY20', 'DY21']

state = 'Alaska'
medicine = '0.9 % Sodium Chloride'

data = pd.read_csv('all_year.csv')

with open('all_year.csv', 'r') as file:
    datareader = csv.DictReader(file)
    for row in datareader:
        if row['Prscrbr_Geo_Desc'] == state and row['Gnrc_Name'] == medicine:
            arr = []
            for year in years:
                arr.append(int(row[year]))

X = np.array([13, 14, 15, 16, 17, 18, 19, 20, 21]).reshape(-1, 1)
y = np.array(arr)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X)

poly.fit(X_poly, y)
lin2 = LinearRegression()
lin2.fit(X_poly, y)

plt.scatter(X, y, color='blue')
plt.plot(X, lin2.predict(poly.fit_transform(X)), color='red')
plt.title('Polynomial Regression')
plt.xlabel('Year')
plt.ylabel('Sales')

plt.show()

time.sleep(1)

plt.close('all')

#flag = input("Do yo want to estimate the next year sale [Y|N] :")
if (True):
    pred2 = 22
    pred2array = np.array([[pred2]])
    prediction=lin2.predict(poly.fit_transform(pred2array))
    print(prediction)
    print(y)
    a=np.append(X,np.array(['Prediction']))
    b=np.append(y,np.array([prediction]))
    print(a.shape,b.shape)
    plt.scatter(a, b, color='blue')
    #plt.plot(a, lin2.predict(poly.fit_transform(X)), color='red')
    plt.show()




