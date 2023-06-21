import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

c=list()
for i in range(14,22):
    for j in range(1,13):
        c.append(f'20{i}-{j}-1')

years = ['DY13','DY14','DY15','DY16','DY17','DY18','DY19','DY20','DY21']

state = input("Enter the state name : ")
medicine = input("Enter the content name : ")

data = pd.read_csv('all_year.csv')

with open('all_year.csv', 'r') as file:
    datareader = csv.DictReader(file)
    for row in datareader:
        #print(row)
        if (row['Prscrbr_Geo_Desc']==state and row['Gnrc_Name']==medicine):
            arr = []
            for year in years:
                arr.append(int(row[year]))



def estimate_coef(x, y):
    # number of observations/points
    n = np.size(x)
 
    # mean of x and y vector
    m_x = np.mean(x)
    m_y = np.mean(y)
 
    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y*x) - n*m_y*m_x
    SS_xx = np.sum(x*x) - n*m_x*m_x
 
    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x
 
    return (b_0, b_1)
 
def plot_regression_line(x, y, b):
    # plotting the actual points as scatter plot
    plt.scatter(x, y, color = "m",
               marker = "o", s = 30)
 
    # predicted response vector
    y_pred = b[0] + b[1]*x
 
    # plotting the regression line
    plt.plot(x, y_pred, color = "g")
 
    # putting labels
    plt.xlabel('x')
    plt.ylabel('y')
 
    # function to show plot
    plt.show()

def main():
    # observations / data
    x = np.array([13,14,15,16,17,18,19,20,21])
    
    y = np.array(arr)
 
    # estimating coefficients
    b = estimate_coef(x, y)
    print("Estimated coefficients:\nb_0 = {}  \
          \nb_1 = {}".format(b[0], b[1]))
 
    # plotting regression line
    plot_regression_line(x, y, b)

    flag = input("Do yo want to estimate the next year sale [Y|N] :")
    if (flag=='Y'):
        num = b[1]*22+b[0]
        print(num)

if __name__ == "__main__":
    main()