import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import csv

import warnings


def read_first_row(csv_file_path):
    with open(csv_file_path, 'r') as file:
        reader = csv.reader(file)
        first_row = next(reader)
        return first_row

def sarimax_model(state,medicine):
    plt.switch_backend('Agg') 
    # Ignore the warning
    warnings.filterwarnings("ignore", message="No frequency information was provided")


    

    # Example usage
    csv_file_path = 'augment.csv'
    first_row = read_first_row(csv_file_path)

    years = first_row[3:]

    print()
    print("--------------------------------SELECT--------------------------------")
    print()

    # state = input("Enter the state name : ")
    # medicine = input("Enter the content name : ")


    data = pd.read_csv('augment.csv')
    arr=[]
    with open('augment.csv', 'r') as file:
        datareader = csv.DictReader(file)
        for row in datareader:
            #print(row)
            if (row['Prscrbr_Geo_Desc']==state and row['Gnrc_Name']==medicine):
                #arr = []
                #print(years)
                for year in years:
                    arr.append(row[year])
                #print(arr)
                break

    values = np.array(arr)

    # Extract the relevant columns
    #years = data['Year']
    #values = data['Value']

    # Create a datetime index
    index = pd.DatetimeIndex(pd.to_datetime(years, format='mixed'))

    # Create a pandas Series with the values and datetime index
    series = pd.Series(pd.to_numeric(values), index=index)

    # Split the data into training and testing sets
    train_data = series.iloc[:-20]  # Use all years except the last two for training
    test_data = series.iloc[-20:]  # Use the last two years for testing

    # Fit the ARIMA model
    # model = ARIMA(train_data, order=(1, 0, 0))  # Adjust the order as per your data
    # model_fit = model.fit()



    import statsmodels.api as sm


    model=sm.tsa.statespace.SARIMAX(train_data,order=(1, 1, 1),seasonal_order=(1,1,1,12))
    results=model.fit()


    predictions=results.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1,dynamic=True)
    #df[['Sales','forecast']].plot(figsize=(12,8))



    # Make predictions
    #predictions = model_fit.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1)

    # Print the predictions
    #print("Predictions:")
    #print(predictions)

    # Plot the actual data and predictions
    plt.plot(train_data.index, train_data, label='Training Data')
    plt.plot(test_data.index, test_data, label='Actual Data')
    plt.plot(test_data.index, predictions, label='Predictions')
    plt.xlabel('Year')
    plt.ylabel('Value')
    plt.title('SARIMAX Predictions')
    plt.legend()
    #plt.show()
    plt.savefig("static/graph.png")

    #print(index)
    
    # Forecast for the next 2 years
    future_years = pd.date_range(index[-1] + pd.DateOffset(months=1), periods=2, freq='M')
    forecast = results.get_forecast(steps=2).conf_int()

    print()
    print("---------------------------FORECAST-------------------------")
    print()

    # Print the forecasted values
    print("Forecasted values:")
    for year, value in zip(future_years, forecast):
        print(year, value)

    from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error

    # Convert predictions and test data to numpy arrays
    predictions = np.array(predictions)
    test_data = np.array(test_data)

    # Calculate the regression metrics
    evs = explained_variance_score(test_data, predictions)
    mae = mean_absolute_error(test_data, predictions)
    mse = mean_squared_error(test_data, predictions)
    rmse = np.sqrt(mse)

    print()
    print("----------------------PARAMETERS-----------------------------")
    print()

    # Print the regression metrics
    print("Regression Metrics:")
    print("Explained Variance Score:", evs)
    print("MAE:", mae)
    print("MSE:", mse)
    print("RMSE:", rmse)

    print()
    print("--------------------------*******************--------------------------")

    return future_years,forecast


