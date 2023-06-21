import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

def read_first_row(csv_file_path):
    with open(csv_file_path, 'r') as file:
        reader = csv.reader(file)
        first_row = next(reader)
        return first_row

# Example usage
def linear(state,medicine):
    csv_file_path = 'augment.csv'
    first_row = read_first_row(csv_file_path)

    years = first_row[3:]

    print()
    print("--------------------------------SELECT--------------------------------")
    print()

    # state = input("Enter the state name : ")
    # medicine = input("Enter the content name : ")

    print("---------------------------------------------------------------------")
    print()

    data = pd.read_csv('augment.csv')

    with open('augment.csv', 'r') as file:
        datareader = csv.DictReader(file)
        for row in datareader:
            #print(row)
            if (row['Prscrbr_Geo_Desc']==state and row['Gnrc_Name']==medicine):
                arr = []
                #print(years)
                for year in years:
                    arr.append(row[year])
                #print(arr)
                break

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, explained_variance_score
    from datetime import datetime

    # Convert the X and Y arrays to numpy arrays
    X = np.array(years)
    k=X[:(len(X)//5)]
    Y = np.array(arr)
    # Convert Y array to integers and remove empty string
    Y = np.array([int(val) for val in Y if val != ''])
    X = np.array([datetime.strptime(date, '%Y-%m-%d').timestamp() for date in X])


    # Convert X array to numerical values (e.g., integers or timestamps) if necessary

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Create and train the linear regression model
    regressor = LinearRegression()
    regressor.fit(X_train.reshape(-1, 1), Y_train)

    # Predict the values for the test set
    Y_pred = regressor.predict(X_test.reshape(-1, 1))

    # Get the coefficients
    coef = regressor.coef_
    intercept = regressor.intercept_

    # Print the coefficient values
    print("Coefficient:", coef)
    print("Intercept:", intercept)

    plt.switch_backend('Agg') 
    print("-------------------------XTest-------------------------------")
    print(X)
    # Plotting the results
    plt.scatter(X_test, Y_test, color='blue', label='Actual')
    plt.plot(X_test, Y_pred, color='red', label='Predicted')
    plt.xlabel('Year')
    plt.ylabel('Y')
    plt.title('Linear Regression')
    plt.legend()
    plt.savefig('static/linGraph.png')

    # Function to convert user input date to timestamp
    def convert_to_timestamp(date_str):
        date = datetime.strptime(date_str, '%d/%m/%Y')
        timestamp = date.timestamp()
        return timestamp
    print()
    # print("--------------------------------PREDICT--------------------------------")
    # print()
    # # Ask the user for the next X value
    # next_date_str = input("Enter the next date (dd/mm/yyyy): ")
    # next_X = convert_to_timestamp(next_date_str)

    # # Predict the corresponding Y value
    # next_Y = regressor.predict([[next_X]])

    # # Print the predicted Y value
    # print("Predicted Y value:", next_Y[0])


    # Select 4 random data points for evaluation
    print()
    print("--------------------------------PERFORMANCE_METRIX--------------------------------")
    print()
    np.random.seed(0)
    indices = np.random.choice(len(X), 4, replace=False)
    selected_X = X[indices].reshape(-1, 1)
    selected_Y = Y[indices]

    # Predict the Y values for the selected data points
    predicted_Y = regressor.predict(selected_X)

    # Calculate and display the evaluation metrics
    for i in range(len(indices)):
        evs = explained_variance_score(selected_Y[i:i+1], predicted_Y[i:i+1])
        mae = mean_absolute_error(selected_Y[i:i+1], predicted_Y[i:i+1])
        mse = mean_squared_error(selected_Y[i:i+1], predicted_Y[i:i+1])
        rmse = np.sqrt(mse)
        print("--------------------------------Evaluation metrics for data point", i+1,"--------------------------------")
        print("Explained Variance Score:", evs)
        print("MAE:", mae)
        print("MSE:", mse)
        print("RMSE:", rmse)
        print()

    print("------------------------------**************************************--------------------------------")






