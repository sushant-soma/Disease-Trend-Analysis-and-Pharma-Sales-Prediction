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
def poly(state,medicine):
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
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score
    import matplotlib.pyplot as plt
    from datetime import datetime

    # Convert the X and Y arrays to numpy arrays
    X = np.array(years)  # Replace with your X array
    Y = np.array(arr)  # Replace with your Y array

    # Convert Y array to integers and remove empty string
    Y = np.array([int(val) for val in Y if val != ''])

    # Convert X array to timestamps
    X = np.array([datetime.strptime(date, '%Y-%m-%d').timestamp() for date in X])

    # Reshape X to a column vector
    X = X.reshape(-1, 1)

    # Create polynomial features
    degree = 3  # Change the degree as desired
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X)

    # Create and train the polynomial regression model
    regressor = LinearRegression()
    regressor.fit(X_poly, Y)

    # Get the coefficients
    coef = regressor.coef_
    intercept = regressor.intercept_

    # Print the coefficient values
    print("Coefficients:", coef)
    print("Intercept:", intercept)
    plt.switch_backend('Agg') 
    # Plot the polynomial regression line and data points
    plt.scatter(X, Y, color='blue', label='Data Points')
    plt.plot(X, regressor.predict(X_poly), color='red', label='Polynomial Regression')
    plt.xlabel('Year')
    plt.ylabel('Sales')
    plt.title('Polynomial Regression')
    plt.legend()
    plt.savefig('static/polyGraph.png')



    print()
    print("--------------------------------PREDICT--------------------------------")
    print()

    # Ask the user for the next X value
    # next_date_str = input("Enter the next date (dd/mm/yyyy): ")
    # next_X = convert_to_timestamp(next_date_str)

    # # Transform next_X into polynomial features
    # next_X_poly = poly_features.transform([[next_X]])

    # # Predict the corresponding Y value
    # next_Y = regressor.predict(next_X_poly)

    # # Print the predicted Y value
    # print("Predicted Y value:", next_Y[0])

    # Select 4 random data points for evaluation
    print()
    print("--------------------------------PERFORMANCE_METRIX--------------------------------")
    print()
    np.random.seed(0)
    indices = np.random.choice(len(X), 4, replace=False)
    selected_X = X[indices]
    selected_X_poly = X_poly[indices]
    selected_Y = Y[indices]

    # Predict the Y values for the selected data points
    predicted_Y = regressor.predict(selected_X_poly)

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

