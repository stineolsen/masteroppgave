import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# Function to read data from CSV files
def read_data(file_path):
    return pd.read_csv(file_path)

# Function to perform linear regression
def perform_linear_regression(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a linear regression model
    model = LinearRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate the mean squared error
    mse = mean_squared_error(y_test, y_pred)

    return model, mse


# Function to calculate correlation
def calculate_correlation(data, column1, column2):
    # Extract columns
    col1_data = data[column1]
    col2_data = data[column2]

    # Calculate Pearson correlation coefficient
    correlation_coefficient, _ = pearsonr(col1_data, col2_data)

    return correlation_coefficient
# Function to plot linear regression for 'hrv' and 'dfa' with 'feeling'
def plot_regression(data, model, X_test, y_test, feature_name):
    #print("X_test_reshaped")
    #print(np.size(X_test, 0), np.size(X_test, 1))    
    
    #print(y_test.size)
    # Plot the regression line
    X_test_reshaped = np.array(X_test).reshape(-1, 1)

    #print("X_test_reshaped")
    #print(np.size(X_test_reshaped, 0), np.size(X_test_reshaped, 1)) 


    plt.figure(figsize=(10, 5))

    # Plot the data points
    plt.scatter(X_test[feature_name[0]], y_test, color='black', label='Actual data')
   

    y_pred = model.predict(X_test)
    plt.plot(X_test, y_pred, color='blue', linewidth=3, label='Regression line')

    plt.title(f"Linear Regression Results for {feature_name}")
    plt.xlabel(feature_name)
    plt.ylabel("Feeling")
    plt.legend()

    # Set feature names for the model
    plt.gca().set_xlabel(f'{feature_name} (feature)')
    plt.gca().set_ylabel('Feeling (target)')

    plt.show()



# Main function
def main():
    os.chdir('C:/Users/Olsen/Desktop/Masteroppgave/')
    # Path to the directory containing data files
    data_directory = "Data/fitfiler/candidate1/csv_files"

    # List to store results
    linear_regression_results = []
    correlation_results = []

    # Loop through each file in the directory
    for file_name in os.listdir(data_directory):
        # Construct the full path to the file
        file_path = os.path.join(data_directory, file_name)

        # Read data from the file
        data = read_data(file_path)

        # Features ('hrv' and 'dfa') and target variable ('feeling')
        features = ['hrv', 'dfa']
        target = 'feeling'

        # Separate features (X) and target variable (y)
        X = data[features]
        y = data[target]
        #print("X: ", X)
        #print("y: ", y)

        # Perform linear regression
        model, mse = perform_linear_regression(X, y)
        linear_regression_results.append((file_name, model, mse))

        # Calculate correlation between 'hrv', 'dfa', and 'feeling'
        correlation_hrv_feeling = calculate_correlation(data, 'hrv', 'feeling')
        correlation_dfa_feeling = calculate_correlation(data, 'dfa', 'feeling')
        correlation_results.append((file_name, correlation_hrv_feeling, correlation_dfa_feeling))
        print(file_name)
        # Plot linear regression for 'hrv' and 'dfa' with 'feeling'
        #for feature in features:
        plot_regression(data, model, X[features], y, features)

    # Display results
    for file_name, model, mse in linear_regression_results:
        print(f"Linear Regression Results for {file_name}:")
        print(f"Coefficients: {model.coef_}")
        print(f"Intercept: {model.intercept_}")
        print(f"Mean Squared Error: {mse}")
        print()

    for file_name, correlation_hrv_feeling, correlation_dfa_feeling in correlation_results:
        print(f"Correlation Results for {file_name}:")
        print(f"Pearson Correlation Coefficient (hrv-feeling): {correlation_hrv_feeling}")
        print(f"Pearson Correlation Coefficient (dfa-feeling): {correlation_dfa_feeling}")
        print()

if __name__ == "__main__":
    main()
