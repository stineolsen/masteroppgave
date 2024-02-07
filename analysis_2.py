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
    try:
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
    except ValueError as e:
        print(f"Skipping file due to error: {e}")
        return None, None

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
    plt.figure(figsize=(10, 5))

    # Plot the data points
    plt.scatter(X_test, y_test, color='black', label='Actual data')

    # Plot the regression line
    X_test_reshaped = np.array(X_test).reshape(-1, 1)
    y_pred = model.predict(X_test_reshaped)
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
    feeling_list = []

    # Loop through each file in the directory
    for file_name in os.listdir(data_directory):
        # Construct the full path to the file
        file_path = os.path.join(data_directory, file_name)

        # Read data from the file
        data = read_data(file_path)

        # Features ('hrv' and 'dfa') and target variable ('feeling')
        features = ['hrv']
        target = 'dfa'

      

        # Lists to store coefficients
        coefficients_hrv_dfa = []
        
        # Check for NaN values in the target variable 'dfa'
        if data[target].isna().any():
            #print(f"Skipping file '{file_name}' due to NaN values in the target variable 'dfa'.")
            continue

        # Separate features (X) and target variable (y)
        X = data[features]
        y = data[target]


        feeling = data['feeling']
        feeling_list.append(feeling[0])

        # Perform linear regression
        model, mse = perform_linear_regression(X, y)

        # Check if the model is None (indicating an error during linear regression)
        if model is None:
            continue

        linear_regression_results.append((file_name, model, mse))

        # Calculate correlation between 'hrv' and 'dfa'
        correlation_hrv_dfa = calculate_correlation(data, 'hrv', 'dfa')
        correlation_results.append((file_name, correlation_hrv_dfa))

        # Plot linear regression for 'hrv' and 'dfa' with 'feeling'
        # plot_regression(data, model, X['hrv'], y, 'hrv')

        # Print and store coefficients
        # mod_coef_hrv = model.coef_
        # coefficients_hrv_dfa.append(mod_coef_hrv)
        # print(f"Coefficient for file '{file_name}': hrv={mod_coef_hrv}")


    # Display results
    #for file_name, model, mse in linear_regression_results:
        # print(f"Linear Regression Results for {file_name}:")
        # print(f"Coefficients: {model.coef_}")
        # print(f"Intercept: {model.intercept_}")
        # print(f"Mean Squared Error: {mse}")
        #print()

    # for file_name, correlation_hrv_dfa in correlation_results:
    #     # print(f"Correlation Results for {file_name}:")
    #     # print(f"Pearson Correlation Coefficient (hrv-dfa): {correlation_hrv_dfa}")
        #print()
    
    # print(feeling_list)
    # print(coefficients_hrv_dfa)
    print(linear_regression_results[0])
    # print("All Coefficients for 'hrv' across files:", coefficients_hrv_dfa)
    # Extract correlation values into a new list
    correlation_values = [mse for _, _, mse in linear_regression_results]

    print(len(correlation_values))
    print(len(feeling_list))

    # Assuming correlation_values and feeling_list are defined and contain valid data

    # Scatter plot
    plt.figure(figsize=(10, 5))
    plt.scatter(correlation_values, feeling_list, color='black', label='Correlation vs Feeling')
    plt.title("Scatter Plot of Correlation vs Feeling")
    plt.xlabel("Correlation Values")
    plt.ylabel("Feeling")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()