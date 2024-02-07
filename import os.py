import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Function to load and preprocess data from a single file
def load_and_preprocess(file_path):
    df = pd.read_csv(file_path)
    X = df[['input_col1', 'input_col2']].values
    y = df['output_col'].values

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    return X, y

# Function to load and preprocess data from multiple files in a folder
def load_data_from_folder(folder_path):
    X_all, y_all = [], []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(folder_path, file_name)
            X, y = load_and_preprocess(file_path)
            X_all.append(X)
            y_all.append(y)

    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)

    return X_all, y_all

# Specify the folder containing your CSV files
data_folder = "your_data_folder"

# Load and preprocess data from multiple files in the folder
X_all, y_all = load_data_from_folder(data_folder)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

# Reshape input data for LSTM
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, activation='relu', input_shape=(1, X_train.shape[2])))
model.add(Dense(units=1))  # Output layer
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")

# Make predictions
predictions = model.predict(X_test)
