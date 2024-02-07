import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import glob
import os
from sklearn.ensemble import HistGradientBoostingRegressor


def replace_feeling(value):
    if value >= 5:
        return 1
    else:
        return 2
    
def read_data(file_path):
    return pd.read_csv(file_path)


# Path to the directory containing data files
os.chdir('C:/Users/Olsen/Desktop/Masteroppgave/')
data_directory = "Data/fitfiler/candidate2/csv_files"

# files = glob.glob('Data/fitfiler/candidate1/csv_files*.csv')  # Adjust the path as needed
# files = glob.glob('*.csv')  # Adjust the path as needed
# # print(files)

# data = pd.concat([pd.read_csv(file) for file in files], ignore_index=True)



# List to store results
hrv_mean_list = []
dfa_mean_list = []
feeling_list = []
dropping_list_all = []

# Loop through each file in the directory
for file_name in os.listdir(data_directory):
    # Construct the full path to the file
    file_path = os.path.join(data_directory, file_name)

    # Read data from the file
    data = read_data(file_path)
    
    data = data.drop(columns=['sport', 'date'])
    for j in range(0,len(data.columns)):
        if not data.iloc[:, j].notnull().all():
            dropping_list_all.append(j)        
            #print(df.iloc[:,j].unique())
        
        # filling nan with mean in any columns

    for j in range(0,len(data.columns)):
        data.iloc[:,j]=data.iloc[:,j].fillna(data.iloc[:,j].mean())
    # another sanity check to make sure that there are not more any nan
    data.isnull().sum()

    hrv_mean = np.mean(data['hrv'])
    dfa_mean = np.mean(data['dfa'])
    feeling = data['feeling'][0]

    hrv_mean_list.append(hrv_mean)
    dfa_mean_list.append(dfa_mean)
    feeling_list.append(feeling)

    # Features ('hrv' and 'dfa') and target variable ('feeling')
    # features = ['hrv']
    # target = 'dfa'

# Generate synthetic data (replace this with your actual data loading)
# np.random.seed(42)
# data = pd.DataFrame({
#     'hrv': np.random.rand(100),
#     'dfa': np.random.rand(100),
#     'feeling': np.random.choice([0, 1], size=100)
# })

# nan_count = data['dfa'].isna().sum()
# print(f"Number of NaN values in 'dfa': {nan_count}")

# # Fill NaN values with the mean of the target variable
# data['dfa'] = data['dfa'].fillna(data['dfa'].mean())

# print(hrv_mean_list)
# dropping_list_all = []

data2 = pd.DataFrame({
    'hrv': hrv_mean_list,
    'dfa': dfa_mean_list,
    'feeling': feeling_list
})

# data = data.drop(columns=['sport', 'date'])
# for j in range(0,len(data.columns)):
#     if not data.iloc[:, j].notnull().all():
#         dropping_list_all.append(j)        
#         #print(df.iloc[:,j].unique())
    
#     # filling nan with mean in any columns

# for j in range(0,len(data.columns)):
#     data.iloc[:,j]=data.iloc[:,j].fillna(data.iloc[:,j].mean())
# # another sanity check to make sure that there are not more any nan
# data.isnull().sum()


data2['feeling'] = data2['feeling'].apply(lambda x: replace_feeling(x))



# Explore and visualize data
plt.scatter(data2['hrv'], data2['dfa'], c=data2['feeling'], cmap='viridis')
plt.xlabel('hrv')
plt.ylabel('dfa')
plt.title('Scatter Plot of hrv vs dfa (Colored by feeling)')
plt.show()



# Path to the directory containing data files
os.chdir('C:/Users/Olsen/Desktop/Masteroppgave/')
data_directory = "Data/fitfiler/candidate1/csv_files"


# Loop through each file in the directory
for file_name in os.listdir(data_directory):
    # Construct the full path to the file
    file_path = os.path.join(data_directory, file_name)

    # Read data from the file
    data3 = read_data(file_path)
    
    data3 = data3.drop(columns=['sport', 'date'])
    for j in range(0,len(data.columns)):
        if not data3.iloc[:, j].notnull().all():
            dropping_list_all.append(j)        
            #print(df.iloc[:,j].unique())
        
        # filling nan with mean in any columns

    for j in range(0,len(data3.columns)):
        data3.iloc[:,j]=data3.iloc[:,j].fillna(data3.iloc[:,j].mean())
    # another sanity check to make sure that there are not more any nan
    data3.isnull().sum()

    hrv_mean = np.mean(data3['hrv'])
    dfa_mean = np.mean(data3['dfa'])
    feeling = data3['feeling'][0]

    hrv_mean_list.append(hrv_mean)
    dfa_mean_list.append(dfa_mean)
    feeling_list.append(feeling)

data4 = pd.DataFrame({
    'hrv': hrv_mean_list,
    'dfa': dfa_mean_list,
    'feeling': feeling_list
})

data4['feeling'] = data4['feeling'].apply(lambda x: replace_feeling(x))


# Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(data2[['hrv', 'dfa']], data2['feeling'], test_size=0.2, random_state=None)

X_train = data2[['hrv', 'dfa']]
X_test =  data4[['hrv', 'dfa']]
y_train = data2['feeling']
y_test = data4['feeling']

# Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# est = HistGradientBoostingRegressor().fit(X_train, y_train)
# est.score(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)
predictions = np.round(y_pred).astype(int)
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy}')

# Interpret Results
print(f'Intercept: {model.intercept_}')
print(f'Coefficients: {model.coef_}')

# Visualize Results
# plt.scatter(data2['hrv'], data2['dfa'], c=data2['feeling'], cmap='viridis')
plt.plot(X_test['hrv'], X_test['dfa'], 'ro', label='Actual', alpha=0.5)
plt.plot(X_test['hrv'], y_pred, 'bo', label='Predicted', alpha=0.5)
plt.xlabel('hrv')
plt.ylabel('dfa')
plt.title('Linear Regression Fit with feeling Colored Points')
plt.legend()
plt.show()
