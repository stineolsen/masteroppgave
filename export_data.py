import os
os.chdir('C:/Users/Olsen/Desktop/Masteroppgave')

import matplotlib.pyplot as plt
import plotly.graph_objects as go 
from plotly.subplots import make_subplots

import pandas as pd
import numpy as np
import scipy.io
from fitparse import FitFile

from scipy import signal 

from functions import *
from datetime import datetime


# Change the current working directory to the directory containing FIT files
fit_files_directory = 'C:/Users/Olsen/Desktop/Masteroppgave/Data/fitfiler/candidate1/'
os.chdir(fit_files_directory)


subjective = pd.read_excel("sample.xlsx") 


# Initialize lists to store data
rmssd_first_list = []
rmssd_last_list = []
sport_list = []
file_name_list = []
min_length = []
samp_length = []


# Loop through all FIT files in the directory
for file_name in os.listdir():
    if file_name.endswith(".FIT"):
        try:
            # Load the FIT file
            fitfile = FitFile(file_name)
            print(file_name)
            
            HRV_time = fit_RR_intervals_with_last_known_timestamp(fitfile)

            timestamps = HRV_time['timestamp']

            # a = timestamps[0]
            # b = timestamps.iloc[-1]

            # c = b - a

            # minutes = c.total_seconds() / 60
            # samples_length = len(HRV_time['rr_interval']) 

            # min_length.append(minutes)
            # samp_length.append(samples_length)
        

        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            samp_length.append("N/A")
            min_length.append("N/A")


        sport = get_fit_sport(fitfile)
        sport_list.append(sport)
        file_name_list.append(file_name)


# Create a DataFrame to store the data
# data = {
#     "file_name": file_name_list,
#     "rmssd_first": rmssd_first_list,
#     "rmssd_last": rmssd_last_list,
#     "sport": sport_list,
# }

data = {
    "file_name": file_name_list,
    "sport": sport_list,
    "minuttes": min_length,
    "samples": samp_length,
}

# Create a CSV file with the data
output_file = "data_lengths_inkl_sport.csv"
df = pd.DataFrame(data)
df.to_csv(output_file, index=False)

print(f"Data saved to {output_file}")
