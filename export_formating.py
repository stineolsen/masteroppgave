import os
#os.chdir('C:/Users/Olsen/Desktop/Masteroppgave')

import matplotlib.pyplot as plt
# import plotly.graph_objects as go 
# from plotly.subplots import make_subplots

import pandas as pd
import numpy as np
import scipy.io
from fitparse import FitFile

from scipy import signal 

from functions import *
from datetime import datetime

import time

# get the start time
st = time.time()

# Change the current working directory to the directory containing FIT files
fit_files_directory = 'Masteroppgave/Data/fitfiler/candidate3/running3'
os.chdir(fit_files_directory)


subjective = pd.read_excel("candidate3_subjective.xlsx") 

print(subjective)



# Initialize lists to store data
# rmssd_first_list = []
# rmssd_last_list = []
# sport_list = []
# file_name_list = []
# min_length = []
# samp_length = []


# Loop through all FIT files in the directory
for file_name in os.listdir():
    if file_name.endswith(".FIT"):
        try:
            # Load the FIT file
            fitfile = FitFile(file_name)
            print(file_name)



            HRV_time = fit_RR_intervals_with_last_known_timestamp(fitfile)

            timestamps = HRV_time['timestamp']
            date = datetime.date(timestamps[0])

            interpolated_data, changed_values_count = interpolate_missing_packages_count(HRV_time['rr_interval'])


            time_elapsed = []
            start_time = HRV_time['timestamp'][0]

            for i in HRV_time['timestamp']:
                a = start_time
                b = i
                c = b-a
                time_elapsed.append(c.total_seconds())



            sport = get_fit_sport(fitfile)


            date_to_find = date  
            sport_to_find = sport['sport (None)']  


            # Find the row where both 'WorkoutDay' and 'WorkoutType' match
            result = subjective[(subjective['WorkoutDay'] == str(date_to_find)) & (subjective['Sport_match'] == sport_to_find[0])]


            # Check if any matching rows were found
            if not result.empty:
                # Get the 'feeling' parameter from the first matching row
                feeling_value = result.iloc[0]['Feeling']
                print(f"The 'feeling' parameter for the given date and sport is: {feeling_value}")
                rpe_value = result.iloc[0]['Rpe']
                print(f"The 'feeling' parameter for the given date and sport is: {rpe_value}")

            else:
                print("No matching row found.")

                
            if pd.isna(feeling_value):
                print("Skip")
            else: 
                df_t = pd.DataFrame()
                df_t['timestamp'] = time_elapsed
                df_t['RR'] = interpolated_data


                features_df_t = computeFeatures(df_t)


                list_length = len(features_df_t['alpha1'])
                date_list = [date] * list_length

                sport_list = [sport_to_find[0]] * list_length
                feeling_list = [feeling_value] * list_length
                rpe_list = [rpe_value] * list_length


                data = {
                    "date": date_list,
                    "timestamp": features_df_t['timestamp'],
                    "hrv": features_df_t['heartrate'],
                    "dfa": features_df_t["alpha1"],
                    "sport": sport_list,
                    "feeling": feeling_list, 
                    "rpe": rpe_list,    
                }


                output_file = f"candidate2_{date_to_find}_{sport_to_find[0]}_{str(feeling_value)}.csv"
                #output_file = "can1_%s.csv" % date
                df = pd.DataFrame(data)
                df.to_csv(output_file, index=False)
                
                
            et = time.time()
            elapsed_time = et - st
            print('Execution time:', elapsed_time/60, 'minuttes')
            print(" ")
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
