import fitparse
import numpy as np
import scipy.io
import pandas as pd
from scipy import signal
from functions import *

import os
os.chdir('C:/Users/Olsen/Desktop/Masteroppgave/Data/fitfiler/candidate1/')


# ----------------------------------------------------
# Get HRV data from FIT file (using function above)
# ----------------------------------------------------
fitfile = fitparse.FitFile('tp-3912799.2023-05-15-09-17-45-718Z.GarminPing.AAAAAGRh-LkJc44-.FIT')


HRV_data = get_fit_hrv(fitfile)
HRV_with_time = fit_RR_intervals_with_last_known_timestamp(fitfile)


interpolated_data, changed_values_count = interpolate_missing_packages_count(HRV_with_time['rr_interval'])





# ----------------------------------------------------
# Export HRV to CSV file
# ----------------------------------------------------

data = {
    "timestamp": HRV_with_time['timestamp'],
    "HRV": interpolated_data,
    
}


# Create a CSV file with the data
output_file = "hrv_interpolated.csv"
df = pd.DataFrame(data)
df.to_csv(output_file, index=False)