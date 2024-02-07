import pyhrv.time_domain as td
import pyhrv.frequency_domain as fd

import numpy as np
import heartpy as hp
import os

os.chdir('C:/Users/Olsen/Desktop/Masteroppgave/Code/Swell/hrv dataset/data/raw/rri')



with open('p1.txt') as f:
    lines = f.readlines()
    x = np.array([line.split()[0] for line in lines])
    y = np.array([line.split()[1] for line in lines])


rr = y.astype(np.float64)
t = x.astype(np.float64)

hr = 60000/rr
# hr_mean = np.mean(hr)
# results = td.rmssd(rr)
# td.sdnn
# td.sdsd


# MEAN_RR
# MEDIAN_RR
# SDRR
# RMSS
# SDSD
# SDRR_RMSSD
# HR
# pNN25
# pNN50
# SD1
# SD2
# KURT
# SKEW
# MEAN_REL_RR
# MEDIAN_REL_RR
# SDRR_REL_RR
# RMSSD_REL_RR
# SDSD_REL_RR
# SDRR_RMSSD_REL_RR
# KURT_REL_RR
# SKEW_REL_RR
# VLF
# VLF_PCT
# LF
# LF_PCT
# LF_NU
# HF
# HF_PCT
# HF_NU
# TP
# LF_HF
# HF_LF
# sampen
# higuci
# datasetId
# condition

import numpy as np
from pyhrv import time_domain, frequency_domain
from datetime import datetime

def calculate_hrv_parameters(timestamps, hrv_values):
    # Convert timestamps to datetime objects
    # datetime_objects = [datetime.utcfromtimestamp(ts) for ts in timestamps]

    # Reshape data into 1-minute segments
    segment_size = 240
    timestamps_1min = np.array_split(timestamps, len(timestamps) // segment_size)
    hrv_values_1min = np.array_split(hrv_values, len(hrv_values) // segment_size)

    # Initialize lists to store results for each minute
    results = []

    for i in range(len(timestamps_1min)):
        segment_timestamps = timestamps_1min[i]
        segment_hrv_values = hrv_values_1min[i]

        # Compute HRV parameters using pyhrv
        rr_intervals = np.diff(segment_timestamps)
        rpeaks = np.cumsum(rr_intervals)
        nn_intervals = np.diff(rpeaks)

        # Time domain analysis
        time_domain_results = time_domain.time_domain(nn_intervals, threshold=25)

        # Frequency domain analysis
        # frequency_domain_results = frequency_domain(nn_intervals)


        # Additional parameters
        mean_rel_rr = np.mean(nn_intervals) / 1000.0
        median_rel_rr = np.median(nn_intervals) / 1000.0
        stdev_rel_rr = np.std(nn_intervals) / 1000.0
        rmssd_rel_rr = np.sqrt(np.mean(np.diff(nn_intervals) ** 2)) / 1000.0
        sdnn_rel_rr = np.std(nn_intervals) / 1000.0

        # Combine results
        parameters = {
            'MEAN_RR': time_domain_results['nni_mean'],
            'MEDIAN_RR': np.median(nn_intervals),
            'SDRR': time_domain_results['sdnn'],
            'RMSSD': time_domain_results['rmssd'],
            'SDSD': time_domain_results['sdsd'],
            'SDRR_RMSSD': time_domain_results['sdsd']/time_domain_results['rmssd'],
            'HR': 60000.0 / time_domain_results['nni_mean'],
            'pNN25': time_domain_results['pnn25'],
            'pNN50': time_domain_results['pnn50'],
            # 'SD1': time_domain_results['sd1'],
            # 'SD2': time_domain_results['sd2'],
            # 'KURT': time_domain_results['nni_kurt'],
            # 'SKEW': time_domain_results['nni_skew'],
            'MEAN_REL_RR': mean_rel_rr,
            'MEDIAN_REL_RR': median_rel_rr,
            'SDRR_REL_RR': stdev_rel_rr,
            'RMSSD_REL_RR': rmssd_rel_rr,
            'SDSD_REL_RR': sdnn_rel_rr,
            'SDRR_RMSSD_REL_RR': stdev_rel_rr / rmssd_rel_rr,
            # 'KURT_REL_RR': np.nan,  # pyhrv does not provide kurtosis for relative RR intervals
            # 'SKEW_REL_RR': np.nan,  # pyhrv does not provide skewness for relative RR intervals
            # Add more parameters as needed
        }

        # Append results for the current minute
        results.append(parameters)

    return results


# Calculate HRV parameters for each minute
results = calculate_hrv_parameters(t, rr)

# Display results
# for i, result in enumerate(results, start=1):
#     print(f"Minute {i}: {result}")

print(results)

