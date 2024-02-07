from fitparse import FitFile
import numpy as np
import pandas as pd
import math
from datetime import datetime



# ----------------------------------
#  Import data from FIT file
# ----------------------------------

def get_fit_hr(fit_file):
    '''
    Import HR data from FIT-file. 
    Timestamp [yyyy-mm-ddThh:mm:ss], HR[bpm]
    '''
    #fit = FitFile(fit_file)
    fit = fit_file
    " Return a df of heart rate and timestamp from a fitparsed .fit-file "
    return pd.DataFrame({'timestamp': [msg.get('timestamp').value if msg.get('timestamp').value else pd.NA for msg in
    fit.get_messages('record')], 'heart_rate': [msg.get('heart_rate').value if msg.get('heart_rate').value else pd.NA for msg in fit.get_messages('record')]})


# def get_fit_sport(fitfile):
#     '''
#     Import sport data from FIT-file. 
#     name, sport, sub_sport
#     '''
#     allHeadings =[]
#     dictR = {}
#     df = pd.DataFrame(columns=allHeadings)

#     for record in fitfile.get_messages("sport"):
#         for data in record:
#             k = str(data.name) + ' (' + str(data.units) + ')'
#             v = {k : data.value}
#             dictR.update(v)
#             new_row = pd.DataFrame.from_records([dictR])
#         df = pd.concat((df, new_row))
#         #df2 = df[['name (None)', 'sport (None)', 'sub_sport (None)']].copy()
#         df2 = df[['sport (None)', 'sub_sport (None)']].copy()

#     return df2

def get_fit_sport(fitfile):
    '''
    Import sport data from FIT-file. 
    name, sport, sub_sport
    '''
    allHeadings =[]
    dictR = {}
    df = pd.DataFrame(columns=allHeadings)

    try:
        for record in fitfile.get_messages("sport"):
            for data in record:
                k = str(data.name) + ' (' + str(data.units) + ')'
                v = {k : data.value}
                dictR.update(v)
                new_row = pd.DataFrame.from_records([dictR])
            df = pd.concat((df, new_row))
            df2 = df[['sport (None)', 'sub_sport (None)']].copy()
            
        return df2

    except Exception as e:
        print(f"Error in get_fit_sport: {e}")

        

def get_fit_set(fitfile):
    '''
    Import set data from FIT-file. 
    name, sport, sub_sport
    '''
    allHeadings =[]
    dictR = {}

    df = pd.DataFrame(columns=allHeadings)
    for record in fitfile.get_messages("activity"):
        for data in record:
            k = str(data.name) + ' (' + str(data.units) + ')'
            v = {k : data.value}
            dictR.update(v)
            new_row = pd.DataFrame.from_records([dictR])
        df = pd.concat((df, new_row))
        #df2 = df[['local_timestamp (None)','timestamp (None)', 'total_timer_time (s)']].copy()
        df2 = df[['timestamp (None)', 'total_timer_time (s)']].copy()
    return df2


def get_fit_recordings(fitfile):
    '''
    Import all recording data from FIT-file. 
    '''
    allHeadings =[]
    dictR = {}

    #Here we iterate over the records and format them into columns which are easier to plot.   
    df = pd.DataFrame(columns=allHeadings)
    for record in fitfile.get_messages("record"):
        for data in record:
            k = str(data.name) + ' (' + str(data.units) + ')'
            v = {k : data.value}
            dictR.update(v)
            new_row = pd.DataFrame.from_records([dictR])
        df = pd.concat((df, new_row))
    return df


def get_fit_hrv(fitfile):
    '''
    Import HRV data from FIT-file. 
    
    HRV[s]
    
    '''

    HRV_data=[]
    for record in fitfile.get_messages('hrv'):
        for record_data in record:
            for RR_interval in record_data.value:
                if RR_interval is not None:
                    HRV_data.append(RR_interval)
    return HRV_data


def fit_RR_intervals_with_last_known_timestamp(fit):

    ' Extract RR intervals and last known timestamps from a fitparsed .fit-file. '

    data = {'timestamp': [], 'rr_interval': []}
    last_ts = None
    for msg in fit.get_messages():

        if msg.name == 'record':
            # Check if the timestamp field is known

            if msg.get('timestamp').value:
                # Update last_ts with the timestamp field value
                last_ts = msg.get('timestamp').value

        elif msg.name == 'hrv':
            # Check if the time field is not None

            if msg.get('time') is not None:
                # Iterate through the five time field values

                for j in range(5):
                    # Check if the time field value is not None

                    if msg.get('time').value[j] is not None:
                        # Append the time field value to 'rr_interval' key in the dict
                        data['rr_interval'].append(msg.get('time').value[j])
                        # Check if last_ts is not None

                        if last_ts:
                            # Append last_ts to 'timestamp' key in the dict
                            data['timestamp'].append(last_ts)
                        else:
                            # Append pd.NA to 'timestamp' key in the dict
                            data['timestamp'].append(pd.NA)

    return pd.DataFrame(data)



# ----------------------------------
#  RMSSD
# ----------------------------------

def calculate_RMSSD(my_list_ms):
    #Calculate SDNN
    SDNN = np.std(my_list_ms)

    #Calculate successive differences, i.e. the SD part in rmsSD
    successive_differences = [my_list_ms[i+1] - my_list_ms[i] for i in range(len(my_list_ms)-1)]

    #Calculate squared successive differences, i.e. the SSD part in rmSSD
    squared_successive_differences = [successive_difference**2 for successive_difference in successive_differences]

    #Calculate the mean of the squared successive differences (rMSSD)
    mean = np.mean(squared_successive_differences)

    #Calculate the root of the mean squared successive differences
    rMSSD = mean**0.5
    # print("RMSSD (root mean squared successive differences): ")
    #print(rMSSD)
    return rMSSD
    
    
    

# ----------------------------------
#  Interpolation
# ----------------------------------


def interpolate_missing_packages_count(data):
    '''
    Finds missing packages based on theory that one missing package has approximatly the double value as neighbour data. Interpolate based on one or two missing packages.
    Interpolates if value is below 0.3 as thats over maximal heart rate possible.  
    
    '''
    interpolated_data = data.copy()
    count_changed_values = 0  # Initialize count of changed values

    for i in range(2, len(interpolated_data)):
        # Assuming the threshold to detect anomalies
        threshold = 1.25 * interpolated_data[i - 1]

        if interpolated_data[i] > threshold:
            # Interpolate missing values based on different scenarios of package loss
            if interpolated_data[i] > 1.7 * interpolated_data[i - 1]:
                interpolated_data[i] = (interpolated_data[i - 1] + interpolated_data[i - 2]) / 2
                count_changed_values += 1
            elif interpolated_data[i] > 2.7 * interpolated_data[i - 1]:
                interpolated_data[i] = (interpolated_data[i - 1] + interpolated_data[i - 2] + interpolated_data[i - 3]) / 3
                count_changed_values += 1
        elif interpolated_data[i] < 0.25:
                interpolated_data[i] = (interpolated_data[i - 1] + interpolated_data[i - 2]) / 2
        elif interpolated_data[i] > 1.75:
                interpolated_data[i] = (interpolated_data[i - 1] + interpolated_data[i - 2]) / 2

    return interpolated_data, count_changed_values



def DFA(pp_values, lower_scale_limit, upper_scale_limit):
    scaleDensity = 30 # scales DFA is conducted between lower_scale_limit and upper_scale_limit
    m = 1 # order of polynomial fit (linear = 1, quadratic m = 2, cubic m = 3, etc...). Alpha 1 means linear

    # initialize, we use logarithmic scales
    start = np.log(lower_scale_limit) / np.log(10)
    stop = np.log(upper_scale_limit) / np.log(10)
    scales = np.floor(np.logspace(np.log10(math.pow(10, start)), np.log10(math.pow(10, stop)), scaleDensity))
    F = np.zeros(len(scales))
    count = 0

    for s in scales:
        rms = []
        # Step 1: Determine the "profile" (integrated signal with subtracted offset)
        x = pp_values
        y_n = np.cumsum(x - np.mean(x))
        # Step 2: Divide the profile into N non-overlapping segments of equal length s
        L = len(x)
        shape = [int(s), int(np.floor(L/s))]
        nwSize = int(shape[0]) * int(shape[1])
        # beginning to end, here we reshape so that we have a number of segments based on the scale used at this cycle
        Y_n1 = np.reshape(y_n[0:nwSize], shape, order="F")
        Y_n1 = Y_n1.T
        # end to beginning
        Y_n2 = np.reshape(y_n[len(y_n) - (nwSize):len(y_n)], shape, order="F")
        Y_n2 = Y_n2.T
        # concatenate
        Y_n = np.vstack((Y_n1, Y_n2))

        # Step 3: Calculate the local trend for each 2Ns segments by a least squares fit of the series
        for cut in np.arange(0, 2 * shape[1]):
            xcut = np.arange(0, shape[0])
            pl = np.polyfit(xcut, Y_n[cut,:], m)
            Yfit = np.polyval(pl, xcut)
            arr = Yfit - Y_n[cut,:]
            rms.append(np.sqrt(np.mean(arr * arr)))

        if (len(rms) > 0):
            F[count] = np.power((1 / (shape[1] * 2)) * np.sum(np.power(rms, 2)), 1/2)
        count = count + 1

    pl2 = np.polyfit(np.log2(scales), np.log2(F), 1)
    alpha = pl2[0]
    return alpha



# def computeFeatures(df):
#     features = []
#     window_size = 120  # Window size in seconds
#     window_step = 1  # Window step in seconds

#     start_time = df['timestamp'].iloc[0]  # Get the initial timestamp
#     end_time = start_time + window_size  # Set the end time of the initial window

#     while end_time <= df['timestamp'].iloc[-1]:  # Continue until the end of the timestamps

#         array_rr = df.loc[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time), 'RR'] * 1000

#         if len(array_rr) < 2:  # Insufficient data points in the window
#             start_time += window_step  # Move the window start time
#             end_time += window_step  # Move the window end time
#             continue

#         heartrate = round(60000 / np.mean(array_rr), 2)
#         NNdiff = np.abs(np.diff(array_rr))
#         rmssd = round(np.sqrt(np.sum((NNdiff * NNdiff) / len(NNdiff))), 2)
#         sdnn = round(np.std(array_rr), 2)
#         alpha1 = DFA(array_rr.to_list(), 4, 16)

#         curr_features = {
#             'timestamp': start_time,
#             'heartrate': heartrate,
#             'rmssd': rmssd,
#             'sdnn': sdnn,
#             'alpha1': alpha1,
#         }

#         features.append(curr_features)
#         start_time += window_step  # Move the window start time
#         end_time += window_step  # Move the window end time

#     features_df = pd.DataFrame(features)
#     return features_df

import numpy as np
import pandas as pd

# Your existing DFA function remains the same

def computeFeatures(df):
    features = []
    window_size = 120
    window_step = 1

    max_index = int(round(np.max(df['timestamp']) / window_step))

    for index in range(0, max_index):
        start_time = index * window_step
        end_time = start_time + window_size

        array_rr = df.loc[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time), 'RR'] * 1000

        if len(array_rr) < 2:  # Insufficient data points in the window
            continue

        heartrate = round(60000 / np.mean(array_rr), 2)
        NNdiff = np.abs(np.diff(array_rr))
        rmssd = round(np.sqrt(np.sum((NNdiff * NNdiff) / len(NNdiff))), 2)
        sdnn = round(np.std(array_rr), 2)
        alpha1 = DFA(array_rr.to_list(), 4, 16)

        curr_features = {
            'timestamp': start_time,
            'heartrate': heartrate,
            'rmssd': rmssd,
            'sdnn': sdnn,
            'alpha1': alpha1,
        }

        features.append(curr_features)

    features_df = pd.DataFrame(features)
    return features_df
