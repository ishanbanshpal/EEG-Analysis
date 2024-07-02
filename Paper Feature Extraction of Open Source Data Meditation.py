#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import mne
from scipy.signal import firwin, lfilter
from mne.time_frequency import tfr_multitaper
from autoreject import AutoReject

results_df = pd.DataFrame(columns=['File'] +
                                [f'Slope_Channel_{i}' for i in range(14)] +
                                [f'ITF_Channel_{i}' for i in range(14)] +
                                [f'ITP_Channel_{i}' for i in range(14)] +
                                [f'IAF_Channel_{i}' for i in range(14)] +
                                [f'IAP_Channel_{i}' for i in range(14)] +
                                [f'IBF_Channel_{i}' for i in range(14)] +
                                [f'IBP_Channel_{i}' for i in range(14)] +
                                [f'Mean_PSD_Channel_Alpha_{i}' for i in range(14)] +
                                [f'Mean_PSD_Channel_Beta_{i}' for i in range(14)])


# List of file names
file_names = ['m_med_14.txt','m_med_20.txt','m_med_24.txt','m_rest_24.txt', 'm_med_25.txt','m_med_31.txt',"m_med_7.txt","m_med_9.txt","m_rest_14.txt","m_rest_20.txt","m_rest_25.txt","m_rest_31.txt","m_rest_7.txt","m_rest_9.txt",'m_med_2.txt' , 'm_rest_2.txt' , 'm_med_3.txt' , 'm_rest_3.txt' , 'm_med_4.txt' ,'m_rest_4.txt' ,'m_med_6.txt' ,'m_rest_6.txt' ,'m_med_10.txt' ,'m_rest_10.txt' ,'m_med_11.txt' ,'m_rest_11.txt' ,'m_med_12.txt' ,'m_rest_12.txt' ,'m_med_13.txt' ,'m_rest_13.txt' ,'m_med_15.txt' ,'m_rest_15.txt' ,'m_med_16.txt' ,'m_rest_16.txt' ,'m_med_17.txt' ,'m_rest_17.txt' ,'m_med_19.txt' ,'m_rest_19.txt' ,'m_med_21.txt' ,'m_rest_21.txt' ,'m_med_22.txt' ,'m_rest_22.txt' ,'m_med_23.txt' ,'m_rest_23.txt' ,'m_med_27.txt' ,'m_rest_27.txt' ,'m_med_28.txt' ,'m_rest_28.txt' ,'m_med_29.txt' ,'m_rest_29.txt' ,'m_med_30.txt' ,'m_rest_30.txt' ,'m_med_32.txt' ,'m_rest_32.txt' ,'m_med_33.txt' ,'m_rest_33.txt' ,'m_med_34.txt' ,'m_rest_34.txt',"c_med_10.txt","c_med_12.txt","c_med_13.txt","c_med_14.txt","c_med_15.txt","c_med_16.txt","c_med_19.txt","c_med_2.txt","c_med_22.txt","c_med_23.txt","c_med_24.txt","c_med_25.txt","c_med_26.txt","c_med_29.txt","c_med_30.txt","c_med_6.txt","c_med_9.txt","c_med_n1.txt","c_med_n10.txt","c_med_n11.txt","c_med_n12.txt","c_med_n2.txt","c_med_n3.txt","c_med_n4.txt","c_med_n5.txt","c_med_n6.txt","c_med_n7.txt","c_med_n8.txt","c_med_n9.txt","c_rest_10.txt","c_rest_12.txt","c_rest_13.txt","c_rest_14.txt","c_rest_15.txt","c_rest_16.txt","c_rest_19.txt","c_rest_2.txt","c_rest_22.txt","c_rest_23.txt","c_rest_24.txt","c_rest_25.txt","c_rest_26.txt","c_rest_29.txt","c_rest_30.txt","c_rest_6.txt","c_rest_9.txt","c_rest_n1.txt","c_rest_n10.txt","c_rest_n11.txt","c_rest_n12.txt","c_rest_n2.txt","c_rest_n3.txt","c_rest_n4.txt","c_rest_n5.txt","c_rest_n6.txt","c_rest_n7.txt","c_rest_n8.txt","c_rest_n9.txt"]
for file_name in file_names:
    delimiter = ','
    df = pd.read_csv(file_name, delimiter=delimiter, encoding='latin1', on_bad_lines='skip')

    import pandas as pd

# Replace 'NaN' values in the 'Events' column with 0
    df['Events'] = df['Events'].replace('NaN', 0)


    # Convert values with more than one decimals to numeric format for all channels
    for column in df.columns:
        df[column] = df[column].astype(str)
        df[column] = df[column].str.replace('[^\d.-]', '', regex=True)
        df[column] = df[column].str.replace('\.{2,}', '.', regex=True)
        df[column] = pd.to_numeric(df[column], errors='coerce')


    df.info()

    df.isna().sum()

    df.head()

    df['Events'] = df['Events'].replace('NaN', 0)

    df.head()

    df['Events'] = df['Events'].fillna(0)
    
    column_averages=df.mean()

    for column in df.columns:
        df[column].fillna(column_averages[column], inplace=True)

    df.head()

    import numpy as np

    import mne


    # Update channel names and types for ICA
    ch_names = ['FP1', 'FP2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4',
                'T4', 'T5', 'P3', 'PZ', 'P4', 'T6', 'O1', 'O2', 'A1', 'A2', 'SensorCEOG', 'SensorDEOG', 'Events']

    # Start with all channels as 'eeg'
    ch_types = ['eeg'] * len(ch_names)

    # Define EOG and misc channels
    eog_channels = ['SensorCEOG', 'SensorDEOG']
    misc_channels = ['A1', 'A2']
    misc_channels.append('Events')

    # Update ch_types for EOG channels
    for channel_name in eog_channels:
        ch_idx = ch_names.index(channel_name)
        ch_types[ch_idx] = 'eog'

    # Update ch_types for misc channels
    for channel_name in misc_channels:
        ch_idx = ch_names.index(channel_name)
        ch_types[ch_idx] = 'misc'



    # Replace 'NaN' values with 0
    df['Events'] = df['Events'].fillna(0)

    # Convert values with more than one decimal to numeric format for all channels
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')

    # Drop rows with any missing values
    df.dropna(inplace=True)

     # Now, update ch_names and ch_types to exclude EOG and Misc channels
    ch_names = [ch for ch in ch_names if ch not in eog_channels + misc_channels]
    ch_types = ['eeg'] * len(ch_names)
    
    # Calculate the average values for A1 and A2
    a1_average = df['A1'].mean()
    a2_average = df['A2'].mean()

    # Common Average Reference according to A1 and A2
    common_average = (df.drop(['A1', 'A2', 'SensorCEOG', 'SensorDEOG', 'Events'], axis=1) - a1_average - a2_average).mean(axis=1)
    referenced_data = df.drop(['A1', 'A2', 'SensorCEOG', 'SensorDEOG', 'Events'], axis=1).sub(common_average, axis=0)

    from scipy.signal import firwin, lfilter

    # Define your desired filter specifications
    lowcut = 2.0  # Lower cutoff frequency in Hz
    highcut = 40.0  # Upper cutoff frequency in Hz
    fs = 512.0  # Sampling frequency in Hz
    numtaps = 101  # Number of filter coefficients

    # Design an FIR band-pass filter
    b = firwin(numtaps, [lowcut, highcut], fs=fs, pass_zero=False)

    # Apply the FIR filter to the referenced EEG data
    filtered_data = lfilter(b, 1.0, referenced_data, axis=0)

    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types=ch_types)

    # Create the RawArray using the filtered EEG data and 'info'
    raw_filtered = mne.io.RawArray(filtered_data.T, info) 

    from scipy.signal import firwin, lfilter

    # Define your desired filter specifications
    lowcut = 1.0  # Lower cutoff frequency in Hz
    highcut = 40.0  # Upper cutoff frequency in Hz
    fs = 512.0  # Sampling frequency in Hz
    numtaps = 101  # Number of filter coefficients

    # Design an FIR band-pass filter
    b = firwin(numtaps, [lowcut, highcut], fs=fs, pass_zero=False)

    # Apply the FIR filter to the referenced EEG data
    filtered_data = lfilter(b, 1.0, referenced_data, axis=0)


    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types=ch_types)

    # Create the RawArray using the filtered EEG data and 'info'
    raw_filtered = mne.io.RawArray(filtered_data.T, info) 

    import mne

    # Define the high-pass filter parameters
    highpass_freq = 1.0  # High-pass cutoff frequency in Hz

    # Apply the high-pass filter to the EEG data
    raw_filtered.filter(l_freq=highpass_freq, h_freq=None, fir_design='firwin')

    # Rename the channels using the provided mapping
    channel_mapping = {'FP1': 'Fp1', 'FP2': 'Fp2', 'PZ': 'Pz'}
    raw_filtered.rename_channels(channel_mapping)

    # Create a 10-20 montage for a 19-channel EEG system
    montage = mne.channels.make_standard_montage('standard_1020', head_size=1.0)

    # Apply the montage to the raw data
    raw_filtered.set_montage(montage)

    # Now you have the 'raw_filtered' object with the applied high-pass filter, channel renaming, and montage
    print(raw_filtered)

    import mne

   
    n_components=len(raw_filtered.ch_names)

    
    # Fit ICA to raw data
    ica = mne.preprocessing.ICA(n_components=n_components, random_state=97, max_iter=800)
    ica.fit(raw_filtered)

   
    


    import mne
    import numpy as np

    epoch_duration = 5  # Duration of each epoch in seconds
    event_id = 1  # Event identifier

    # Calculate event samples based on the epoch duration
    event_samples = np.arange(0, raw_filtered.n_times, int(epoch_duration * fs))  

    # Create events array with appropriate columns
    events = np.column_stack((event_samples, np.zeros(len(event_samples), dtype=int), np.full(len(event_samples), event_id)))

    
    channel_names = ['Fp1', 'Fp2','Pz',"F3",'Cz','T5','T6',"F4","P3","P4","O1","O2","F7","F8"]
    picks_ica = mne.pick_channels(raw_filtered.ch_names, include=channel_names)  # Use raw_filtered.ch_names instead of ica_raw.ch_names

    # Set a threshold for epoch rejection based on EEG amplitude
    reject_threshold = 100  # Adjust this threshold based on your data

    # Create epochs using mne.Epochs, selecting specific EEG channels
    epochs = mne.Epochs(raw_filtered, events=events, event_id=event_id, tmin=0, tmax=epoch_duration,
                        baseline=None, picks=picks_ica, preload=True,
                        reject=dict(eeg=reject_threshold), detrend=1, reject_by_annotation=False)

    

    ar = AutoReject(n_interpolate=[8], cv=5)


    ar.fit(epochs)
    epochs_cleaned = ar.transform(epochs.copy())
    
    
   
        # Get the data matrix of the epochs
    epochs_data = epochs_cleaned.get_data()

    # Check the shape of the data matrix
    if epochs_data.shape[0] > 0:
        print("Epochs are not empty")
    else:
        print("Epochs are empty")


    picks_eeg = mne.pick_types(raw_filtered.info, meg=False, eeg=True, eog=False)


    from mne.time_frequency import tfr_multitaper

    # Define the desired sampling frequency for analysis
    desired_sampling_freq = 120.0

    # Resample the data to the desired sampling frequency
    resampled_epochs = epochs_cleaned.copy()
    resampled_epochs.resample(sfreq=desired_sampling_freq)

    # Define parameters for the wavelet transform
    freqs = np.arange(2, 30, 0.5)  # Frequencies from 2 to 30 Hz with 0.5 Hz resolution
    n_cycles = freqs / 6.0  # Number of cycles for each frequency

    # Define the desired downsampling factor
    downsampling_factor = 2  # You can adjust this value as needed

    # Apply downsampling to the resampled_epochs
    downsampled_epochs = resampled_epochs.copy()
    downsampled_epochs.resample(sfreq=desired_sampling_freq / downsampling_factor)

    # Define the picks for EEG channels
    picks_eeg = mne.pick_types(downsampled_epochs.info, meg=False, eeg=True)

    # Perform Morlet wavelet transform for all epochs
    tfr_power = tfr_multitaper(downsampled_epochs, freqs=freqs, n_cycles=n_cycles,
                               time_bandwidth=2.0, return_itc=False,
                               picks=picks_eeg, average=False)


    power_values = tfr_power.data  # The shape is (n_epochs, n_channels, n_freqs, n_times)

    # Iterate through epochs and channels
    for epoch_idx in range(len(downsampled_epochs)):
        for channel_idx in range(len(picks_eeg)):
            # Extract power values for the current epoch, channel, frequency range, and all time points
            power_epoch_channel = power_values[epoch_idx, channel_idx]




   
    
    slope_values = np.zeros(len(picks_eeg))

    # Iterate through channels (electrodes)
    for channel_idx in range(len(picks_eeg)):
        # Extract power values for the current channel and all epochs, frequencies, and time points
        power_channel = tfr_power.data[:, channel_idx, :, :]

        # Calculate the 1/f slope for the current channel using OLS regression on log-log transformed data
        log_freqs = np.log(freqs)
        log_power = np.log(power_channel)

        # Exclude the 7 – 14 Hz range containing the alpha peak
        alpha_freq_indices = np.where((freqs < 7) | (freqs > 14))
        log_freqs_alpha = log_freqs[alpha_freq_indices]
        log_power_alpha = log_power[:, :, alpha_freq_indices]

        # Reshape log_power_alpha to (n_epochs * n_time_points, n_alpha_freqs)
        log_power_alpha_reshaped = log_power_alpha.reshape(-1, log_power_alpha.shape[-1])

        # Perform OLS regression
        X = np.column_stack((np.ones(log_freqs_alpha.shape), log_freqs_alpha))
        results = np.linalg.lstsq(X, log_power_alpha_reshaped.T, rcond=None)
        slope = results[0][1]  # Extract the slope coefficient

        slope_values[channel_idx] = slope[0]  


    theta_range = (4, 8)

    itf_values = np.zeros((len(picks_eeg)))  # Individual Theta Frequency (ITF)
    itp_values = np.zeros((len(picks_eeg)))  # Individual Theta Power (ITP)

    # Iterate through channels (electrodes)
    for channel_idx in range(len(picks_eeg)):
        # Extract power values for the current channel and all epochs, frequencies, and time points
        power_channel = tfr_power.data[:, channel_idx, :, :]

        # Find the indices of the frequencies within the theta range
        theta_freq_indices = np.where((freqs >= theta_range[0]) & (freqs <= theta_range[1]))

        # Calculate the ITF (Individual Theta Frequency) for the current channel and all epochs
        itf_values[channel_idx] = np.mean(freqs[theta_freq_indices][np.argmax(power_channel[:, theta_freq_indices, :], axis=2)])

        # Calculate the ITP (Individual Theta Power) for the current channel and all epochs
        itp_values[channel_idx] = np.mean(power_channel[:, theta_freq_indices])
      
        
        
        
    alpha_range = (8, 13)  

    iaf_values = np.zeros((len(picks_eeg)))

    # Iterate through channels (electrodes)
    for channel_idx in range(len(picks_eeg)):
        # Extract power values for the current channel and all epochs, frequencies, and time points
        power_channel = tfr_power.data[:, channel_idx, :, :]

        # Find the index of the frequency with the maximum power within the alpha range for each epoch
        alpha_freq_indices = np.where((freqs >= alpha_range[0]) & (freqs <= alpha_range[1]))
        max_alpha_freqs = freqs[alpha_freq_indices]
        max_alpha_freq_indices = np.argmax(power_channel[:, alpha_freq_indices, :], axis=2)
        iaf_values_all = np.mean(max_alpha_freqs[max_alpha_freq_indices], axis=1)
        iaf_values[channel_idx] = np.mean(iaf_values_all)

   
        
    iap_values = np.zeros((len(picks_eeg)))

    # Iterate through channels (electrodes)
    for channel_idx in range(len(picks_eeg)):
        # Extract power values for the current channel and all epochs, frequencies, and time points
        power_channel = tfr_power.data[:, channel_idx, :, :]

        # Find the indices of the frequencies within the alpha range
        alpha_freq_indices = np.where((freqs >= alpha_range[0]) & (freqs <= alpha_range[1]))

        # Calculate the IAP for the current channel and all epochs
        iap_values[channel_idx] = np.mean(power_channel[:, alpha_freq_indices])



    beta_range = (13, 30)

    # Initialize arrays to store the values
    ibf_values = np.zeros((len(picks_eeg)))
    ibp_values = np.zeros((len(picks_eeg)))

    # Iterate through channels (electrodes)
    for channel_idx in range(len(picks_eeg)):
        # Extract power values for the current channel and all epochs, frequencies, and time points
        power_channel = tfr_power.data[:, channel_idx, :, :]

        # Find the indices of the frequencies within the beta range
        beta_freq_indices = np.where((freqs >= beta_range[0]) & (freqs <= beta_range[1]))[0]
        beta_freqs = freqs[beta_freq_indices]

        # Find the frequency value with maximum power within the beta frequency range for each epoch
        max_power_freq_values = beta_freqs[np.argmax(power_channel[:, beta_freq_indices, :], axis=1)]

        # Calculate the IBF for the current channel and all epochs
        ibf_values[channel_idx] = np.mean(max_power_freq_values)

        # Calculate the IBP for the current channel and all epochs
        ibp_values[channel_idx] = np.mean(power_channel[:, beta_freq_indices, :])



        # Calculate the mean PSD across all channels for each frequency
        mean_psd = np.mean(power_values, axis=(0, 1))
        
    
    
    # Calculate the mean PSD values for the alpha and beta frequency ranges
    alpha_freq_indices = np.where((freqs >= alpha_range[0]) & (freqs <= alpha_range[1]))
    beta_freq_indices = np.where((freqs >= beta_range[0]) & (freqs <= beta_range[1]))

    # Initialize arrays to store mean PSD values for each channel
    mean_psd_alpha_channels = np.zeros(len(picks_eeg))
    mean_psd_beta_channels = np.zeros(len(picks_eeg))

    # Iterate through channels (electrodes)
    for channel_idx in range(len(picks_eeg)):
        # Extract power values for the current channel and all epochs, frequencies, and time points
        power_channel = tfr_power.data[:, channel_idx, :, :]

         # Calculate the mean PSD for the alpha frequency range for the current channel
        mean_psd_alpha = np.mean(power_channel[:, alpha_freq_indices, :], axis=(0, 2))
        mean_psd_alpha_channels[channel_idx] = np.mean(mean_psd_alpha)

        # Calculate the mean PSD for the beta frequency range for the current channel
        mean_psd_beta = np.mean(power_channel[:, beta_freq_indices, :], axis=(0, 2))
        mean_psd_beta_channels[channel_idx] = np.mean(mean_psd_beta)


    
    
    
    
    # Add results to the DataFrame
    results_df = results_df.append({'File': file_name,
                                    **{f'Slope_Channel_{i}': slope_values[i] for i in range(14)},
                                    **{f'ITF_Channel_{i}': itf_values[i] for i in range(14)},
                                    **{f'ITP_Channel_{i}': itp_values[i] for i in range(14)},
                                    **{f'IAF_Channel_{i}': iaf_values[i] for i in range(14)},
                                    **{f'IAP_Channel_{i}': iap_values[i] for i in range(14)},
                                    **{f'IBF_Channel_{i}': ibf_values[i] for i in range(14)},
                                    **{f'IBP_Channel_{i}': ibp_values[i] for i in range(14)},
                                    **{f'Mean_PSD_Channel_Alpha_{i}': mean_psd_alpha_channels[i] for i in range(14)},
                                    **{f'Mean_PSD_Channel_Beta_{i}': mean_psd_beta_channels[i] for i in range(14)}}, ignore_index=True)

    # Rest of your code...

# Save the results DataFrame to a CSV file
results_csv_file = 'analysis_results12Newwith Theta0910.csv'
results_df.to_csv(results_csv_file, index=False)

print("Results saved to:", results_csv_file)
    


# In[ ]:


import pandas as pd
import numpy as np
import mne
from scipy.signal import firwin, lfilter
from mne.time_frequency import tfr_multitaper
from autoreject import AutoReject

results_df = pd.DataFrame(columns=['File'] +
                                [f'Slope_Channel_{i}' for i in range(14)] +
                                [f'ITF_Channel_{i}' for i in range(14)] +
                                [f'ITP_Channel_{i}' for i in range(14)] +
                                [f'IAF_Channel_{i}' for i in range(14)] +
                                [f'IAP_Channel_{i}' for i in range(14)] +
                                [f'IBF_Channel_{i}' for i in range(14)] +
                                [f'IBP_Channel_{i}' for i in range(14)] +
                                [f'Mean_PSD_Channel_Alpha_{i}' for i in range(14)] +
                                [f'Mean_PSD_Channel_Beta_{i}' for i in range(14)])


# List of file names
file_names = ['m_med_14.txt','m_med_20.txt','m_med_24.txt','m_rest_24.txt', 'm_med_25.txt','m_med_31.txt',"m_med_7.txt","m_med_9.txt","m_rest_14.txt","m_rest_20.txt","m_rest_25.txt","m_rest_31.txt","m_rest_7.txt","m_rest_9.txt",'m_med_2.txt' , 'm_rest_2.txt' , 'm_med_3.txt' , 'm_rest_3.txt' , 'm_med_4.txt' ,'m_rest_4.txt' ,'m_med_6.txt' ,'m_rest_6.txt' ,'m_med_10.txt' ,'m_rest_10.txt' ,'m_med_11.txt' ,'m_rest_11.txt' ,'m_med_12.txt' ,'m_rest_12.txt' ,'m_med_13.txt' ,'m_rest_13.txt' ,'m_med_15.txt' ,'m_rest_15.txt' ,'m_med_16.txt' ,'m_rest_16.txt' ,'m_med_17.txt' ,'m_rest_17.txt' ,'m_med_19.txt' ,'m_rest_19.txt' ,'m_med_21.txt' ,'m_rest_21.txt' ,'m_med_22.txt' ,'m_rest_22.txt' ,'m_med_23.txt' ,'m_rest_23.txt' ,'m_med_27.txt' ,'m_rest_27.txt' ,'m_med_28.txt' ,'m_rest_28.txt' ,'m_med_29.txt' ,'m_rest_29.txt' ,'m_med_30.txt' ,'m_rest_30.txt' ,'m_med_32.txt' ,'m_rest_32.txt' ,'m_med_33.txt' ,'m_rest_33.txt' ,'m_med_34.txt' ,'m_rest_34.txt',"c_med_10.txt","c_med_12.txt","c_med_13.txt","c_med_14.txt","c_med_15.txt","c_med_16.txt","c_med_19.txt","c_med_2.txt","c_med_22.txt","c_med_23.txt","c_med_24.txt","c_med_25.txt","c_med_26.txt","c_med_29.txt","c_med_30.txt","c_med_6.txt","c_med_9.txt","c_med_n1.txt","c_med_n10.txt","c_med_n11.txt","c_med_n12.txt","c_med_n2.txt","c_med_n3.txt","c_med_n4.txt","c_med_n5.txt","c_med_n6.txt","c_med_n7.txt","c_med_n8.txt","c_med_n9.txt","c_rest_10.txt","c_rest_12.txt","c_rest_13.txt","c_rest_14.txt","c_rest_15.txt","c_rest_16.txt","c_rest_19.txt","c_rest_2.txt","c_rest_22.txt","c_rest_23.txt","c_rest_24.txt","c_rest_25.txt","c_rest_26.txt","c_rest_29.txt","c_rest_30.txt","c_rest_6.txt","c_rest_9.txt","c_rest_n1.txt","c_rest_n10.txt","c_rest_n11.txt","c_rest_n12.txt","c_rest_n2.txt","c_rest_n3.txt","c_rest_n4.txt","c_rest_n5.txt","c_rest_n6.txt","c_rest_n7.txt","c_rest_n8.txt","c_rest_n9.txt"]
for file_name in file_names:
    delimiter = ','
    df = pd.read_csv(file_name, delimiter=delimiter, encoding='latin1', on_bad_lines='skip')

    import pandas as pd

# Replace 'NaN' values in the 'Events' column with 0
    df['Events'] = df['Events'].replace('NaN', 0)


    # Convert values with more than one decimals to numeric format for all channels
    for column in df.columns:
        df[column] = df[column].astype(str)
        df[column] = df[column].str.replace('[^\d.-]', '', regex=True)
        df[column] = df[column].str.replace('\.{2,}', '.', regex=True)
        df[column] = pd.to_numeric(df[column], errors='coerce')


    df.info()

    df.isna().sum()

    df.head()

    df['Events'] = df['Events'].replace('NaN', 0)

    df.head()

    df['Events'] = df['Events'].fillna(0)
    
    column_averages=df.mean()

    for column in df.columns:
        df[column].fillna(column_averages[column], inplace=True)

    df.head()

    import numpy as np

    import mne


    # Update channel names and types for ICA
    ch_names = ['FP1', 'FP2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4',
                'T4', 'T5', 'P3', 'PZ', 'P4', 'T6', 'O1', 'O2', 'A1', 'A2', 'SensorCEOG', 'SensorDEOG', 'Events']

    # Start with all channels as 'eeg'
    ch_types = ['eeg'] * len(ch_names)

    # Define EOG and misc channels
    eog_channels = ['SensorCEOG', 'SensorDEOG']
    misc_channels = ['A1', 'A2']
    misc_channels.append('Events')

    # Update ch_types for EOG channels
    for channel_name in eog_channels:
        ch_idx = ch_names.index(channel_name)
        ch_types[ch_idx] = 'eog'

    # Update ch_types for misc channels
    for channel_name in misc_channels:
        ch_idx = ch_names.index(channel_name)
        ch_types[ch_idx] = 'misc'



    # Replace 'NaN' values with 0
    df['Events'] = df['Events'].fillna(0)

    # Convert values with more than one decimal to numeric format for all channels
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')

    # Drop rows with any missing values
    df.dropna(inplace=True)

     # Now, update ch_names and ch_types to exclude EOG and Misc channels
    ch_names = [ch for ch in ch_names if ch not in eog_channels + misc_channels]
    ch_types = ['eeg'] * len(ch_names)
    
    # Calculate the average values for A1 and A2
    a1_average = df['A1'].mean()
    a2_average = df['A2'].mean()

    # Common Average Reference according to A1 and A2
    common_average = (df.drop(['A1', 'A2', 'SensorCEOG', 'SensorDEOG', 'Events'], axis=1) - a1_average - a2_average).mean(axis=1)
    referenced_data = df.drop(['A1', 'A2', 'SensorCEOG', 'SensorDEOG', 'Events'], axis=1).sub(common_average, axis=0)

    from scipy.signal import firwin, lfilter

    # Define your desired filter specifications
    lowcut = 2.0  # Lower cutoff frequency in Hz
    highcut = 40.0  # Upper cutoff frequency in Hz
    fs = 512.0  # Sampling frequency in Hz
    numtaps = 101  # Number of filter coefficients

    # Design an FIR band-pass filter
    b = firwin(numtaps, [lowcut, highcut], fs=fs, pass_zero=False)

    # Apply the FIR filter to the referenced EEG data
    filtered_data = lfilter(b, 1.0, referenced_data, axis=0)

    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types=ch_types)

    # Create the RawArray using the filtered EEG data and 'info'
    raw_filtered = mne.io.RawArray(filtered_data.T, info) 

    from scipy.signal import firwin, lfilter

    # Define your desired filter specifications
    lowcut = 1.0  # Lower cutoff frequency in Hz
    highcut = 40.0  # Upper cutoff frequency in Hz
    fs = 512.0  # Sampling frequency in Hz
    numtaps = 101  # Number of filter coefficients

    # Design an FIR band-pass filter
    b = firwin(numtaps, [lowcut, highcut], fs=fs, pass_zero=False)

    # Apply the FIR filter to the referenced EEG data
    filtered_data = lfilter(b, 1.0, referenced_data, axis=0)


    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types=ch_types)

    # Create the RawArray using the filtered EEG data and 'info'
    raw_filtered = mne.io.RawArray(filtered_data.T, info) 

    import mne

    # Define the high-pass filter parameters
    highpass_freq = 1.0  # High-pass cutoff frequency in Hz

    # Apply the high-pass filter to the EEG data
    raw_filtered.filter(l_freq=highpass_freq, h_freq=None, fir_design='firwin')

    # Rename the channels using the provided mapping
    channel_mapping = {'FP1': 'Fp1', 'FP2': 'Fp2', 'PZ': 'Pz'}
    raw_filtered.rename_channels(channel_mapping)

    # Create a 10-20 montage for a 19-channel EEG system
    montage = mne.channels.make_standard_montage('standard_1020', head_size=1.0)

    # Apply the montage to the raw data
    raw_filtered.set_montage(montage)

    # Now you have the 'raw_filtered' object with the applied high-pass filter, channel renaming, and montage
    print(raw_filtered)

    import mne

   
    n_components=len(raw_filtered.ch_names)

    
    # Fit ICA to raw data
    ica = mne.preprocessing.ICA(n_components=n_components, random_state=97, max_iter=800)
    ica.fit(raw_filtered)

   
    


    import mne
    import numpy as np

    epoch_duration = 5  # Duration of each epoch in seconds
    event_id = 1  # Event identifier

    # Calculate event samples based on the epoch duration
    event_samples = np.arange(0, raw_filtered.n_times, int(epoch_duration * fs))  

    # Create events array with appropriate columns
    events = np.column_stack((event_samples, np.zeros(len(event_samples), dtype=int), np.full(len(event_samples), event_id)))

    
    channel_names = ['Fp1', 'Fp2','Pz',"F3",'Cz','T5','T6',"F4","P3","P4","O1","O2","F7","F8"]
    picks_ica = mne.pick_channels(raw_filtered.ch_names, include=channel_names)  # Use raw_filtered.ch_names instead of ica_raw.ch_names

    # Set a threshold for epoch rejection based on EEG amplitude
    reject_threshold = 100  # Adjust this threshold based on your data

    # Create epochs using mne.Epochs, selecting specific EEG channels
    epochs = mne.Epochs(raw_filtered, events=events, event_id=event_id, tmin=0, tmax=epoch_duration,
                        baseline=None, picks=picks_ica, preload=True,
                        reject=dict(eeg=reject_threshold), detrend=1, reject_by_annotation=False)

    

    ar = AutoReject(n_interpolate=[8], cv=5)


    ar.fit(epochs)
    epochs_cleaned = ar.transform(epochs.copy())
    
    
   
        # Get the data matrix of the epochs
    epochs_data = epochs_cleaned.get_data()

    # Check the shape of the data matrix
    if epochs_data.shape[0] > 0:
        print("Epochs are not empty")
    else:
        print("Epochs are empty")


    picks_eeg = mne.pick_types(raw_filtered.info, meg=False, eeg=True, eog=False)


    from mne.time_frequency import tfr_multitaper

    # Define the desired sampling frequency for analysis
    desired_sampling_freq = 120.0

    # Resample the data to the desired sampling frequency
    resampled_epochs = epochs_cleaned.copy()
    resampled_epochs.resample(sfreq=desired_sampling_freq)

    # Define parameters for the wavelet transform
    freqs = np.arange(2, 30, 0.5)  # Frequencies from 2 to 30 Hz with 0.5 Hz resolution
    n_cycles = freqs / 6.0  # Number of cycles for each frequency

    # Define the desired downsampling factor
    downsampling_factor = 2  # You can adjust this value as needed

    # Apply downsampling to the resampled_epochs
    downsampled_epochs = resampled_epochs.copy()
    downsampled_epochs.resample(sfreq=desired_sampling_freq / downsampling_factor)

    # Define the picks for EEG channels
    picks_eeg = mne.pick_types(downsampled_epochs.info, meg=False, eeg=True)

    # Perform Morlet wavelet transform for all epochs
    tfr_power = tfr_multitaper(downsampled_epochs, freqs=freqs, n_cycles=n_cycles,
                               time_bandwidth=2.0, return_itc=False,
                               picks=picks_eeg, average=False)


    power_values = tfr_power.data  # The shape is (n_epochs, n_channels, n_freqs, n_times)

    # Iterate through epochs and channels
    for epoch_idx in range(len(downsampled_epochs)):
        for channel_idx in range(len(picks_eeg)):
            # Extract power values for the current epoch, channel, frequency range, and all time points
            power_epoch_channel = power_values[epoch_idx, channel_idx]




   
    
    slope_values = np.zeros(len(picks_eeg))

    # Iterate through channels (electrodes)
    for channel_idx in range(len(picks_eeg)):
        # Extract power values for the current channel and all epochs, frequencies, and time points
        power_channel = tfr_power.data[:, channel_idx, :, :]

        # Calculate the 1/f slope for the current channel using OLS regression on log-log transformed data
        log_freqs = np.log(freqs)
        log_power = np.log(power_channel)

        # Exclude the 7 – 14 Hz range containing the alpha peak
        alpha_freq_indices = np.where((freqs < 7) | (freqs > 14))
        log_freqs_alpha = log_freqs[alpha_freq_indices]
        log_power_alpha = log_power[:, :, alpha_freq_indices]

        # Reshape log_power_alpha to (n_epochs * n_time_points, n_alpha_freqs)
        log_power_alpha_reshaped = log_power_alpha.reshape(-1, log_power_alpha.shape[-1])

        # Perform OLS regression
        X = np.column_stack((np.ones(log_freqs_alpha.shape), log_freqs_alpha))
        results = np.linalg.lstsq(X, log_power_alpha_reshaped.T, rcond=None)
        slope = results[0][1]  # Extract the slope coefficient

        slope_values[channel_idx] = slope[0]  


    theta_range = (4, 8)

    itf_values = np.zeros((len(picks_eeg)))  # Individual Theta Frequency (ITF)
    itp_values = np.zeros((len(picks_eeg)))  # Individual Theta Power (ITP)

    # Iterate through channels (electrodes)
    for channel_idx in range(len(picks_eeg)):
        # Extract power values for the current channel and all epochs, frequencies, and time points
        power_channel = tfr_power.data[:, channel_idx, :, :]

        # Find the indices of the frequencies within the theta range
        theta_freq_indices = np.where((freqs >= theta_range[0]) & (freqs <= theta_range[1]))

        # Calculate the ITF (Individual Theta Frequency) for the current channel and all epochs
        itf_values[channel_idx] = np.mean(freqs[theta_freq_indices][np.argmax(power_channel[:, theta_freq_indices, :], axis=2)])

        # Calculate the ITP (Individual Theta Power) for the current channel and all epochs
        itp_values[channel_idx] = np.mean(power_channel[:, theta_freq_indices])
      
        
        
        
    alpha_range = (8, 13)  

    iaf_values = np.zeros((len(picks_eeg)))

    # Iterate through channels (electrodes)
    for channel_idx in range(len(picks_eeg)):
        # Extract power values for the current channel and all epochs, frequencies, and time points
        power_channel = tfr_power.data[:, channel_idx, :, :]

        # Find the index of the frequency with the maximum power within the alpha range for each epoch
        alpha_freq_indices = np.where((freqs >= alpha_range[0]) & (freqs <= alpha_range[1]))
        max_alpha_freqs = freqs[alpha_freq_indices]
        max_alpha_freq_indices = np.argmax(power_channel[:, alpha_freq_indices, :], axis=2)
        iaf_values_all = np.mean(max_alpha_freqs[max_alpha_freq_indices], axis=1)
        iaf_values[channel_idx] = np.mean(iaf_values_all)

   
        
    iap_values = np.zeros((len(picks_eeg)))

    # Iterate through channels (electrodes)
    for channel_idx in range(len(picks_eeg)):
        # Extract power values for the current channel and all epochs, frequencies, and time points
        power_channel = tfr_power.data[:, channel_idx, :, :]

        # Find the indices of the frequencies within the alpha range
        alpha_freq_indices = np.where((freqs >= alpha_range[0]) & (freqs <= alpha_range[1]))

        # Calculate the IAP for the current channel and all epochs
        iap_values[channel_idx] = np.mean(power_channel[:, alpha_freq_indices])



    beta_range = (13, 30)

    # Initialize arrays to store the values
    ibf_values = np.zeros((len(picks_eeg)))
    ibp_values = np.zeros((len(picks_eeg)))

    # Iterate through channels (electrodes)
    for channel_idx in range(len(picks_eeg)):
        # Extract power values for the current channel and all epochs, frequencies, and time points
        power_channel = tfr_power.data[:, channel_idx, :, :]

        # Find the indices of the frequencies within the beta range
        beta_freq_indices = np.where((freqs >= beta_range[0]) & (freqs <= beta_range[1]))[0]
        beta_freqs = freqs[beta_freq_indices]

        # Find the frequency value with maximum power within the beta frequency range for each epoch
        max_power_freq_values = beta_freqs[np.argmax(power_channel[:, beta_freq_indices, :], axis=1)]

        # Calculate the IBF for the current channel and all epochs
        ibf_values[channel_idx] = np.mean(max_power_freq_values)

        # Calculate the IBP for the current channel and all epochs
        ibp_values[channel_idx] = np.mean(power_channel[:, beta_freq_indices, :])



        # Calculate the mean PSD across all channels for each frequency
        mean_psd = np.mean(power_values, axis=(0, 1))
        
    
    
    # Calculate the mean PSD values for the alpha and beta frequency ranges
    alpha_freq_indices = np.where((freqs >= alpha_range[0]) & (freqs <= alpha_range[1]))
    beta_freq_indices = np.where((freqs >= beta_range[0]) & (freqs <= beta_range[1]))

    # Initialize arrays to store mean PSD values for each channel
    mean_psd_alpha_channels = np.zeros(len(picks_eeg))
    mean_psd_beta_channels = np.zeros(len(picks_eeg))

    # Iterate through channels (electrodes)
    for channel_idx in range(len(picks_eeg)):
        # Extract power values for the current channel and all epochs, frequencies, and time points
        power_channel = tfr_power.data[:, channel_idx, :, :]

         # Calculate the mean PSD for the alpha frequency range for the current channel
        mean_psd_alpha = np.mean(power_channel[:, alpha_freq_indices, :], axis=(0, 2))
        mean_psd_alpha_channels[channel_idx] = np.mean(mean_psd_alpha)

        # Calculate the mean PSD for the beta frequency range for the current channel
        mean_psd_beta = np.mean(power_channel[:, beta_freq_indices, :], axis=(0, 2))
        mean_psd_beta_channels[channel_idx] = np.mean(mean_psd_beta)


    
    
    
    
    # Add results to the DataFrame
    results_df = results_df.append({'File': file_name,
                                    **{f'Slope_Channel_{i}': slope_values[i] for i in range(14)},
                                    **{f'ITF_Channel_{i}': itf_values[i] for i in range(14)},
                                    **{f'ITP_Channel_{i}': itp_values[i] for i in range(14)},
                                    **{f'IAF_Channel_{i}': iaf_values[i] for i in range(14)},
                                    **{f'IAP_Channel_{i}': iap_values[i] for i in range(14)},
                                    **{f'IBF_Channel_{i}': ibf_values[i] for i in range(14)},
                                    **{f'IBP_Channel_{i}': ibp_values[i] for i in range(14)},
                                    **{f'Mean_PSD_Channel_Alpha_{i}': mean_psd_alpha_channels[i] for i in range(14)},
                                    **{f'Mean_PSD_Channel_Beta_{i}': mean_psd_beta_channels[i] for i in range(14)}}, ignore_index=True)

    # Rest of your code...

# Save the results DataFrame to a CSV file
results_csv_file = 'analysis_results12Newwith Theta0910.csv'
results_df.to_csv(results_csv_file, index=False)

print("Results saved to:", results_csv_file)
    

