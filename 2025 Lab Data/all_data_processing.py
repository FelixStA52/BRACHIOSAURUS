import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sp
import os
import datetime
import freq_estimator
from processing_helpers import *
from peek_func import *
from parameters import *
import warnings

if IGNORE_RUNTIME_WARNINGS:
    warnings.filterwarnings("ignore", category=RuntimeWarning) 

data_processing = PROCESSING
DONT_USE_SECOND_PEAK = SECOND_PEAK
main_dir = DIR

#/Users/felix/Documents/McGill_Bsc/Radio Lab/singing poles/apr2025_deploy_data_code/data_analysis

### main code ######################################################################

script_directory = os.path.dirname(os.path.abspath(__file__))
box_n=make_boxes()

if data_processing:
    
    print("All boxes:",box_n,"\n")
    
    unify_data(main_dir)

    base_dir = os.path.dirname(os.path.abspath(__file__))
            
    names = box_n
    for i in range(len(names)):
        names[i] = str(names[i])

    trials = len(names)
    lengths = []

    all_comp = np.nan #all the computed lenghts from the frequecies
    err_comp = np.nan #error on the computed lengths

    data_dtype = np.dtype([('epoch_time', np.int32),('vReal', np.float32, (ARRAY_SIZE,))])

    for u in range(np.shape(box_n)[0]):
        
        #processed_data.csv: time, temperature, motion length, error, data length, error, depth sensor length, error
        
        time_col = 0
        temp_col = 1
        mot_col = 2
        mot_err_col = 3
        data_col = 4
        data_err_col = 5
        depth_col = 6
        depth_err_col = 7
        f1_col = 8
        f1_err_col = 9
        f2_col = 10
        f2_err_col = 11
        
        data_tit = os.path.join(script_directory, UNIFIED_FOLDER, "data_" + names[u] + ".csv")
        motion_tit = os.path.join(script_directory, UNIFIED_FOLDER, "motion_" + names[u] + ".bin")

        all_data = np.genfromtxt(data_tit, delimiter=",")
        all_motion = np.fromfile(motion_tit, dtype=data_dtype)
        all_motion = np.column_stack((all_motion['epoch_time'], all_motion['vReal']))
        
        data_time_col = 2
        motion_time_col = 0
        data_times = all_data[:,data_time_col]
        motion_times = all_motion[:,motion_time_col]
        times = np.append(data_times, motion_times)
        times = np.sort(np.array(list(set(times))))
        
        total_col_final_data = 12
        final_data = np.zeros((len(times), total_col_final_data)) #for csv file: time, temperature, motion length, error, data length, error, depth sensor length, error
        final_data[:,time_col] = times
        
        all_motion = all_motion[:,motion_time_col+1:-1] # shape (n, 2048) skips nan values at the end and time values in the beginning
        
        for time in times:
        
            motion = all_motion[motion_times==time,:]
            data = all_data[data_times==time,:]
            
            data_peak1_col = 4
            data_peak2_col = 5
            frequency_corr_factor = 1
            max_freqs_data = data[:,data_peak1_col:data_peak2_col+1]*frequency_corr_factor #get the two main frequencies, they need to be multiplied by two to work with other standards in the code
            if DONT_USE_SECOND_PEAK: #can be useful if the second peak is bad / wrong
                max_freqs_data[:,1]=0
            
            max_freqs_data = max_freqs_data[max_freqs_data>MIN_FREQ]
            all_rec_freq_data, all_rec_freq_data_f2 = frequency_analysis(max_freqs_data.flatten().tolist())
            
            ll1, ll2 = freq_to_len(all_rec_freq_data, all_rec_freq_data_f2)
            
            all_comp = np.mean(ll1)
            err_comp = np.std(ll1)/np.sqrt(np.shape(ll1)[0])
            
            final_data[final_data[:,time_col]==time,data_col] = all_comp
            final_data[final_data[:,time_col]==time,data_err_col] = err_comp
            
            MOTION_LEN = np.shape(motion)[1]
            
            window = np.blackman(MOTION_LEN)
            motion *= window #makes the data cleaner
        
            sampling_frequency = SAMP_FREQ
            fft_result = np.fft.fft(motion, axis=1)
            freq_x = np.fft.fftfreq(MOTION_LEN, 1/sampling_frequency) #frequencies on the x axis

            amp_fft = np.abs(fft_result)[:,:len(freq_x)//2] #amplitudes found via the fft
            pos_freq = freq_x[:len(freq_x)//2] #ordered frequencies on the x axis

            index_list = np.arange(0,MOTION_LEN//2)
            
            # Extract frequencies from peak_info
            freq1 = []
            freq1_harmonic = []
            freq2 = []
            freq2_harmonic = []
            freq2_modulated = []
            others = []

            for i in range(0, np.shape(motion)[0]):
                amp_fft = np.abs(fft_result)[i, :len(pos_freq)]
            
                # Calculate average amplitude for noise threshold
                avg_amp = np.mean(amp_fft)
                peak_info = identify_peaks(amp_fft, pos_freq, noise_threshold=GLOBAL_NOISE_THRESH*avg_amp)
                
                # Always include fundamental1
                if peak_info['fundamental1']['freq']:
                    freq1.append(peak_info['fundamental1']['freq'])
                
                if peak_info['fundamental1']['harmonics']:
                    freq1_harmonic += peak_info['fundamental1']['harmonics']
            
                # Include fundamental2 if found
                if peak_info['fundamental2']['freq']:
                    freq2.append(peak_info['fundamental2']['freq'])
                    
                if peak_info['fundamental2']['harmonics']:
                    freq2_harmonic += peak_info['fundamental2']['harmonics']
                    
                if peak_info['fundamental2']['modulated']:
                    freq2_modulated += peak_info['fundamental2']['modulated']
                    
                if peak_info['other_peaks']:
                    others += peak_info['other_peaks']
            
            all_freqs = (freq1, freq1_harmonic, freq2, freq2_harmonic, freq2_modulated, others)
            filtered_f1, filtered_f2 = frequency_analysis(*all_freqs)
            
            if np.shape(filtered_f1)[0] > 0:
                final_data[final_data[:,time_col]==time,f1_col] = np.nanmean(filtered_f1)
                final_data[final_data[:,time_col]==time,f1_err_col] = np.nanstd(filtered_f1)/np.sqrt(np.shape(filtered_f1))
                
            if np.shape(filtered_f2)[0] > 0:
                final_data[final_data[:,time_col]==time,f2_col] = np.nanmean(filtered_f2)
                final_data[final_data[:,time_col]==time,f2_err_col] = np.nanstd(filtered_f2)/np.sqrt(np.shape(filtered_f2))
            
            data_temp_col = 3
            temperature = np.mean(data[:,data_temp_col])
            if np.isnan(temperature):
                print("Used default temperature for", time, "\n")
                temperature = TEMP_GUESS[datetime.datetime.utcfromtimestamp(time).month - 1]

            len1, len2 = freq_to_len(filtered_f1, filtered_f2, r_outer=R_O, r_inner=R_I, young=youngs_modulus(temperature), rho=DENSITY, m_tip=MASS)
            
            filtered_all_freq = vote_select(np.append(len1,len2))
            all_comp = np.mean(filtered_all_freq)
            err_comp = np.std(filtered_all_freq)/np.sqrt(np.shape(filtered_all_freq)[0])
            
            data_depth_col = 6
            cm_to_m = 100
            
            #for csv file: time, temperature, motion length, error, data length, error, depth sensor length, error
            final_data[final_data[:,time_col]==time,temp_col] = np.mean(data[:,data_temp_col])
            
            final_data[final_data[:,time_col]==time,mot_col] = all_comp
            final_data[final_data[:,time_col]==time,mot_err_col] = err_comp
            
            final_data[final_data[:,time_col]==time,depth_col] = np.nanmean(data[:,data_depth_col])/cm_to_m
            final_data[final_data[:,time_col]==time,depth_err_col] = np.nanstd(data[:,data_depth_col])/np.sqrt(np.shape(data[:,data_depth_col]))/cm_to_m

        final_data[:,time_col] = np.array([unix_to_yyyymmddhhmm(t) for t in times])
        
        processed_dir = os.path.join(base_dir, PROCESSED_FOLDER)
        os.makedirs(processed_dir, exist_ok=True)
        
        final_file_title = os.path.join(processed_dir, f'processed_data_{box_n[u]}.csv')
        header="Time,Temperature,Motion Length,Error,Data Length,Error,Depth Sensor Length,Error,F1,Error,F2,Error"
        np.savetxt(final_file_title, final_data, delimiter=",", header=header, comments="")
        print("[Box " + str(box_n[u]) + "]", final_file_title, "saved, shape", np.shape(final_data),"\n")
        
    print("Done processing data")
    print("Peek options:")
    print('\tpeek_data("peak1", "YYYYMMDDHH", "YYYYMMDDHH") for the data as processed by the ESP32 code')
    print('\tpeek_pdata("all", "YYYYMMDDHH", "YYYYMMDDHH", y_lim=(0,5)) for the data as processed by this algorithm')
    print('\tpeek_motion_sum("YYYYMMDDHH", "YYYYMMDDHH") for a proxi to wind strength')
    print('\tpeek_spectrum("YYYYMMDDHH") to see the motion amplitude as a function of frequency')
    print('\tpeek_acc("YYYYMMDDHH") to see the acceleration amplitude as a function of time')
    print('\nFeel free to use help(peek_function) for a full docstring, explanation, and example uses')
    


