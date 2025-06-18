import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sp
import os
import datetime
from processing_helpers import *
from peek_func import *
from parameters import *

data_processing = PROCESSING
DONT_USE_SECOND_PEAK = SECOND_PEAK
main_dir = DIR

#/Users/felix/Documents/McGill_Bsc/Radio Lab/singing poles/apr2025_deploy_data_code/data_analysis

### main code ######################################################################

script_directory = os.path.dirname(os.path.abspath(__file__))
box_n=make_boxes()

if data_processing:
    unify_data(main_dir)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    script_directory = os.path.join(base_dir, UNIFIED_FOLDER)

    files = [f for f in os.listdir(script_directory) if os.path.isfile(os.path.join(script_directory, f))]
    box_n = []

    i=0
    while i < len(files): #find the data and motion files
        if files[i][:4] == "data":
            box_n.append(int(files[i].strip('_.datcsv')))
        if files[i][:6] == "motion":
            box_n.append(int(files[i].strip('_.motinb')))
        i+=1

    box_n = list(set(box_n)) #list of unique box numbers
    box_n.sort()

    print("All boxes:",box_n)
            
    names = box_n
    for i in range(len(names)):
        names[i] = str(names[i])

    trials = len(names)
    lengths = []

    all_comp = np.nan #all the computed lenghts from the frequecies
    err_comp = np.nan #error on the computed lengths

    data_dtype = np.dtype([('epoch_time', np.int32),('vReal', np.float32, (2048,))])

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
        
        data_tit = os.path.join(script_directory, "data_" + names[u] + ".csv")
        motion_tit = os.path.join(script_directory, "motion_" + names[u] + ".bin")

        all_data = np.genfromtxt(data_tit, delimiter=",")
        all_motion = np.fromfile(motion_tit, dtype=data_dtype)
        all_motion = np.column_stack((all_motion['epoch_time'], all_motion['vReal']))
        
        data_time_col = 2
        motion_time_col = 0
        data_times = all_data[:,data_time_col]
        motion_times = all_motion[:,motion_time_col]
        times = np.append(data_times, motion_times)
        times = np.sort(np.array(list(set(times))))
        
        total_col_final_data = 8
        final_data = np.zeros((len(times), total_col_final_data)) #for csv file: time, temperature, motion length, error, data length, error, depth sensor length, error
        final_data[:,time_col] = times
        
        all_motion = all_motion[:,motion_time_col+1:-1] # shape (n, 2048) skips nan values at the end and time values in the beginning
        
        for time in times:
        
            motion = all_motion[motion_times==time,:]
            data = all_data[data_times==time,:]
            
            data_peak1_col = 4
            data_peak2_col = 5
            frequency_corr_factor = 2
            max_freqs_data = data[:,data_peak1_col:data_peak2_col+1]*frequency_corr_factor #get the two main frequencies, they need to be multiplied by two to work with other standards in the code
            if DONT_USE_SECOND_PEAK: #can be useful if the second peak is bad / wrong
                max_freqs_data[:,1]=0
            
            all_rec_freq_data = norm_harmonic(max_freqs_data)
            
            all_comp, err_comp = sample_selection(all_rec_freq_data)
            
            final_data[final_data[:,time_col]==time,data_col] = all_comp
            final_data[final_data[:,time_col]==time,data_err_col] = err_comp
            
            MOTION_LEN = np.shape(motion)[1]
            
            window = np.blackman(MOTION_LEN)
            motion *= window #makes the data cleaner
        
            sampling_frequency = 200
            fft_result = np.fft.fft(motion, axis=1)
            freq_x = np.fft.fftfreq(MOTION_LEN, 1/sampling_frequency) #frequencies on the x axis

            amp_fft = np.abs(fft_result)[:,:len(freq_x)//2] #amplitudes found via the fft
            pos_freq = freq_x[:len(freq_x)//2] #ordered frequencies on the x axis

            index_list = np.arange(0,MOTION_LEN//2)

            all_freqs_motion = np.zeros((np.shape(motion)[0],2))

            for i in range(0, np.shape(motion)[0]): #iterates over all data samples

                max_amp = 0
                max_i = np.array([0,0])
                current_max_i = 0
                start_index = 2
                end_index = MOTION_LEN//5
                for l in range(2): #finds the 2 main peaks for every data sample
                    avg = 0
                    for k in range(start_index, end_index): #DATA_LEN//5 means that we ignore very high frequencies
                        avg += amp_fft[i,k] #finds the average
                    avg /= end_index - start_index
                    for k in range(start_index, end_index):
                        near_index = 3 #needs to be max 3 indicies away to be considered near
                        near1 = abs(k-max_i[0]) <= near_index
                        near2 = abs(k-max_i[1]) <= near_index
                        #near3 = abs(k-max_i[2]) <= near_index
                        if not (near1 or near2):
                            noise_thresh = 10
                            if amp_fft[i,k] > max_amp and amp_fft[i,k] >= noise_thresh*avg and amp_fft[i,k] > amp_fft[i,k+1] and amp_fft[i,k] > amp_fft[i,k-1]:
                                # if it has max amplitude, is not noise, and is actully a local max
                                max_amp = amp_fft[i,k]
                                current_max_i = k
                    max_i[l] = current_max_i
                    current_max_i = 0
                    max_amp = 0

                max_freqs = max_i*sampling_frequency/(MOTION_LEN//2)
                if DONT_USE_SECOND_PEAK:
                    max_freqs[1] = 0
                
                all_freqs_motion[i] = max_freqs
            
            all_freqs = norm_harmonic(all_freqs_motion)
            
            all_comp, err_comp = sample_selection(all_freqs)
            
            data_temp_col = 3
            data_depth_col = 6
            cm_to_m = 100
            
            #for csv file: time, temperature, motion length, error, data length, error, depth sensor length, error
            final_data[final_data[:,time_col]==time,temp_col] = np.mean(data[:,data_temp_col])
            
            final_data[final_data[:,time_col]==time,mot_col] = all_comp
            final_data[final_data[:,time_col]==time,mot_err_col] = err_comp
            
            final_data[final_data[:,time_col]==time,depth_col] = np.nanmean(data[:,data_depth_col])/cm_to_m
            final_data[final_data[:,time_col]==time,depth_err_col] = np.nanstd(data[:,data_depth_col])/np.sqrt(np.shape(data[:,data_depth_col]))/cm_to_m

        final_data[:,time_col] = np.array([unix_to_yyyymmddhh(t) for t in times])
        
        processed_dir = os.path.join(base_dir, PROCESSED_FOLDER)
        os.makedirs(processed_dir, exist_ok=True)
        
        final_file_title = os.path.join(processed_dir, f'processed_data_{box_n[u]}.csv')
        header="Time,Temperature,Motion Length,Error,Data Length,Error,Depth Sensor Length,Error"
        np.savetxt(final_file_title, final_data, delimiter=",", header=header, comments="")
        print(final_file_title, "saved, shape", np.shape(final_data))
        
    print("\nDone\n")



