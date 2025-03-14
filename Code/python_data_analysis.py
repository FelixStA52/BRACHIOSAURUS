import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sp
import os
import datetime

DONT_USE_SECOND_PEAK = False

#/Users/felix/Documents/McGill_Bsc/Summer 2024/singing poles/final/bogus_data

def unify_data(main_dir):
    current_dir = os.getcwd()
    
    data_groups = {}
    motion_groups = {}

    for dir_name in os.listdir(main_dir):
        dir_path = os.path.join(main_dir, dir_name)
        if not os.path.isdir(dir_path):
            continue
        
        parts = dir_name.split('_')
        if len(parts) < 3:
            continue
            
        try:
            n = int(parts[-1])
            if dir_name.startswith('data_files'):
                data_groups[n] = dir_path
            elif dir_name.startswith('motion_files'):
                motion_groups[n] = dir_path
        except ValueError:
            continue

    for n, data_dir in data_groups.items():
        csv_files = [
            f for f in os.listdir(data_dir)
            if f.startswith(f'data_{n}_') and f.endswith('.csv')
        ]
        
        if not csv_files:
            continue
            
        csv_files.sort(key=lambda x: x.split('_')[2].split('.')[0])
        output_path = os.path.join(current_dir, f'data_{n}.csv')
        
        with open(output_path, 'w') as outfile:
            first = True
            for filename in csv_files:
                with open(os.path.join(data_dir, filename)) as infile:
                    header = infile.readline()
                    if first:
                        outfile.write(header)
                        first = False
                    else:
                        infile.readline()
                    outfile.writelines(infile.readlines())

    for n, motion_dir in motion_groups.items():
        bin_files = [
            f for f in os.listdir(motion_dir)
            if f.startswith(f'motion_{n}_') and f.endswith('.bin')
        ]
        
        if not bin_files:
            continue
        
        bin_files.sort(key=lambda x: x.split('_')[2].split('.')[0])
        output_path = os.path.join(current_dir, f'motion_{n}.bin')
        
        dtype = np.dtype([('epoch_time', np.int32), ('vReal', np.float32, (2048,))])
        combined = np.empty(0, dtype=dtype)
        
        for filename in bin_files:
            file_path = os.path.join(motion_dir, filename)
            data = np.fromfile(file_path, dtype=dtype)
            combined = np.concatenate((combined, data))
        print("Compiled files")
        combined.tofile(output_path)
        
main_dir = input("Directory to the data and motion folders: ")

unify_data(main_dir)

script_directory = os.path.dirname(os.path.abspath(__file__))

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

def f(l, k, c, m=0.638, s=0.533): #function for frequency as a function of length
    '''
    l = measured length
    k = 3EI = spring "constant"
    c = c #note that the c is squared, so a negative value is acceptable (the same as a positive)
    m = mass of the electronics
    s = linear mass density of the pole
    '''
    
    return np.sqrt((k*((m+s*l)**2)/((m*l+s*(l**2)/2)**3)) - ((c/(m+s*l))**2) )

def zero_fffunc(l, freq_solve, k=182.64026108, c=3.91926779):
    """ function allowing to find the the length from the frequency"""
    return np.abs(freq_solve - f(l, k, c))

def norm_harmonic(max_freqs_data):
    """
    Function that takes the n x 2 array containing the frequency peaks
    Returns the rectified frequencies (brought back down to what is probably
    the lowest harmonic of the oscillation)
    """
    all_rec_freq_data = np.zeros(np.shape(max_freqs_data)[0])
    for m in range(np.shape(max_freqs_data)[0]):
        max_freqs = max_freqs_data[m]
        first_zero = False
        second_zero = False

        if max_freqs[0]==0: #looks to see if some frequencies == 0
            first_zero = True
        elif max_freqs[1]==0:
            second_zero = True
        else:
            ratio = max_freqs[0]/max_freqs[1]
        
        accepted_ratio_diff_half = 0.05
        accepted_ratio_diff_twice = 0.15
        
        if first_zero and second_zero: #if both are 0, null result
            rec_freq = np.nan
        elif first_zero: #if one peak is 0, only the other is good data
            rec_freq = max_freqs[1]
        elif second_zero: #if one peak is 0, only the other is good data
            rec_freq = max_freqs[0]
        elif abs(ratio-0.5)<accepted_ratio_diff_half: #looking for which peak is the bigger one
            rec_freq = (max_freqs[0] + (max_freqs[1]/2))/2
        elif abs(ratio-2)<accepted_ratio_diff_twice:
            rec_freq = ((max_freqs[0]/2) + max_freqs[1])/2
        else: #in case the peaks are simply not related (probably bad data)
            rec_freq = np.nan
                    
        all_rec_freq_data[m] = rec_freq
        
    all_rec_freq_data[all_rec_freq_data==0] = np.nan
    all_rec_freq_data = all_rec_freq_data[~np.isnan(all_rec_freq_data)]
    all_rec_freq_data = np.sort(all_rec_freq_data)
    
    return all_rec_freq_data

def sample_selection(all_rec_freq_data):
    """
    Takes in all the rectified data from a given sample
    Outputs new rectified frequencies based on what should be
    the lowest harmonic frequency of this much larger set of data
    """
    n = np.shape(all_rec_freq_data)[0]
    min_n = 3
    if n > min_n: #if the set of data is large enough, tries to bring all the frequencies back down to lowest frequency
        mid = n//2 - 1
        mod = all_rec_freq_data[mid]
        
        accepted_ratio_diff_half = 0.05
        accepted_ratio_diff_twice = 0.15
        accepted_ratio_diff_same = 0.1
        
        for v in range(n):  
            ratio = all_rec_freq_data[v]/mod
                
            #this does the same data selection as before, but on the entire set as opposed to the 2 main peaks
            if abs(ratio-0.5)<accepted_ratio_diff_half:
                all_rec_freq_data[v] = all_rec_freq_data[v]*2
            elif abs(ratio-2)<accepted_ratio_diff_twice:
                all_rec_freq_data[v] = all_rec_freq_data[v]/2
            elif not(abs(ratio-1)<accepted_ratio_diff_same):
                #print("Deleted", all_rec_freq_data[v])
                all_rec_freq_data[v] = np.nan
                 
        all_rec_freq_data = all_rec_freq_data[~np.isnan(all_rec_freq_data)]
        
        comp_lens = []
        for w in range(np.shape(all_rec_freq_data)[0]): #uses the Length v. freq function to determine the length from each freq
            initial_guess = 0.5
            new_len = sp.fsolve(zero_fffunc, [initial_guess], args=(all_rec_freq_data[w]))[0]
            comp_lens.append(new_len)
                
        comp_lens = np.array(comp_lens) #all computed lengths
        all_comp = np.nanmean(comp_lens) #mean length for a given run
        err_comp = np.nanstd(comp_lens)/np.sqrt(np.shape(comp_lens)[0]) #error on the mean length
        
    else:
        #print("Inconclusive set")
        all_comp = np.nan  
        err_comp = np.nan
        
    return all_comp, err_comp



def get_unix_time_range(date_str1, date_str2):
    def parse_date(date_str):
        if len(date_str) == 8:
            dt = datetime.datetime.strptime(date_str, "%Y%m%d")
            end = dt + datetime.timedelta(days=1) - datetime.timedelta(seconds=1)
        elif len(date_str) == 10:
            dt = datetime.datetime.strptime(date_str, "%Y%m%d%H")
            end = dt + datetime.timedelta(hours=1) - datetime.timedelta(seconds=1)
        else:
            raise ValueError("Invalid date format. Must be YYYYMMDD (8 digits) or YYYYMMDDHH (10 digits).")
        dt_utc = dt.replace(tzinfo=datetime.timezone.utc)
        end_utc = end.replace(tzinfo=datetime.timezone.utc)
        return dt_utc, end_utc
    
    start1, end1 = parse_date(date_str1)
    start2, end2 = parse_date(date_str2)
    
    overall_start = min(start1, start2)
    overall_end = max(end1, end2)
    
    return (int(overall_start.timestamp()), int(overall_end.timestamp()))

def peek_motion(start_date_str, end_date_str, box_number=None):
    # Create plots directory
    plot_dir = os.path.join(script_directory, "motion_plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    # Get Unix timestamp range
    start_ts, end_ts = get_unix_time_range(start_date_str, end_date_str)
    
    # Determine boxes to process
    boxes = [box_number] if box_number is not None else box_n
    
    for box in boxes:
        motion_file = f'motion_{box}.bin'
        if not os.path.exists(motion_file):
            print(f"Motion file for box {box} not found")
            continue
        
        try:
            # Load and filter motion data
            motion_data = np.fromfile(motion_file, dtype=data_dtype)
            time_mask = (motion_data['epoch_time'] >= start_ts) & (motion_data['epoch_time'] <= end_ts)
            filtered_data = motion_data[time_mask]
            
            if len(filtered_data) == 0:
                print(f"No data for box {box} in range")
                continue
            
            # Create plot
            plt.figure(figsize=(12, 6))
            for i in range(min(3, len(filtered_data))):
                plt.plot(filtered_data['vReal'][i], 
                        label=f"Sample {i+1} @ {filtered_data['epoch_time'][i]}")
            
            plt.title(f"Motion Data - Box {box}\n{start_date_str} to {end_date_str}")
            plt.xlabel("Sample Index")
            plt.ylabel("Acceleration")
            plt.legend()
            
            # Save plot
            plot_path = os.path.join(plot_dir, f"box_{box}_motion.png")
            plt.savefig(plot_path)
            plt.close()
            print(f"Saved plot: {plot_path}")
            
        except Exception as e:
            print(f"Error processing box {box}: {str(e)}")

def peek_data(parameter, start_date_str, end_date_str, box_number=None):
    # Parameter configuration with value and error columns
    param_config = {
        "temp": {
            "val_col": 3, 
            "err_col": -1,
            "label": "Temperature",
            "unit": "°C"
        },
        "peak1": {
            "val_col": 4,
            "err_col": -1,
            "label": "Frequency Peak 1",
            "unit": "Hz"
        },
        "peak2": {
            "val_col": 5,
            "err_col": -1,
            "label": "Frequency Peak 2",
            "unit": "Hz"
        },
        "peaks": {
            "cols": [
                {"val_col": 4, "err_col": -1},
                {"val_col": 5, "err_col": -1}
            ],
            "labels": ["Peak 1", "Peak 2"],
            "unit": "Hz",
            "label": "Frequency Peaks",  # Add a label for the "peaks" case
            "ylabel": "Frequency (Hz)"  # Add a y-axis label for the "peaks" case
        },
        "depth": {
            "val_col": 6,
            "err_col": -1,  # No error column (set to -1 or a valid column index)
            "label": "Depth Sensor",
            "unit": "cm"
        }
    }

    # Validate parameter
    if parameter not in param_config:
        raise ValueError("Invalid parameter. Use: 'temp', 'peak1', 'peak2', 'peaks', or 'depth'")

    # Create plots directory
    plot_dir = os.path.join(script_directory, "raw_data_plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    # Get Unix timestamp range
    start_ts, end_ts = get_unix_time_range(start_date_str, end_date_str)
    
    # Determine boxes to process
    boxes = [box_number] if box_number is not None else box_n
    
    for box in boxes:
        raw_file = f'data_{box}.csv'
        if not os.path.exists(raw_file):
            print(f"Data file for box {box} not found")
            continue
            
        try:
            # Load and filter data
            raw_data = np.genfromtxt(raw_file, delimiter=',', skip_header=1)
            time_mask = (raw_data[:, 2] >= start_ts) & (raw_data[:, 2] <= end_ts)
            filtered_data = raw_data[time_mask]
            
            if len(filtered_data) == 0:
                print(f"No data for box {box} in range")
                continue
                
            # Convert timestamps to datetime
            timestamps = [datetime.datetime.utcfromtimestamp(ts) for ts in filtered_data[:, 2]]
            
            plt.figure(figsize=(12, 6))
            
            if parameter == "peaks":
                # Plot both peaks with error bars
                for i, peak_config in enumerate(param_config[parameter]["cols"]):
                    vals = filtered_data[:, peak_config["val_col"]]
                    errs = filtered_data[:, peak_config["err_col"]] if peak_config["err_col"] != -1 else None
                    label = f"{param_config[parameter]['labels'][i]} ({param_config[parameter]['unit']})"
                    plt.errorbar(timestamps, vals, yerr=errs, fmt='-o', capsize=5, label=label)
            else:
                # Plot single parameter with error bars
                config = param_config[parameter]
                vals = filtered_data[:, config["val_col"]]
                errs = filtered_data[:, config["err_col"]] if config["err_col"] != -1 else None
                label = f"{config['label']} ({config['unit']})"
                plt.errorbar(timestamps, vals, yerr=errs, fmt='-o', capsize=5, label=label)
            
            # Set plot title and labels
            if parameter == "peaks":
                plt.title(f"Raw {param_config[parameter]['label']} - Box {box}\n{start_date_str} to {end_date_str}")
                plt.xlabel("UTC Time")
                plt.ylabel(param_config[parameter]["ylabel"])
            else:
                plt.title(f"Raw {config['label']} Data with Errors - Box {box}\n{start_date_str} to {end_date_str}")
                plt.xlabel("UTC Time")
                plt.ylabel(f"{config['label']} ({config['unit']})")
            
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(plot_dir, f"box_{box}_raw_{parameter}_with_errors.png")
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()
            print(f"Saved: {plot_path}")
            
        except Exception as e:
            print(f"Error processing box {box}: {str(e)}")
            
def peek_pdata(parameter, start_date_str, end_date_str, box_number=None):
    """
    Plots processed data (from processed_data_*.csv files) for a given parameter.

    Parameters:
        parameter (str): The parameter to plot. Options: 'temp', 'motion', 'data', 'depth', or 'all'.
        start_date_str (str): Start date in YYYYMMDD or YYYYMMDDHH format.
        end_date_str (str): End date in YYYYMMDD or YYYYMMDDHH format.
        box_number (int, optional): Specific box number to plot. If None, plots all boxes.
    """
    # Parameter configuration with value and error columns
    param_config = {
        "temp": {
            "val_col": 1,  # Temperature value column
            "err_col": -1,  # No error column for temperature
            "label": "Temperature",
            "unit": "°C"
        },
        "motion": {
            "val_col": 2,  # Motion length value column
            "err_col": 3,  # Motion length error column
            "label": "Motion Length",
            "unit": "m"
        },
        "data": {
            "val_col": 4,  # Data length value column
            "err_col": 5,  # Data length error column
            "label": "Data Length",
            "unit": "m"
        },
        "depth": {
            "val_col": 6,  # Depth sensor value column
            "err_col": 7,  # Depth sensor error column
            "label": "Depth Sensor",
            "unit": "m"
        }
    }

    # Validate parameter
    valid_parameters = ["temp", "motion", "data", "depth", "all"]
    if parameter not in valid_parameters:
        raise ValueError(f"Invalid parameter. Use: {', '.join(valid_parameters)}")

    # Create plots directory
    plot_dir = os.path.join(script_directory, "processed_data_plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    # Get Unix timestamp range
    start_ts, end_ts = get_unix_time_range(start_date_str, end_date_str)
    
    # Determine boxes to process
    boxes = [box_number] if box_number is not None else box_n
    
    for box in boxes:
        processed_file = f'processed_data_{box}.csv'
        if not os.path.exists(processed_file):
            print(f"Processed data file for box {box} not found")
            continue
            
        try:
            # Load and filter data
            processed_data = np.genfromtxt(processed_file, delimiter=',', skip_header=1)
            time_mask = (processed_data[:, 0] >= start_ts) & (processed_data[:, 0] <= end_ts)
            filtered_data = processed_data[time_mask]
            
            if len(filtered_data) == 0:
                print(f"No data for box {box} in range")
                continue
                
            # Convert timestamps to datetime
            timestamps = [datetime.datetime.utcfromtimestamp(ts) for ts in filtered_data[:, 0]]
            
            plt.figure(figsize=(12, 6))
            
            if parameter == "all":
                # Plot data, motion, and depth on the same plot
                for param in ["data", "motion", "depth"]:
                    config = param_config[param]
                    vals = filtered_data[:, config["val_col"]]
                    errs = filtered_data[:, config["err_col"]] if config["err_col"] != -1 else None
                    label = f"{config['label']} ({config['unit']})"
                    plt.errorbar(timestamps, vals, yerr=errs, fmt='-o', capsize=5, label=label)
                
                # Set plot title and labels
                plt.title(f"Processed Data, Motion, and Depth - Box {box}\n{start_date_str} to {end_date_str}")
                plt.xlabel("UTC Time")
                plt.ylabel("Value (m)")
            else:
                # Plot single parameter with error bars
                config = param_config[parameter]
                vals = filtered_data[:, config["val_col"]]
                errs = filtered_data[:, config["err_col"]] if config["err_col"] != -1 else None
                label = f"{config['label']} ({config['unit']})"
                plt.errorbar(timestamps, vals, yerr=errs, fmt='-o', capsize=5, label=label)
                
                # Set plot title and labels
                plt.title(f"Processed {config['label']} - Box {box}\n{start_date_str} to {end_date_str}")
                plt.xlabel("UTC Time")
                plt.ylabel(f"{config['label']} ({config['unit']})")
            
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save plot
            plot_name = f"box_{box}_processed_{parameter}_with_errors.png" if parameter != "all" else f"box_{box}_processed_all.png"
            plot_path = os.path.join(plot_dir, plot_name)
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()
            print(f"Saved: {plot_path}")
            
        except Exception as e:
            print(f"Error processing box {box}: {str(e)}")
                   
def unix_to_yyyymmddhh(unix_time):
    """
    Convert Unix timestamp to YYYYMMDDHH format as an integer.
    """
    dt = datetime.datetime.utcfromtimestamp(unix_time)
    return int(dt.strftime("%Y%m%d%H"))

### main code ######################################################################

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
    
    data_tit = "data_" + names[u] + ".csv"
    motion_tit = "motion_" + names[u] + ".bin"

    all_data = np.genfromtxt(data_tit, delimiter = ",")
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

    final_file_title = 'processed_data_'+ str(box_n[u]) +'.csv'
    header="Time,Temperature,Motion Length,Error,Data Length,Error,Depth Sensor Length,Error"
    np.savetxt(final_file_title, final_data, delimiter=",", header=header, comments="")
    print(final_file_title, "saved, shape", np.shape(final_data))
    print("end")























