import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sp
import os
import datetime
from parameters import *

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

def unix_to_yyyymmddhh(unix_time):
    """
    Convert Unix timestamp to YYYYMMDDHH format as an integer.
    """
    dt = datetime.datetime.utcfromtimestamp(unix_time)
    return int(dt.strftime("%Y%m%d%H"))

def second_moment(ro, ri):
    return np.pi / 4 * (ro**4 - ri**4)

def freq_estimator(L, E=69e9, r_o=1.277/100, r_i=1.021/100, p=2700, m=0.638):
    m_pole = np.pi*(r_o**2-r_i**2)*L*p
    return (1/(2*np.pi))*np.sqrt(3*E*second_moment(r_o, r_i)/(L**3*(m+0.24*m_pole)))

def make_boxes():
    script_directory = DIR

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
    print(box_n)
    return box_n

"""
def load_parameters(filename='parameters.txt'):
    # Ensure the file exists
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Parameters file not found: {filename}")

    second_peak = None
    dir_path = None
    processing=None

    with open(filename, 'r') as f:
        for line in f:
            # Strip whitespace and ignore empty or comment lines
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Parse key and value
            if '=' not in line:
                continue
            key, value = map(str.strip, line.split('=', 1))

            if key == 'second_peak':
                # Convert to boolean
                lower_val = value.lower()
                if lower_val in ('true', '1', 'yes'):
                    second_peak = True
                elif lower_val in ('false', '0', 'no'):
                    second_peak = False
                else:
                    raise ValueError(f"Invalid boolean value for second_peak: {value}")

            elif key == 'dir':
                # Remove surrounding quotes if present
                if (value.startswith('"') and value.endswith('"')) or \
                   (value.startswith("'") and value.endswith("'")):
                    directory = value[1:-1]
                else:
                    directory = value
               
            elif key == 'action':
                if value=="process_main":
                    processing=True
                else:
                    processing=False

    # Validate we got both parameters
    if second_peak is None:
        raise KeyError("Parameter 'second_peak' not found in file")
    if directory is None:
        raise KeyError("Parameter 'dir' not found in file")
    if processing is None:
        raise KeyError("Parameter 'action' not found in file")

    return second_peak, directory, processing#"""

def unify_data(main_dir):
    current_dir = os.getcwd()
    
    unified_dir = os.path.join(current_dir, UNIFIED_FOLDER)
    os.makedirs(unified_dir, exist_ok=True)
    
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
        output_path = os.path.join(unified_dir, f'data_{n}.csv')
        
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
        output_path = os.path.join(unified_dir, f'motion_{n}.bin')
        
        dtype = np.dtype([('epoch_time', np.int32), ('vReal', np.float32, (2048,))])
        combined = np.empty(0, dtype=dtype)
        
        for filename in bin_files:
            file_path = os.path.join(motion_dir, filename)
            data = np.fromfile(file_path, dtype=dtype)
            combined = np.concatenate((combined, data))
        print("Compiled files")
        combined.tofile(output_path)

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


















