import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sp
import os
import datetime
from parameters import *
from freq_estimator import *

# F2 v L lookup table columns
length_column = 0
analytical_freq_column = 1
FEM_freq1_column = 2
FEM_freq2_column = 3

filename = 'freq_vs_length.csv'

if not os.path.exists(filename):
    print(f"{filename} not found. Creating it...")
    make_f_file()

# Load lookup table once using numpy
_data = np.loadtxt('freq_vs_length.csv', delimiter=',', skiprows=1)
_L_table = _data[:, length_column]       # column 0: L
_F1_table = _data[:, FEM_freq1_column]   # column 2: F1_FEM
_F2_table = _data[:, FEM_freq2_column]   # column 3: F2_FEM

def unix_to_yyyymmddhh(unix_time):
    """
    Convert Unix timestamp to YYYYMMDDHH format as an integer.
    """
    dt = datetime.datetime.utcfromtimestamp(unix_time)
    return int(dt.strftime("%Y%m%d%H"))

def unix_to_yyyymmddhhmm(unix_time):
    """
    Convert Unix timestamp to YYYYMMDDHHMM format as an integer.
    """
    dt = datetime.datetime.utcfromtimestamp(unix_time)
    return int(dt.strftime("%Y%m%d%H%M"))

def get_unix_time_range(start_date_str, end_date_str):
    """
    Convert date strings to Unix timestamp range.
    Supports YYYYMMDD, YYYYMMDDHH, and YYYYMMDDHHMM formats.
    Times are interpreted as UTC.
    """
    def parse_date_str(date_str):
        if len(date_str) == 8:  # YYYYMMDD
            dt = datetime.datetime.strptime(date_str, "%Y%m%d")
        elif len(date_str) == 10:  # YYYYMMDDHH
            dt = datetime.datetime.strptime(date_str, "%Y%m%d%H")
        elif len(date_str) == 12:  # YYYYMMDDHHMM
            dt = datetime.datetime.strptime(date_str, "%Y%m%d%H%M")
        else:
            raise ValueError(f"Date string must be YYYYMMDD, YYYYMMDDHH, or YYYYMMDDHHMM format, got: {date_str}")
        
        # Explicitly set timezone to UTC
        return dt.replace(tzinfo=datetime.timezone.utc)
    
    start_dt = parse_date_str(start_date_str)
    end_dt = parse_date_str(end_date_str)
    
    start_ts = int(start_dt.timestamp())
    end_ts = int(end_dt.timestamp())
    
    return start_ts, end_ts

def second_moment(ro, ri):
    return np.pi / 4 * (ro**4 - ri**4)

def freq_estimator(L, E=YOUNG, r_o=R_O, r_i=R_I, p=DENSITY, m=MASS):
    m_pole = np.pi*(r_o**2-r_i**2)*L*p
    return (1/(2*np.pi))*np.sqrt(3*E*second_moment(r_o, r_i)/(L**3*(m+0.24*m_pole)))

def make_boxes():
    script_directory = DIR

    # Scan for directories starting with 'box_'
    folders = [f for f in os.listdir(script_directory) 
               if os.path.isdir(os.path.join(script_directory, f))]
    box_n = []

    for folder in folders:
        if folder.startswith('box_'):
            try:
                # Extract the number from 'box_<n>'
                n = int(folder.split('_')[1])
                box_n.append(n)
            except (IndexError, ValueError):
                continue  # Skip malformed folder names

    box_n = list(set(box_n))
    box_n.sort()
    return box_n

def unify_data(main_dir):
    current_dir = os.getcwd()
    unified_dir = os.path.join(current_dir, UNIFIED_FOLDER)
    os.makedirs(unified_dir, exist_ok=True)
    
    # Process each box folder
    for box_folder in os.listdir(main_dir):
        if not box_folder.startswith('box_'):
            continue
            
        try:
            n = int(box_folder.split('_')[1])
        except (IndexError, ValueError):
            continue
            
        box_path = os.path.join(main_dir, box_folder)
        #print(f"\nProcessing box {n} at {box_path}")
        
        # DATA FILES PROCESSING
        data_files_dir = os.path.join(box_path, f"data_files_{n}")
        if os.path.exists(data_files_dir) and os.path.isdir(data_files_dir):
            csv_files = [
                f for f in os.listdir(data_files_dir)
                if f.startswith(f'data_{n}_') and f.endswith('.csv')
            ]
            
            if csv_files:
                # Sort by date (YYYYMMDD)
                csv_files.sort(key=lambda x: x.split('_')[2].split('.')[0])
                output_path = os.path.join(unified_dir, f'data_{n}.csv')
                
                #print(f"Found {len(csv_files)} CSV files. Merging into {output_path}")
                with open(output_path, 'w') as outfile:
                    first_file = True
                    for filename in csv_files:
                        file_path = os.path.join(data_files_dir, filename)
                        with open(file_path) as infile:
                            header = infile.readline()
                            if first_file:
                                outfile.write(header)
                                first_file = False
                            else:
                                infile.readline()  # Skip header
                            outfile.writelines(infile.readlines())
            else:
                print(f"No CSV files found in data_files_{n}")
        else:
            print(f"data_files_{n} directory not found for box {n}")
        
        # MOTION FILES PROCESSING
        motion_files_dir = os.path.join(box_path, f"motion_files_{n}")
        if os.path.exists(motion_files_dir) and os.path.isdir(motion_files_dir):
            bin_files = [
                f for f in os.listdir(motion_files_dir)
                if f.startswith(f'motion_{n}_') and f.endswith('.bin')
            ]
            
            if bin_files:
                # Sort by date (YYYYMMDD)
                bin_files.sort(key=lambda x: x.split('_')[2].split('.')[0])
                output_path = os.path.join(unified_dir, f'motion_{n}.bin')
                
                #print(f"Found {len(bin_files)} BIN files. Merging into {output_path}")
                dtype = np.dtype([('epoch_time', np.int32), ('vReal', np.float32, (ARRAY_SIZE,))])
                combined = np.empty(0, dtype=dtype)
                
                for filename in bin_files:
                    file_path = os.path.join(motion_files_dir, filename)
                    data = np.fromfile(file_path, dtype=dtype)
                    combined = np.concatenate((combined, data))
                
                combined.tofile(output_path)
            else:
                print(f"No BIN files found in motion_files_{n}")
        else:
            print(f"motion_files_{n} directory not found for box {n}")


def identify_peaks(amps, freqs, noise_threshold=0, tol=TOL, min_freq=MIN_FREQ):
    """
    Identifies fundamental frequencies and their harmonics/sidebands in a spectrum.
    Uses local noise calculation with 1Hz neighborhood for each peak.
    
    Parameters:
        amps (np.array): Array of FFT amplitudes
        freqs (np.array): Array of corresponding frequencies
        noise_threshold (float): Minimum amplitude for peak consideration
        tol (float): Frequency tolerance for harmonic matching (fractional)
        min_freq (float): Minimum frequency to consider for peaks (Hz)
    
    Returns:
        dict: Dictionary with detected peaks categorized by type
    """
    # Find all peaks above minimum frequency
    peaks = []
    for i in range(1, len(amps)-1):
        # Skip frequencies below minimum
        if freqs[i] < min_freq or amps[i] < noise_threshold:
            continue
            
        # Check if local maximum
        if amps[i] > amps[i-1] and amps[i] > amps[i+1]:
            # Calculate local noise in 1Hz neighborhood
            if freqs[i] < 6*MIN_FREQ:
                window_rad = MIN_FREQ
            else:
                window_rad = 3*MIN_FREQ
                
            window_low = max(0, freqs[i] - window_rad)
            window_high = freqs[i] + window_rad
            mask = (freqs >= window_low) & (freqs <= window_high)
            
            adjacent_freq_rad = 0.1
            # Exclude the peak itself and adjacent bins
            peak_mask = (freqs >= freqs[i] - adjacent_freq_rad) & (freqs <= freqs[i] + adjacent_freq_rad)
            noise_mask = mask & ~peak_mask
            
            if np.any(noise_mask):
                # Calculate local noise as median of surrounding amplitudes
                local_noise = np.mean(amps[noise_mask])# - np.min(amps[noise_mask])
                local_noise_std = np.std(amps[noise_mask])
                # Only keep peak if significantly above local noise
                if amps[i] > local_noise + LOCAL_NOISE_THRESH*local_noise_std: #and amps[i] > noise_threshold:
                    peaks.append((freqs[i], amps[i]))
    
    # Initialize result structure
    result = {
        'fundamental1': {'freq': None, 'harmonics': []},
        'fundamental2': {'freq': None, 'harmonics': [], 'modulated': []},
        'other_peaks': []
    }
    
    if not peaks:
        return result
    
    # Sort peaks by frequency (lowest first)
    peaks.sort(key=lambda x: x[0])
    
    # Identify fundamental1 as the first (lowest frequency) peak
    fundamental1 = None
    # Sort peaks by frequency (lowest first)
    sorted_peaks = sorted(peaks, key=lambda x: x[0])
    
    # Iterate through all peaks to find valid F1 candidate
    for i, (candidate_freq, candidate_amp) in enumerate(sorted_peaks):
        # Calculate harmonic band upper limit
        upper_bound = 2 * candidate_freq * (1 + L_TOL)
        
        # Find all peaks in the harmonic band
        band_peaks = [
            (f, a) for f, a in sorted_peaks 
            if min_freq <= f <= upper_bound
        ]
        
        # Skip if no peaks found in band
        if not band_peaks:
            continue
            
        # Find max amplitude peak in band
        max_peak = max(band_peaks, key=lambda x: x[1])
        
        # Check if current candidate is the max in its band
        if abs(max_peak[0] - candidate_freq) < tol:
            fundamental1 = candidate_freq
            break  # Valid F1 found
    
    # Fallback to lowest peak if no valid F1 found
    if fundamental1 is None and sorted_peaks:
        fundamental1 = sorted_peaks[0][0]
    result['fundamental1']['freq'] = fundamental1
    
    # Try to find fundamental2 with triplet pattern
    fundamental2 = None
    triplet_found = False
    
    # Look for potential fundamental2 candidates
    candidates = []

    # half‑width of the window:
    delta_f = fundamental1 * (1 + L_TOL)

    for freq, amp in peaks[1:]:
        # 1) skip harmonics of f1
        ratios = [freq/fundamental1, fundamental1/freq]
        if any(abs(r - n) < tol for r in ratios for n in [2,3,4]):
            continue

        # 2) build absolute-frequency mask over the **full** spectrum
        low  = freq - delta_f
        high = freq + delta_f
        mask = (freqs >= low) & (freqs <= high)

        # 3) find the maximum amplitude *in that entire span* of the spectrum
        if not mask.any():
            continue
        max_in_window = amps[mask].max()

        # 4) only keep if your candidate is that local maximum
        if amp >= max_in_window:
            candidates.append((freq, amp))

    
    # Check for triplet pattern around each candidate
    for candidate, amp in candidates:
        lower_side = None
        upper_side = None
        
        # Look for sidebands at candidate ± fundamental1
        for f, a in peaks:
            if abs(f - (candidate - fundamental1)) < tol * fundamental1:
                lower_side = f
            if abs(f - (candidate + fundamental1)) < tol * fundamental1:
                upper_side = f
        
        # If we have two sidebands, it's a valid triplet
        if lower_side and upper_side:
            fundamental2 = candidate
            triplet_found = True
            # Store the sidebands we found
            if lower_side:
                result['fundamental2']['modulated'].append(lower_side)
            if upper_side:
                result['fundamental2']['modulated'].append(upper_side)
            break
    
    # If no triplet found, use the strongest candidate as fundamental2
    if not triplet_found and candidates:
        # Find candidate with highest amplitude
        candidates.sort(key=lambda x: x[1], reverse=True)
        fundamental2 = candidates[0][0]
    
    result['fundamental2']['freq'] = fundamental2
    
    # Identify harmonics and other peaks
    for freq, amp in peaks:
        # Skip fundamentals
        if freq == fundamental1 or freq == fundamental2:
            continue
        
        # Check if harmonic of fundamental1
        for n in range(2, 3):
            if fundamental1 and abs(freq - n*fundamental1) < tol*n*fundamental1:
                result['fundamental1']['harmonics'].append(freq)
                break
        else:
            # Check if harmonic of fundamental2
            for n in range(2, 3):
                if fundamental2 and abs(freq - n*fundamental2) < tol*n*fundamental2:
                    result['fundamental2']['harmonics'].append(freq)
                    break
            else:
                # Check if modulated peak (f2 ± n*f1)
                if fundamental2:
                    for n in range(1, 2):
                        if (abs(freq - (fundamental2 + n*fundamental1)) < tol*(fundamental2 + n*fundamental1) or
                            abs(freq - (fundamental2 - n*fundamental1)) < tol*(fundamental2 - n*fundamental1)):
                            result['fundamental2']['modulated'].append(freq)
                            break
                    else:
                        result['other_peaks'].append(freq)
                else:
                    result['other_peaks'].append(freq)
    
    return result

def check_multiple(main, others=[], tol=TOL):
    """
    Identify the dominant fundamental frequency (and close relatives) from two lists of peaks.

    This routine takes a list of “main” candidate frequencies and (optionally) a list of
    “other” frequencies.  It:
      1. Compares each pair in `main` to spot integer‑multiple (harmonic) relationships.
      2. Normalizes such harmonics back down to the same base frequency and tallies votes.
      3. Checks each entry in `others` for the same harmonic relationship and adds them in.
      4. Recounts mutual agreement among all candidates.
      5. Selects the highest‑voted base frequency, then returns every candidate within
         ±3x`tol` relative tolerance of that best fundamental.

    Parameters
    main : array_like of float
        A one‑dimensional array of primary frequency peaks (likely fundamentals).
    others : array_like of float, optional
        A second array of additional peaks to check for harmonics of any candidate
        in `main`.  Defaults to an empty list (i.e.\ no extras).
    tol : float, optional
        Relative tolerance for detecting “harmonic = ±n×” agreement.
        For example, `tol=0.05` means you accept ratios in `[n·(1−0.05), n·(1+0.05)]`.
        Default is `0.05`.

    Returns
    ndarray, shape (m, 1)
        Array of filtered candidates around the best fundamental:
        - Column 0: frequency value (either a main entry or a normalized harmonic).
        Only rows within ±3x`tol` of the top‑voted fundamental are kept.
    
    Example use
    >>> check_multiple([13,13,59,29,17,2.01,4.02,1.98,4.05,1.99,2.05,3.98,2.1,2.02,2.01],
        others=[13,59,26,17,2.01,4.02,1.0,6.0])
    array([2.01 , 2.01 , 1.98 , 2.025, 1.99 , 2.05 , 1.99 , 2.1  , 2.02 ,
           2.01 , 2.01 , 2.01 , 2.   ])
    """
    f1_len =  len(main)
    f1_arr = np.zeros((f1_len,2))
    for i in range(f1_len):
        for j in range(i+1, f1_len):
            if main[j] == 0:
                continue
            for n in [1,2,3]:
                ratio = main[i]/main[j]
                if abs(ratio/n - 1) < tol:
                    f1_arr[i,0] = main[i] / n
                    f1_arr[i,1] += 1
                    f1_arr[j,1] += 1
                elif abs(ratio*n - 1) < tol:
                    f1_arr[j,0] = main[j] / n
                    f1_arr[i,1] += 1
                    f1_arr[j,1] += 1
        if f1_arr[i,0] == 0:
            f1_arr[i,0] = main[i]
    for i in range(len(others)):
        used = False
        for j in range(f1_len):
            for n in [1,2,3]:
                ratio = others[i]/f1_arr[j,0]
                if abs(ratio/n - 1) < tol and not used:
                    f1_arr = np.vstack([f1_arr,np.array([others[i] / n, 1])])
                    used = True
    for i in range(np.shape(f1_arr)[0] - 1, f1_len - 1, -1):
        for j in range(i):
            ratio = f1_arr[i,0]/f1_arr[j,0]
            if abs(ratio - 1) < tol:
                f1_arr[i,1] += 1
                f1_arr[j,1] += 1
    if f1_arr.size > 0:
        idx = np.argmax(f1_arr[:, 1])
        f1 = f1_arr[idx, 0]
        ratio = f1_arr[:,0]/f1
        """diff = ratio-1
        selected_freq = f1_arr[np.abs(diff) < L_TOL]
        selected_freq = selected_freq[:,0]
        return selected_freq"""
        ratio = np.round(ratio)
        return f1_arr[:,0]/ratio
    else:
        return np.array([])

def frequency_analysis(freq1, freq1_harmonic=[], freq2=[], freq2_harmonic=[], freq2_modulated=[], others=[], tol=0.05):
    """
    Identify and filter dominant base frequencies from two frequency bands using harmonic and modulation relationships.

    Parameters
    freq1, freq1_harmonic, freq2, freq2_harmonic, freq2_modulated : list of float
        Lists of frequency peaks including main, harmonic, and modulated components for two frequency bands.
    others : list of float
        Additional frequencies to be considered as potential harmonics.
    tol : float, optional
        Relative tolerance for detecting harmonic agreement. Default is 0.05.

    Returns
    filtered_freq1 : ndarray
        Filtered frequencies supporting the dominant base frequency for `freq1`.
    filtered_freq2 : ndarray
        Filtered frequencies supporting the dominant base frequency for `freq2`.
    """
    all_freq1 = freq1 + freq1_harmonic
    all_freq2 = freq2 + freq2_harmonic
    
    filtered_freq1 = check_multiple(all_freq1, tol=tol)
    filtered_freq2 = check_multiple(all_freq2, others=others+all_freq1, tol=tol)
    
    if filtered_freq1.size > 0 and filtered_freq2.size > 0:
        f1 = np.median(filtered_freq1)
        f2 = np.median(filtered_freq2)
        
        modulated = np.array(freq2_modulated)
        modulated = np.abs(modulated - f2)
        
        for i in range(np.shape(modulated)[0]):
            if (modulated[i] / f1) - 1 < L_TOL:
                filtered_freq1 = np.append(filtered_freq1, modulated)
    
    return filtered_freq1, filtered_freq2

def freq_to_len(f1, f2, r_outer=R_O, r_inner=R_I, young=YOUNG, rho=DENSITY, m_tip=MASS, initial_guess=POLE_MAX_L):
    """
    Convert arrays of first‑mode (f1) and second‑mode (f2) frequencies into pole lengths.

    - f1 uses analytic inversion.
    - f2 looks up the closest FEM result in 'freq_vs_length.csv'.
    """
    # --- analytic for f2 ---
    I = second_moment(r_outer, r_inner)
    A = np.pi * (r_outer**2 - r_inner**2)
    
    freq_guess1 = np.median(f1)
    initial_guess1 = _L_table[np.argmin(np.abs(_F1_table - freq_guess1))]
    len1 = np.array([sp.fsolve(lambda L: exact_frequencies(L,young,r_outer,r_inner,rho,m_tip)[0] - freq, x0=initial_guess1)[0] for freq in f1])
    

    # --- table lookup for f2 ---
    # For each target freq, find closest entry in the CSV data
    freq_guess2 = np.median(f2)
    #print(freq_guess2)
    
    #initial_guess2 = _L_table[np.argmin(np.abs(_F2_table - freq_guess2))]
    valid_mask = ~np.isnan(_F2_table)
    valid_indices = np.where(valid_mask)[0]
    valid_differences = np.abs(_F2_table[valid_indices] - freq_guess2)
    best_valid_idx = valid_indices[np.argmin(valid_differences)]
    initial_guess2 = _L_table[best_valid_idx]
    #print(initial_guess2)
    
    len2 = np.array([sp.fsolve(lambda L: exact_frequencies(L,young,r_outer,r_inner,rho,m_tip)[1] - freq, x0=initial_guess2)[0] for freq in f2])
    #exact_frequencies returns (f1, f2)
    #print(len2)
    len1 = len1[len1<POLE_MAX_L-0.01]
    len1 = len1[POLE_MIN_L+0.01<len1]
    len2 = len2[len2<POLE_MAX_L-0.01]
    len2 = len2[POLE_MIN_L+0.01<len2]

    return len1, len2

def vote_select(main, tol=TOL):
    f1_len =  len(main)
    f1_arr = np.zeros((f1_len,2))
    for i in range(f1_len):
        for j in range(i+1, f1_len):
            if main[j] == 0:
                continue
            ratio = main[i]/main[j]
            if abs(ratio - 1) < tol:
                f1_arr[i,0] = main[i]
                f1_arr[i,1] += 1
                f1_arr[j,1] += 1
        if f1_arr[i,0] == 0:
            f1_arr[i,0] = main[i]
    if f1_arr.size > 0:
        idx = np.argmax(f1_arr[:, 1])
        f1 = f1_arr[idx, 0]
        #print(f1)
        #print(f1_arr[idx, 1])
        if f1_arr[idx, 1] < 2:
            #print("Not enough votes: skipping")
            return []
        ratio = f1_arr[:,0]/f1
        diff = ratio-1
        f1_arr = f1_arr[np.abs(diff) < L_TOL]
        return f1_arr[:,0]
    else:
        return np.array([])






