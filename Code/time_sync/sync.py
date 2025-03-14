import csv
import glob
import numpy as np
import os

corrections = []

for filepath in glob.glob("time*.csv"):
    filename = os.path.basename(filepath)
    if not filename.startswith("time") or not filename.endswith(".csv"):
        continue
    box_str = filename[4:-4]
    try:
        box = int(box_str)
    except ValueError:
        continue
    
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        iterations = []
        clock = []
        real = []
        for row in reader:
            iterations.append(int(row[0]))
            clock.append(int(row[1]))
            real.append(int(row[2]))
        
        #detect reboot (iteration resets to 0)
        cut_index = len(iterations)
        for i in range(1, len(iterations)):
            if iterations[i] == 0:
                cut_index = i
                break
        
        #truncate data to ignore post-reboot entries
        clock_trunc = clock[:cut_index]
        real_trunc = real[:cut_index]
        
        #calculate multiplicative correction factor using the mean
        if len(clock_trunc) < 2:
            correction = 1.0
        else:
            clock_np = np.array(clock_trunc, dtype=np.float64)
            real_np = np.array(real_trunc, dtype=np.float64)
            clock_diffs = np.diff(clock_np)
            real_diffs = np.diff(real_np)
            
            #filter out intervals where real time didn't change
            valid_mask = real_diffs != 0
            valid_clock_diffs = clock_diffs[valid_mask]
            valid_real_diffs = real_diffs[valid_mask]
            
            if len(valid_real_diffs) == 0:
                correction = 1.0
            else:
                ratios = valid_clock_diffs / valid_real_diffs
                correction = float(np.mean(ratios))
        
        corrections.append((box, correction))

#compute median of all correction factors for default
if corrections:
    all_factors = [cf for _, cf in corrections]
    median_correction = np.median(all_factors)
else:
    median_correction = 1.0 

#generate Arduino correction function
cpp_code = [
    "float correctionFactor(int box_n){"
]
for box, cf in sorted(corrections, key=lambda x: x[0]):
    cpp_code.append(f"    if (box_n == \"{box}\") return {cf:.6f}f;")
cpp_code.append(f"    return {median_correction:.6f}f; // Median of all factors if box is not specified")
cpp_code.append("}")

print("\n".join(cpp_code))































