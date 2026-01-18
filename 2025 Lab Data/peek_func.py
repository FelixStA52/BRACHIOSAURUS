import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sp
import os
import datetime
from processing_helpers import *
from parameters import *
plt.style.use('tableau-colorblind10')

data_processing = PROCESSING
DONT_USE_SECOND_PEAK = SECOND_PEAK
main_dir = DIR

script_directory = os.path.dirname(os.path.abspath(__file__))
box_n=make_boxes()

def peek_data(parameter, start_date_str, end_date_str, box_number=None, x_lim=False, y_lim=False, title_size=12):
    """
    Peek raw data, as saved on the micro SD cards of the boxes

    Parameters:
        parameter (str): The parameter to plot. Options: 'temp', 'peak1' (highest frequency peak),
            'peak2' (second highest frequency peak), 'peaks', or 'depth'.
        start_date_str (str): Start date to plot in YYYYMMDD or YYYYMMDDHH format.
        end_date_str (str): End date to plot in YYYYMMDD or YYYYMMDDHH format.
        box_number (int, optional): Specific box number to plot. If None, plots all boxes.
        x_lim (tuple (str, str), optional): Left and right limit of the x axis for the plot. Use "YYYYMMDD" or "YYYYMMDDHH".
        y_lim (tuple (float, float), optional): Bottom and top limit of the y axis for the plot.
        
    Example use:
        peek_data("temp", "2025050500", "2025052512", box_number=8, x_lim=("20250501","20250601"), y_lim=(-40,20))
        peek_data("peak1", "20250505", "20250525")
        peek_data("depth", "2025050500", "2025052512", y_lim=(0,5))
    """
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
    plot_dir = os.path.join(script_directory, RAW_DATA_PLOTS_FOLDER)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Get Unix timestamp range
    start_ts, end_ts = get_unix_time_range(start_date_str, end_date_str)
    
    # Determine boxes to process
    boxes = [box_number] if box_number is not None else box_n
    
    for box in boxes:
        raw_file = UNIFIED_FOLDER + "/" + f'data_{box}.csv'
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
                    plt.errorbar(timestamps, vals, yerr=errs, fmt='o', capsize=5, label=label)
            else:
                # Plot single parameter with error bars
                config = param_config[parameter]
                vals = filtered_data[:, config["val_col"]]
                errs = filtered_data[:, config["err_col"]] if config["err_col"] != -1 else None
                label = f"{config['label']} ({config['unit']})"
                plt.errorbar(timestamps, vals, yerr=errs, fmt='o', capsize=5, label=label)
                
            print("Mean value:", np.nanmean(vals))
            print("std:", np.nanstd(vals))
            
            # Set plot title and labels
            if parameter == "peaks":
                plt.title(f"Raw {param_config[parameter]['label']} - Box {box}", fontsize=title_size)
                plt.xlabel("UTC Time", size=title_size)
                plt.ylabel(param_config[parameter]["ylabel"], size=title_size)
            else:
                plt.title(f"Raw {config['label']} Data with Errors - Box {box}", fontsize=title_size)
                plt.xlabel("UTC Time", size=title_size)
                plt.ylabel(f"{config['label']} ({config['unit']})", size=title_size)
            
            plt.legend(fontsize=20)
            plt.grid()
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            if x_lim:   
                x_lim_dt = []
                for date_str in x_lim:
                    if len(date_str) == 8:  # YYYYMMDD
                        x_lim_dt.append(datetime.datetime.strptime(date_str, "%Y%m%d"))
                    elif len(date_str) == 10:  # YYYYMMDDHH
                        x_lim_dt.append(datetime.datetime.strptime(date_str, "%Y%m%d%H"))
                    elif len(date_str) == 12:  # YYYYMMDDHHMM
                        x_lim_dt.append(datetime.datetime.strptime(date_str, "%Y%m%d%H%M"))
                    else:
                        raise ValueError("x_lim dates must be YYYYMMDD or YYYYMMDDHH or YYYYMMDDHHMM format")
                plt.xlim(x_lim_dt)
            
            if y_lim:
                plt.ylim(y_lim)
            
            # Save plot
            plot_path = os.path.join(plot_dir, f"box_{box}_raw_{parameter}_with_errors.png")
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()
            print(f"Saved: {plot_path}")
            
        except Exception as e:
            print(f"Error processing box {box}: {str(e)}")
            
def peek_pdata(parameter, start_date_str, end_date_str, box_number=None, x_lim=False, y_lim=False, title_size=12):
    """
    Plots processed data (from processed_data_*.csv files) for a given parameter.

    Parameters:
        parameter (str): The parameter to plot. Options: 'temp', 'motion' (length as computed from motion bin files),
            'data' (length as computed from the data csv files), 'depth', 'f1' (fundamental frequency 1),
            'f2' (fundamental frequency 2), 'both_f' (both f1 and f2), or 'all' (motion + data + depth).
        start_date_str (str): Start date in YYYYMMDD, YYYYMMDDHH, or YYYYMMDDHHMM format.
        end_date_str (str): End date in YYYYMMDD, YYYYMMDDHH, or YYYYMMDDHHMM format.
        box_number (int, optional): Specific box number to plot. If None, plots all boxes.
        x_lim (tuple (str, str), optional): Left and right limit of the x axis for the plot. Use "YYYYMMDD", "YYYYMMDDHH", or "YYYYMMDDHHMM".
        y_lim (tuple (float, float), optional): Bottom and top limit of the y axis for the plot.
        
    Example use:
        peek_pdata("temp", "202505050000", "202505251200", box_number=8, x_lim=("202505010000","202506010000"), y_lim=(-40,20))
        peek_pdata("motion", "20250505", "20250525")
        peek_pdata("data", "20250505", "20250525", y_lim=(0.8,1))
        peek_pdata("f1", "20250505", "20250525", y_lim=(0,50))
        peek_pdata("f2", "20250505", "20250525", y_lim=(0,100))
        peek_pdata("both_f", "20250505", "20250525", y_lim=(0,100))
        peek_pdata("all", "2025050500", "2025052512", y_lim=(0,5))
    """
    # Parameter configuration with value and error columns
    param_config = {
        "temp": {
            "val_col": 1,  # Temperature value column
            "err_col": -1,  # No error column for temperature
            "label": "Temperature",
            "unit": "°C",
            "color": "purple"
        },
        "motion": {
            "val_col": 2,  # Motion length value column
            "err_col": 3,  # Motion length error column
            "label": "Computer Motion Analysis Length",
            "unit": "m",
            "color": "deepskyblue"
        },
        "data": {
            "val_col": 4,  # Data length value column
            "err_col": 5,  # Data length error column
            "label": "Microcontroller Analysis Length",
            "unit": "m",
            "color": "royalblue"
        },
        "depth": {
            "val_col": 6,  # Depth sensor value column
            "err_col": 7,  # Depth sensor error column
            "label": "Depth Sensor Length",
            "unit": "m",
            "color": "blue"
        },
        "f1": {
            "val_col": 8,  # F1 frequency value column
            "err_col": 9,  # F1 frequency error column
            "label": "Fundamental Frequency 1",
            "unit": "Hz",
            "color": "red"
        },
        "f2": {
            "val_col": 10,  # F2 frequency value column
            "err_col": 11,  # F2 frequency error column
            "label": "Fundamental Frequency 2",
            "unit": "Hz",
            "color": "blue"
        }
    }

    # Validate parameter
    valid_parameters = ["temp", "motion", "data", "depth", "f1", "f2", "both_f", "all"]
    if parameter not in valid_parameters:
        raise ValueError(f"Invalid parameter. Use: {', '.join(valid_parameters)}")

    # Create plots directory
    plot_dir = os.path.join(script_directory, PROCESSED_DATA_PLOTS_FOLDER)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Get Unix timestamp range and convert to integer minute representation
    start_ts_unix, end_ts_unix = get_unix_time_range(start_date_str, end_date_str)
    start_int = int(unix_to_yyyymmddhhmm(start_ts_unix))
    end_int = int(unix_to_yyyymmddhhmm(end_ts_unix))
    
    # Determine boxes to process
    boxes = [box_number] if box_number is not None else box_n
    
    for box in boxes:
        processed_file = PROCESSED_FOLDER + "/" + f'processed_data_{box}.csv'
        if not os.path.exists(processed_file):
            print(f"Processed data file for box {box} not found")
            continue
            
        try:
            # Load and filter data
            processed_data = np.genfromtxt(processed_file, delimiter=',', skip_header=1)
            
            # Handle single-row case
            if processed_data.ndim == 1:
                processed_data = processed_data.reshape(1, -1)

            data_timestamps = processed_data[:, 0].astype(np.int64)
            time_mask = (data_timestamps >= start_int) & (data_timestamps <= end_int)
            filtered_data = processed_data[time_mask]
            
            if len(filtered_data) == 0:
                print(f"No data for box {box} in range")
                continue
                
            # Convert timestamps to datetime
            timestamps = [
                datetime.datetime.strptime(str(int(ts)), "%Y%m%d%H%M")
                for ts in filtered_data[:, 0]
            ]
            
            plt.figure(figsize=(12, 6))
            
            if parameter == "all":
                # Plot data, motion, and depth on the same plot
                for param in ["depth", "motion", "data"]:
                    config = param_config[param]
                    vals = filtered_data[:, config["val_col"]]
                    errs = filtered_data[:, config["err_col"]] if config["err_col"] != -1 else None
                    vals[errs > 0.1] = np.nan
                    errs[errs > 0.1] = np.nan
                    label = f"{config['label']} ({config['unit']})"
                    plt.errorbar(timestamps, vals, yerr=errs, fmt='o', color=config['color'], capsize=5, label=label, alpha=0.5)
                
                # Set plot title and labels
                plt.title(f"Lengths as Measured by BRACHIOSAURUS Through Time - Box {box}", fontsize=title_size)
                plt.xlabel("UTC Time", size=title_size)
                plt.ylabel("Length (m)", size=title_size)
                
            elif parameter == "both_f":
                # Plot both f1 and f2 frequencies on the same plot
                for param in ["f1", "f2"]:
                    config = param_config[param]
                    vals = filtered_data[:, config["val_col"]]
                    errs = filtered_data[:, config["err_col"]] if config["err_col"] != -1 else None
                    # Filter out NaN values and zero values for frequencies
                    valid_mask = ~(np.isnan(vals) | (vals == 0))
                    vals_filtered = vals[valid_mask]
                    errs_filtered = errs[valid_mask] if errs is not None else None
                    timestamps_filtered = [timestamps[i] for i in range(len(timestamps)) if valid_mask[i]]
                    
                    label = f"{config['label']} ({config['unit']})"
                    plt.errorbar(timestamps_filtered, vals_filtered, yerr=errs_filtered, fmt='o', 
                               color=config['color'], capsize=5, label=label, alpha=0.7)
                
                # Set plot title and labels
                plt.title(f"Fundamental Frequencies - Box {box}", fontsize=title_size)
                plt.xlabel("UTC Time", size=title_size)
                plt.ylabel("Frequency (Hz)", size=title_size)
                
            else:
                # Plot single parameter with error bars
                config = param_config[parameter]
                vals = filtered_data[:, config["val_col"]]
                errs = filtered_data[:, config["err_col"]] if config["err_col"] != -1 else None
                
                # For frequency parameters, filter out NaN and zero values
                if parameter in ["f1", "f2"]:
                    valid_mask = ~(np.isnan(vals) | (vals == 0))
                    vals = vals[valid_mask]
                    errs = errs[valid_mask] if errs is not None else None
                    timestamps = [timestamps[i] for i in range(len(timestamps)) if valid_mask[i]]
                
                label = f"{config['label']} ({config['unit']})"
                plt.errorbar(timestamps, vals, yerr=errs, fmt='o', color=config['color'], capsize=5, label=label)
                
                # Set plot title and labels
                plt.title(f"Processed {config['label']} - Box {box}", fontsize=title_size)
                plt.xlabel("UTC Time", size=title_size)
                plt.ylabel(f"{config['label']} ({config['unit']})", size=title_size)
                
                print(f"[Box {box}] Mean value:", np.nanmean(vals))
                print(f"[Box {box}] std:", np.nanstd(vals))
            
            if x_lim:   
                x_lim_dt = []
                for date_str in x_lim:
                    if len(date_str) == 8:  # YYYYMMDD
                        x_lim_dt.append(datetime.datetime.strptime(date_str, "%Y%m%d"))
                    elif len(date_str) == 10:  # YYYYMMDDHH
                        x_lim_dt.append(datetime.datetime.strptime(date_str, "%Y%m%d%H"))
                    elif len(date_str) == 12:  # YYYYMMDDHHMM
                        x_lim_dt.append(datetime.datetime.strptime(date_str, "%Y%m%d%H%M"))
                    else:
                        raise ValueError("x_lim dates must be YYYYMMDD, YYYYMMDDHH, or YYYYMMDDHHMM format")
                plt.xlim(x_lim_dt)
            
            if y_lim:
                plt.ylim(y_lim)
            
            plt.legend(fontsize=15)
            plt.grid()
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save plot
            plot_name = f"box_{box}_processed_{parameter}_with_errors.png"
            plot_path = os.path.join(plot_dir, plot_name)
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()
            print(f"[Box {box}] Saved: {plot_path}\n")
            
        except Exception as e:
            print(f"Error processing box {box}: {str(e)}")
    
def peek_motion_sum(start_date_str, end_date_str, box_number=None, x_lim=False, y_lim=False, title_size=12):
    """
    Plots the sum of the acceleration data recorded in the motion bin file. This is a proxi for wind strength.

    Parameters:
        start_date_str (str): Start date in YYYYMMDD or YYYYMMDDHH format.
        end_date_str (str): End date in YYYYMMDD or YYYYMMDDHH format.
        box_number (int, optional): Specific box number to plot. If None, plots all boxes.
        x_lim (tuple (str, str), optional): Left and right limit of the x axis for the plot. Use "YYYYMMDD" or "YYYYMMDDHH".
        y_lim (tuple (float, float), optional): Bottom and top limit of the y axis for the plot.
        
    Example use:
        peek_motion_sum("2025050500", "2025052512", box_number=8, x_lim=("20250501","20250601"), y_lim=(0,1e5))
        peek_motion_sum("20250505", "20250525")
    """
    # Create directory for motion sum plots
    plot_dir = os.path.join(script_directory, MOTION_AMP_FOLDER)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Convert date strings to Unix timestamp range
    start_ts, end_ts = get_unix_time_range(start_date_str, end_date_str)
    
    # Determine boxes to process
    boxes = [box_number] if box_number is not None else box_n
    
    for box in boxes:
        motion_file = UNIFIED_FOLDER + "/" + f"motion_{box}.bin"
        if not os.path.exists(motion_file):
            print(f"Motion file for box {box} not found")
            continue

        try:
            # Load motion data from binary file
            data_dtype = np.dtype([
                ("epoch_time", np.int32),
                ("vReal", np.float32, (2048,))
            ])
            motion_data = np.fromfile(motion_file, dtype=data_dtype)
            
            # Filter data by time range
            time_mask = (motion_data["epoch_time"] >= start_ts) & (motion_data["epoch_time"] <= end_ts)
            filtered_data = motion_data[time_mask]
            
            if filtered_data.size == 0:
                print(f"No data for box {box} in range")
                continue
                
            # Calculate sum of absolute acceleration values
            time_list = []
            abs_sum_list = []
            
            for rec in filtered_data:
                # Convert timestamp to datetime
                dt = datetime.datetime.utcfromtimestamp(rec["epoch_time"])
                time_list.append(dt)
                
                # Calculate sum of absolute acceleration values
                abs_sum = np.sum(np.abs(rec["vReal"]))
                abs_sum_list.append(abs_sum)
            
            # Create plot
            plt.figure(figsize=(12, 6))
            plt.scatter(
                time_list,
                abs_sum_list,
                marker="o",
                linestyle="None",
                label=f"Absolute Acceleration Sum"
            )
            plt.title(f"Motion Amplitude of the Pole – Box {box}", fontsize=title_size)
            plt.xlabel("UTC Time", size=title_size)
            plt.ylabel("Motion Amplitude", size=title_size)
            
            print("Mean value:", np.nanmean(abs_sum_list))
            print("std:", np.nanstd(abs_sum_list))
            
            if x_lim:   
                x_lim_dt = []
                for date_str in x_lim:
                    if len(date_str) == 8:  # YYYYMMDD
                        x_lim_dt.append(datetime.datetime.strptime(date_str, "%Y%m%d"))
                    elif len(date_str) == 10:  # YYYYMMDDHH
                        x_lim_dt.append(datetime.datetime.strptime(date_str, "%Y%m%d%H"))
                    elif len(date_str) == 12:  # YYYYMMDDHHMM
                        x_lim_dt.append(datetime.datetime.strptime(date_str, "%Y%m%d%H%M"))
                    else:
                        raise ValueError("x_lim dates must be YYYYMMDD, YYYYMMDDHH, or YYYYMMDDHHMM format")
                plt.xlim(x_lim_dt)
            
            if y_lim:
                plt.ylim(y_lim)
            
            plt.legend(fontsize=20)
            plt.grid()
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save plot
            out_path = os.path.join(plot_dir, f"box_{box}_motion_sum.png")
            plt.savefig(out_path)
            plt.close()
            print(f"Saved motion sum plot: {out_path}")
            
        except Exception as e:
            print(f"Error processing box {box}: {str(e)}")

def peek_spectrum(time_str, sample_num=None, box_number=None, y_lim=False, x_lim=(0,100), title_size=12):
    """
    Acceleration amplitude as a function of time as read from the motion bin file, with peak identification.

    Parameters:
        time_str (str): Date in YYYYMMDDHH format. The data within a wake-up cycle that is closest
            to time_str will be plotted.
        sample_num (int or None): 
            - If an integer (1-10), plots that specific sample
            - If None, plots every sample in the chosen wake-up cycle in separate plots
            - If -1, plots the sum of all spectra from the wake-up cycle
        box_number (int, optional): Specific box number to plot. If None, plots all boxes.
        y_lim (tuple (float, float), optional): Bottom and top limit of the y axis for the plot.
        x_lim (tuple (float, float), optional): Left and right limit of the x axis for the plot.

    Example use:
        peek_spectrum("2025051400", sample_num=3, box_number=8, y_lim=(0,200))
        peek_spectrum("2025051500", sample_num=-1, title_size=24)
    """
    # Validate time_str and convert to full format
    if isinstance(time_str, str) and len(time_str) == 8 and time_str.isdigit():
        # YYYYMMDD format - add '0000' for hours and minutes
        time_str_full = time_str + "0000"
        dt_req = datetime.datetime.strptime(time_str_full, "%Y%m%d%H%M")
    elif isinstance(time_str, str) and len(time_str) == 10 and time_str.isdigit():
        # YYYYMMDDHH format - add '00' for minutes
        time_str_full = time_str + "00"
        dt_req = datetime.datetime.strptime(time_str_full, "%Y%m%d%H%M")
    elif isinstance(time_str, str) and len(time_str) == 12 and time_str.isdigit():
        # YYYYMMDDHHMM format
        time_str_full = time_str
        dt_req = datetime.datetime.strptime(time_str_full, "%Y%m%d%H%M")
    else:
        print(f"ERROR: time_str must be 'YYYYMMDD', 'YYYYMMDDHH', or 'YYYYMMDDHHMM'; got '{time_str}'.")
        return

    # Convert to Unix timestamp (UTC)
    desired_ts = int(dt_req.replace(tzinfo=datetime.timezone.utc).timestamp())

    # Extract the day portion for bounding the search
    day_str = time_str_full[:8]  # "YYYYMMDD"

    # Build directory for spectrum plots
    plot_dir = os.path.join(script_directory, SPECTRUM_FOLDER)
    os.makedirs(plot_dir, exist_ok=True)

    # Convert the single-day string into UTC-based Unix-timestamp bounds
    start_ts, end_ts = get_unix_time_range(day_str, day_str)

    # Decide which boxes to run through
    boxes = [box_number] if (box_number is not None) else box_n

    for box in boxes:
        motion_file = UNIFIED_FOLDER + "/" + f"motion_{box}.bin"
        if not os.path.exists(motion_file):
            print(f"[Box {box}] Motion file not found → skipping.")
            continue

        # Load the entire .bin as a structured array
        data_dtype = np.dtype([
            ("epoch_time", np.int32),
            ("vReal",     np.float32, (2048,))
        ])
        motion_data = np.fromfile(motion_file, dtype=data_dtype)

        # Filter down to records whose epoch_time is on that day
        mask = (motion_data["epoch_time"] >= start_ts) & (motion_data["epoch_time"] <= end_ts)
        day_data = motion_data[mask]
        if day_data.size == 0:
            print(f"[Box {box}] No motion records on {day_str} → skipping.")
            continue

        # Find all unique epoch_time values (each is one wake-up time)
        unique_times = np.unique(day_data["epoch_time"])
        if unique_times.shape[0] == 0:
            print(f"[Box {box}] No wake-up timestamps found on {day_str}.")
            continue

        # Find the closest wake-up time to desired_ts
        diffs = np.abs(unique_times.astype(np.int64) - np.int64(desired_ts))
        idx = int(np.argmin(diffs))   # 0-based index in unique_times
        target_time = unique_times[idx]
        chosen_wake_iter = idx + 1    # For user info (1-based)

        # Inform the user which wake-up was chosen
        dt_chosen = datetime.datetime.utcfromtimestamp(int(target_time))
        time_chosen_str = dt_chosen.strftime("%Y-%m-%d %H:%M")
        print(
            f"[Box {box}] Requested {time_str[:4]}-{time_str[4:6]}-{time_str[6:8]} "
            f"{time_str[8:10]}:00 UTC → closest wake-up at "
            f"{time_chosen_str} UTC (wake #{chosen_wake_iter} of {unique_times.shape[0]})"
        )

        # Extract only the samples at that exact epoch_time
        group = day_data[day_data["epoch_time"] == target_time]
        if group.shape[0] == 0:
            print(f"[Box {box}] No samples at epoch_time={target_time} -> skipping.")
            continue

        # Format the chosen time for filenames
        file_time_str = dt_chosen.strftime("%Y%m%d%H%M")
        
        # Handle special case for sample_num=-1 (sum of all spectra)
        if sample_num == -1:
            # Initialize an array to hold the summed magnitude spectrum
            summed_mag = None
            freqs = None
            
            # Compute FFT for each sample and sum the magnitudes
            for i, rec in enumerate(group):
                v = rec["vReal"]
                fft_vals = np.fft.rfft(v)
                mag = np.abs(fft_vals)
                
                if summed_mag is None:
                    summed_mag = mag
                    # Compute frequencies (only need to do this once)
                    N = v.size
                    fs = 200.0
                    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
                else:
                    summed_mag += mag
            
            # Create plot
            plt.figure(figsize=(12, 6))
            plt.grid()
            plt.plot(freqs[2:], summed_mag[2:], linewidth=1, color="k")
            
            # Identify peaks
            peaks_dict = identify_peaks(summed_mag, freqs)
            
            # Plot fundamental frequencies if found
            if peaks_dict['fundamental1'] is not None and peaks_dict['fundamental1']['freq'] is not None:
                idx = np.argmin(np.abs(freqs - peaks_dict['fundamental1']['freq']))
                f1_freq = freqs[idx] # This is saved for the harmonics
                plt.plot(freqs[idx], summed_mag[idx], 'ro', markersize=8, label='Fundamental 1')
                plt.text(freqs[idx], summed_mag[idx], f"  $F_1$\n${freqs[idx]:.2f}$ Hz\n", va='bottom', ha='left')
            
            if peaks_dict['fundamental2'] is not None and peaks_dict['fundamental2']['freq'] is not None:
                idx = np.argmin(np.abs(freqs - peaks_dict['fundamental2']['freq']))
                f2_freq = freqs[idx] # This is saved for the harmonics
                plt.plot(freqs[idx], summed_mag[idx], 'bo', markersize=8, label='Fundamental 2')
                plt.text(freqs[idx], summed_mag[idx], f"  $F_2$\n${freqs[idx]:.2f}$ Hz\n", va='bottom', ha='left')
            
            # Plot harmonics if found
            if peaks_dict['fundamental1'] is not None:
                for h in peaks_dict['fundamental1']['harmonics']:
                    idx = np.argmin(np.abs(freqs - h))
                    plt.plot(freqs[idx], summed_mag[idx], 'rs', markersize=6, label='Harmonics of F1')
                    which_harmonic = int(np.round(freqs[idx]/f1_freq))
                    plt.text(freqs[idx], summed_mag[idx], f"  ${which_harmonic} F_1$\n${freqs[idx]:.2f}$ Hz\n", va='bottom', ha='left')
            
            if peaks_dict['fundamental2'] is not None:
                for h in peaks_dict['fundamental2']['harmonics']:
                    idx = np.argmin(np.abs(freqs - h))
                    plt.plot(freqs[idx], summed_mag[idx], 'bs', markersize=6, label='Harmonics of F2')
                    which_harmonic = int(np.round(freqs[idx]/f2_freq))
                    plt.text(freqs[idx], summed_mag[idx], f"  ${which_harmonic} F_2$\n${freqs[idx]:.2f}$ Hz\n", va='bottom', ha='left')
            
            # Plot modulated peaks if found
            if peaks_dict['fundamental2'] is not None:
                for m in peaks_dict['fundamental2']['modulated']:
                    idx = np.argmin(np.abs(freqs - m))
                    plt.plot(freqs[idx], summed_mag[idx], 'g^', markersize=6, label='Modulated peaks')
                    label = "M_+" if (freqs[idx] - f2_freq) > 0 else "M_-"
                    offset = "left" if (freqs[idx] - f2_freq) > 0 else "right"
                    plt.text(freqs[idx], summed_mag[idx], f"  ${label}$\n${freqs[idx]:.2f}$ Hz\n", va='bottom', ha=offset)
            
            
            # Plot other peaks if found
            if 'other_peaks' in peaks_dict:
                for op in peaks_dict['other_peaks']:
                    idx = np.argmin(np.abs(freqs - op))
                    plt.plot(freqs[idx], summed_mag[idx], 'kx', markersize=6, label='Other peaks')
                    plt.text(freqs[idx], summed_mag[idx], f"  Other\n${freqs[idx]:.2f}$ Hz\n", va='bottom', ha='left')
            
            
            # Add legend if any peaks were found
            if plt.gca().has_data():
                handles, labels = plt.gca().get_legend_handles_labels()
                by_label = dict(zip(labels, handles))  # Remove duplicate labels
                plt.legend(by_label.values(), by_label.keys(), loc='best', fontsize=15)
            
            plt.title(
                f"Motion Frequency Spectrum - Box {box} at {time_chosen_str} UTC\n"
                f"All samples of wake #{chosen_wake_iter} (n={len(group)})"
                , fontsize=title_size
            )
            plt.xlabel("Frequency (Hz)", size=title_size)
            plt.xlim(x_lim)
            plt.ylabel("Magnitude", size=title_size)
            plt.tight_layout()
            
            if y_lim:
                plt.ylim(y_lim)
            else:
                lim = plt.ylim()
                plt.ylim((lim[0],lim[1]*1.1))

            # Save the figure
            out_fname = f"box_{box}_spectrum_sum_{file_time_str}.png"
            out_path = os.path.join(plot_dir, out_fname)
            plt.savefig(out_path)
            plt.close()
            print(f"[Box {box}] Saved: {out_path}")
            continue

        # Determine samples to plot
        if sample_num is None:
            samples_to_plot = range(1, len(group) + 1)  # All samples (1-based)
        else:
            if sample_num < 1 or sample_num > len(group):
                print(
                    f"[Box {box}] Requested sample_num={sample_num} out of range "
                    f"(1 to {len(group)}) at wake-up #{chosen_wake_iter} → skipping."
                )
                continue
            samples_to_plot = [sample_num]
        
        for sample in samples_to_plot:
            rec = group[sample - 1]
            v = rec["vReal"]   # 2048-point time series for that single sample

            # Compute one-sided FFT (fs = 200 Hz)
            N = v.size                # should be 2048
            fs = 200.0                # Hz
            fft_vals = np.fft.rfft(v)
            mag = np.abs(fft_vals)    # magnitude spectrum
            freqs = np.fft.rfftfreq(N, d=1.0 / fs)

            # Create plot for this sample
            plt.figure(figsize=(12, 6))
            plt.grid()
            min_index =  np.argmin(freqs-MIN_FREQ)
            plt.plot(freqs[min_index:], mag[min_index:], linewidth=1, color="k")
            
            # Identify peaks
            peaks_dict = identify_peaks(mag, freqs)
            print("Peaks:", peaks_dict)
            
            # Plot fundamental frequencies if found
            if peaks_dict['fundamental1'] is not None and peaks_dict['fundamental1']['freq'] is not None:
                idx = np.argmin(np.abs(freqs - peaks_dict['fundamental1']['freq']))
                f1_freq = freqs[idx] # This is saved for the harmonics
                plt.plot(freqs[idx], mag[idx], 'ro', markersize=8, label='Fundamental 1')
                plt.text(freqs[idx], mag[idx], f"  $F_1$\n${freqs[idx]:.2f}$ Hz\n", va='bottom', ha='center')
            
            if peaks_dict['fundamental2'] is not None and peaks_dict['fundamental2']['freq'] is not None:
                idx = np.argmin(np.abs(freqs - peaks_dict['fundamental2']['freq']))
                f2_freq = freqs[idx] # This is saved for the harmonics
                plt.plot(freqs[idx], mag[idx], 'bo', markersize=8, label='Fundamental 2')
                plt.text(freqs[idx], mag[idx], f"  $F_2$\n${freqs[idx]:.2f}$ Hz\n", va='bottom', ha='center')
            
            # Plot harmonics if found
            if peaks_dict['fundamental1'] is not None:
                for h in peaks_dict['fundamental1']['harmonics']:
                    idx = np.argmin(np.abs(freqs - h))
                    plt.plot(freqs[idx], mag[idx], 'rs', markersize=6, label='Harmonics of F1')
                    which_harmonic = int(np.round(freqs[idx]/f1_freq))
                    plt.text(freqs[idx], mag[idx], f"  ${which_harmonic} F_1$\n${freqs[idx]:.2f}$ Hz\n", va='bottom', ha='center')
            
            if peaks_dict['fundamental2'] is not None:
                for h in peaks_dict['fundamental2']['harmonics']:
                    idx = np.argmin(np.abs(freqs - h))
                    plt.plot(freqs[idx], mag[idx], 'bs', markersize=6, label='Harmonics of F2')
                    which_harmonic = int(np.round(freqs[idx]/f2_freq))
                    plt.text(freqs[idx], mag[idx], f"  ${which_harmonic} F_2$\n${freqs[idx]:.2f}$ Hz\n", va='bottom', ha='center')
            
            # Plot modulated peaks if found
            if peaks_dict['fundamental2'] is not None:
                for m in peaks_dict['fundamental2']['modulated']:
                    idx = np.argmin(np.abs(freqs - m))
                    plt.plot(freqs[idx], mag[idx], 'g^', markersize=6, label='Modulated peaks')
                    label = "M_+" if (freqs[idx] - f2_freq) > 0 else "M_-"
                    offset = "left" if (freqs[idx] - f2_freq) > 0 else "right"
                    plt.text(freqs[idx], mag[idx], f"  ${label}$\n${freqs[idx]:.2f}$ Hz\n", va='bottom', ha=offset)
            
            
            # Plot other peaks if found
            if 'other_peaks' in peaks_dict:
                for op in peaks_dict['other_peaks']:
                    idx = np.argmin(np.abs(freqs - op))
                    plt.plot(freqs[idx], mag[idx], 'kx', markersize=6, label='Other peaks')
                    plt.text(freqs[idx], mag[idx], f"  Other\n${freqs[idx]:.2f}$ Hz\n", va='bottom', ha='center')
            
            # Add legend if any peaks were found
            if plt.gca().has_data():
                handles, labels = plt.gca().get_legend_handles_labels()
                by_label = dict(zip(labels, handles))  # Remove duplicate labels
                plt.legend(by_label.values(), by_label.keys(), loc='best', fontsize=15)
            
            plt.title(
                f"Motion Frequency Spectrum - Box {box} at {time_chosen_str} UTC\n"
                f"Sample #{sample} of wake #{chosen_wake_iter}"
                , fontsize=title_size
            )
            plt.xlabel("Frequency (Hz)", size=title_size)
            plt.xlim(x_lim)
            plt.ylabel("Magnitude", size=title_size)
            plt.tight_layout()
            
            if y_lim:
                plt.ylim(y_lim)
            else:
                lim = plt.ylim()
                plt.ylim((lim[0],lim[1]*1.1))

            # Save the figure
            out_fname = f"box_{box}_spectrum_{file_time_str}_sample{sample}.png"
            out_path = os.path.join(plot_dir, out_fname)
            plt.savefig(out_path)
            plt.close()
            print(f"[Box {box}] Saved: {out_path}")

def peek_acc(time_str, sample_num=None, box_number=None, y_lim=False, title_size=12):
    """
    Plots raw acceleration data as a function of time from the motion bin file.

    Parameters:
        time_str (str): Date in YYYYMMDDHH format. The data within a wake-up cycle that is closest
            to time_str will be plotted.
        sample_num (int or None): If an integer, number of the sample you want to plot within a wake-up cycle (usually 1-10).
            If None, plots every sample in the chosen wake-up cycle in separate plots.
        box_number (int, optional): Specific box number to plot. If None, plots all boxes.
        y_lim (tuple (float, float), optional): Bottom and top limit of the y axis for the plot.
        
    Example use:
        peek_acc("2025051400", sample_num=3, box_number=16, y_lim=(0, 5))
        peek_acc("2025051412")
    """
    # Validate time_str and convert to datetime
    if isinstance(time_str, str) and len(time_str) == 8 and time_str.isdigit():
        # YYYYMMDD format - use midnight
        dt_req = datetime.datetime.strptime(time_str, "%Y%m%d")
    elif isinstance(time_str, str) and len(time_str) == 10 and time_str.isdigit():
        # YYYYMMDDHH format
        dt_req = datetime.datetime.strptime(time_str, "%Y%m%d%H")
    elif isinstance(time_str, str) and len(time_str) == 12 and time_str.isdigit():
        # YYYYMMDDHHMM format
        dt_req = datetime.datetime.strptime(time_str, "%Y%m%d%H%M")
    else:
        print(f"ERROR: time_str must be 'YYYYMMDD', 'YYYYMMDDHH', or 'YYYYMMDDHHMM'; got '{time_str}'.")
        return

    # Convert to Unix timestamp (UTC, minutes & seconds = 0)
    desired_ts = int(dt_req.replace(tzinfo=datetime.timezone.utc).timestamp())

    # Extract the day portion for bounding the search
    day_str = time_str[:8]  # "YYYYMMDD"

    # Create directory for acceleration plots
    plot_dir = os.path.join(script_directory, ACCELERATION_FOLDER)
    os.makedirs(plot_dir, exist_ok=True)

    # Convert the single-day string into UTC-based Unix-timestamp bounds
    start_ts, end_ts = get_unix_time_range(day_str, day_str)

    # Determine boxes to process
    boxes = [box_number] if box_number is not None else box_n

    for box in boxes:
        motion_file = UNIFIED_FOLDER + "/" + f"motion_{box}.bin"
        if not os.path.exists(motion_file):
            print(f"[Box {box}] Motion file not found → skipping.")
            continue

        # Load the entire .bin as a structured array
        data_dtype = np.dtype([
            ("epoch_time", np.int32),
            ("vReal",     np.float32, (2048,))
        ])
        motion_data = np.fromfile(motion_file, dtype=data_dtype)

        # Filter down to records whose epoch_time is on that day
        mask = (motion_data["epoch_time"] >= start_ts) & (motion_data["epoch_time"] <= end_ts)
        day_data = motion_data[mask]
        if day_data.size == 0:
            print(f"[Box {box}] No motion records on {day_str} → skipping.")
            continue

        # Find all unique epoch_time values (each is one wake-up time)
        unique_times = np.unique(day_data["epoch_time"])
        if unique_times.shape[0] == 0:
            print(f"[Box {box}] No wake-up timestamps found on {day_str}.")
            continue

        # Find the closest wake-up time to desired_ts
        diffs = np.abs(unique_times.astype(np.int64) - np.int64(desired_ts))
        idx = int(np.argmin(diffs))   # 0-based index in unique_times
        target_time = unique_times[idx]
        chosen_wake_iter = idx + 1    # For user info (1-based)

        # Inform the user which wake-up was chosen
        dt_chosen = datetime.datetime.utcfromtimestamp(int(target_time))
        time_chosen_str = dt_chosen.strftime("%Y-%m-%d %H:%M")
        print(
            f"[Box {box}] Requested {time_str[:4]}-{time_str[4:6]}-{time_str[6:8]} "
            f"{time_str[8:10]}:00 UTC → closest wake-up at "
            f"{time_chosen_str} UTC (wake #{chosen_wake_iter} of {unique_times.shape[0]})"
        )

        # Extract only the samples at that exact epoch_time
        group = day_data[day_data["epoch_time"] == target_time]
        if group.shape[0] == 0:
            print(f"[Box {box}] No samples at epoch_time={target_time} → skipping.")
            continue

        # Determine samples to plot
        if sample_num is None:
            samples_to_plot = range(1, len(group) + 1)  # All samples (1-based)
        else:
            if sample_num < 1 or sample_num > len(group):
                print(
                    f"[Box {box}] Requested sample_num={sample_num} out of range "
                    f"(1 to {len(group)}) at wake-up #{chosen_wake_iter} → skipping."
                )
                continue
            samples_to_plot = [sample_num]

        # Format the chosen time for filenames
        file_time_str = dt_chosen.strftime("%Y%m%d%H%M")
        
        for sample in samples_to_plot:
            rec = group[sample - 1]
            v = rec["vReal"]   # 2048-point time series for that single sample
            
            # Create time array (sampling rate = 200 Hz)
            time_array = np.arange(len(v)) / 200.0  # Time in seconds
            
            plt.figure(figsize=(12, 6))
            plt.grid()
            plt.plot(time_array, v, linewidth=1)
            
            print("Mean value:", np.nanmean(v))
            print("std:", np.nanstd(v))
            
            plt.title(
                f"Acceleration Amplitude as a Function of Time at {time_chosen_str} UTC - Box {box}\n"
                f"Sample #{sample} of wake #{chosen_wake_iter}"
                , fontsize=title_size
            )
            plt.xlabel("Time (seconds)", size=title_size)
            plt.ylabel("Acceleration Amplitude", size=title_size)
            plt.tight_layout()
            
            if y_lim:
                plt.ylim(y_lim)

            # Save the figure
            out_fname = f"box_{box}_acc_{file_time_str}_sample{sample}.png"
            out_path = os.path.join(plot_dir, out_fname)
            plt.savefig(out_path)
            plt.close()
            print(f"[Box {box}] Saved: {out_path}")
















def peek_spectrum_glued(time_str, box_number=None, y_lim=False, x_lim=(0,100), title_size=12):
    """
    Acceleration amplitude spectrum from concatenated motion data in a wake-up cycle.

    Parameters:
        time_str (str): Date in YYYYMMDDHH format. The data within a wake-up cycle that is closest
            to time_str will be plotted.
        box_number (int, optional): Specific box number to plot. If None, plots all boxes.
        y_lim (tuple (float, float), optional): Bottom and top limit of the y axis for the plot.
        x_lim (tuple (float, float), optional): Left and right limit of the x axis for the plot.

    Example use:
        peek_spectrum_glued("2025051400", box_number=8, y_lim=(0,200))
    """
    # Validate time_str and convert to datetime
    if isinstance(time_str, str) and len(time_str) == 8 and time_str.isdigit():
        # YYYYMMDD format - use midnight
        dt_req = datetime.datetime.strptime(time_str, "%Y%m%d")
    elif isinstance(time_str, str) and len(time_str) == 10 and time_str.isdigit():
        # YYYYMMDDHH format
        dt_req = datetime.datetime.strptime(time_str, "%Y%m%d%H")
    elif isinstance(time_str, str) and len(time_str) == 12 and time_str.isdigit():
        # YYYYMMDDHHMM format
        dt_req = datetime.datetime.strptime(time_str, "%Y%m%d%H%M")
    else:
        print(f"ERROR: time_str must be 'YYYYMMDD', 'YYYYMMDDHH', or 'YYYYMMDDHHMM'; got '{time_str}'.")
        return

    # Convert to Unix timestamp (UTC, minutes & seconds = 0)
    desired_ts = int(dt_req.replace(tzinfo=datetime.timezone.utc).timestamp())

    # Extract the day portion for bounding the search
    day_str = time_str[:8]  # "YYYYMMDD"

    # Build directory for spectrum plots
    plot_dir = os.path.join(script_directory, SPECTRUM_FOLDER)
    os.makedirs(plot_dir, exist_ok=True)

    # Convert the single-day string into UTC-based Unix-timestamp bounds
    start_ts, end_ts = get_unix_time_range(day_str, day_str)

    # Decide which boxes to run through
    boxes = [box_number] if (box_number is not None) else box_n

    for box in boxes:
        motion_file = UNIFIED_FOLDER + "/" + f"motion_{box}.bin"
        if not os.path.exists(motion_file):
            print(f"[Box {box}] Motion file not found → skipping.")
            continue

        # Load the entire .bin as a structured array
        data_dtype = np.dtype([
            ("epoch_time", np.int32),
            ("vReal",     np.float32, (2048,))
        ])
        motion_data = np.fromfile(motion_file, dtype=data_dtype)

        # Filter down to records whose epoch_time is on that day
        mask = (motion_data["epoch_time"] >= start_ts) & (motion_data["epoch_time"] <= end_ts)
        day_data = motion_data[mask]
        if day_data.size == 0:
            print(f"[Box {box}] No motion records on {day_str} → skipping.")
            continue

        # Find all unique epoch_time values (each is one wake-up time)
        unique_times = np.unique(day_data["epoch_time"])
        if unique_times.shape[0] == 0:
            print(f"[Box {box}] No wake-up timestamps found on {day_str}.")
            continue

        # Find the closest wake-up time to desired_ts
        diffs = np.abs(unique_times.astype(np.int64) - np.int64(desired_ts))
        idx = int(np.argmin(diffs))   # 0-based index in unique_times
        target_time = unique_times[idx]
        chosen_wake_iter = idx + 1    # For user info (1-based)

        # Inform the user which wake-up was chosen
        dt_chosen = datetime.datetime.utcfromtimestamp(int(target_time))
        time_chosen_str = dt_chosen.strftime("%Y-%m-%d %H:%M")
        print(
            f"[Box {box}] Requested {time_str[:4]}-{time_str[4:6]}-{time_str[6:8]} "
            f"{time_str[8:10]}:00 UTC → closest wake-up at "
            f"{time_chosen_str} UTC (wake #{chosen_wake_iter} of {unique_times.shape[0]})"
        )

        # Extract only the samples at that exact epoch_time
        group = day_data[day_data["epoch_time"] == target_time]
        if group.shape[0] == 0:
            print(f"[Box {box}] No samples at epoch_time={target_time} -> skipping.")
            continue

        # Format the chosen time for filenames
        file_time_str = dt_chosen.strftime("%Y%m%d%H%M")
        
        # Concatenate all samples into a single array
        glued_data = np.concatenate([rec["vReal"] for rec in group])
        n_samples = len(group)
        total_points = len(glued_data)
        
        # Compute FFT for the glued data
        N = glued_data.size
        fs = 200.0  # Sampling frequency in Hz
        fft_vals = np.fft.rfft(glued_data)
        mag = np.abs(fft_vals)
        freqs = np.fft.rfftfreq(N, d=1.0 / fs)
        
        # Create plot
        plt.figure(figsize=(12, 6))
        plt.grid()
        plt.plot(freqs[2:], mag[2:], linewidth=1, color="k")
        
        # Identify peaks
        peaks_dict = identify_peaks(mag, freqs)
        
        # Plot fundamental frequencies if found
        if peaks_dict['fundamental1'] is not None and peaks_dict['fundamental1']['freq'] is not None:
            idx = np.argmin(np.abs(freqs - peaks_dict['fundamental1']['freq']))
            f1_freq = freqs[idx]
            plt.plot(freqs[idx], mag[idx], 'ro', markersize=8, label='Fundamental 1')
            plt.text(freqs[idx], mag[idx], f"  $F_1$\n${freqs[idx]:.2f}$ Hz\n", va='bottom', ha='left')
        
        if peaks_dict['fundamental2'] is not None and peaks_dict['fundamental2']['freq'] is not None:
            idx = np.argmin(np.abs(freqs - peaks_dict['fundamental2']['freq']))
            f2_freq = freqs[idx]
            plt.plot(freqs[idx], mag[idx], 'bo', markersize=8, label='Fundamental 2')
            plt.text(freqs[idx], mag[idx], f"  $F_2$\n${freqs[idx]:.2f}$ Hz\n", va='bottom', ha='left')
        
        # Plot harmonics if found
        if peaks_dict['fundamental1'] is not None:
            for h in peaks_dict['fundamental1']['harmonics']:
                idx = np.argmin(np.abs(freqs - h))
                plt.plot(freqs[idx], mag[idx], 'rs', markersize=6, label='Harmonics of F1')
                which_harmonic = int(np.round(freqs[idx]/f1_freq))
                plt.text(freqs[idx], mag[idx], f"  ${which_harmonic} F_1$\n${freqs[idx]:.2f}$ Hz\n", va='bottom', ha='left')
        
        if peaks_dict['fundamental2'] is not None:
            for h in peaks_dict['fundamental2']['harmonics']:
                idx = np.argmin(np.abs(freqs - h))
                plt.plot(freqs[idx], mag[idx], 'bs', markersize=6, label='Harmonics of F2')
                which_harmonic = int(np.round(freqs[idx]/f2_freq))
                plt.text(freqs[idx], mag[idx], f"  ${which_harmonic} F_2$\n${freqs[idx]:.2f}$ Hz\n", va='bottom', ha='left')
        
        # Plot modulated peaks if found
        if peaks_dict['fundamental2'] is not None:
            for m in peaks_dict['fundamental2']['modulated']:
                idx = np.argmin(np.abs(freqs - m))
                plt.plot(freqs[idx], mag[idx], 'g^', markersize=6, label='Modulated peaks')
                label = "M_+" if (freqs[idx] - f2_freq) > 0 else "M_-"
                offset = "left" if (freqs[idx] - f2_freq) > 0 else "right"
                plt.text(freqs[idx], mag[idx], f"  ${label}$\n${freqs[idx]:.2f}$ Hz\n", va='bottom', ha=offset)
        
        # Plot other peaks if found
        if 'other_peaks' in peaks_dict:
            for op in peaks_dict['other_peaks']:
                idx = np.argmin(np.abs(freqs - op))
                plt.plot(freqs[idx], mag[idx], 'kx', markersize=6, label='Other peaks')
                plt.text(freqs[idx], mag[idx], f"  Other\n${freqs[idx]:.2f}$ Hz\n", va='bottom', ha='left')
        
        # Add legend if any peaks were found
        if plt.gca().has_data():
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))  # Remove duplicate labels
            plt.legend(by_label.values(), by_label.keys(), loc='best', fontsize=15)
        
        plt.title(
            f"Motion Frequency Spectrum (Glued Data) - Box {box} at {time_chosen_str} UTC\n"
            f"Wake #{chosen_wake_iter} (n={n_samples}, {total_points} points)"
            , fontsize=title_size
        )
        plt.xlabel("Frequency (Hz)", size=title_size)
        plt.xlim(x_lim)
        plt.ylabel("Magnitude", size=title_size)
        plt.tight_layout()
        
        if y_lim:
            plt.ylim(y_lim)
        else:
            lim = plt.ylim()
            plt.ylim((lim[0], lim[1]*1.1))

        # Save the figure
        out_fname = f"box_{box}_spectrum_glued_{file_time_str}.png"
        out_path = os.path.join(plot_dir, out_fname)
        plt.savefig(out_path)
        plt.show()
        plt.close()
        print(f"[Box {box}] Saved: {out_path}")








