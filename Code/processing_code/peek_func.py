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
                    else:
                        raise ValueError("x_lim dates must be YYYYMMDD or YYYYMMDDHH format")
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
            
def peek_pdata(parameter, start_date_str, end_date_str, box_number=None, x_lim=False, y_lim=False, title_size=12, overlap=None, overlap_y_lim=False):
    """
    Plots processed data (from processed_data_*.csv files) for a given parameter.

    Parameters:
        parameter (str): The parameter to plot. Options: 'temp', 'motion' (length as computed from motion bin files),
            'data' (length as computed from the data csv files), 'depth', 'f1' (fundamental frequency 1),
            'f2' (fundamental frequency 2), 'both_f' (both f1 and f2), 'l1' (length from fundamental frequency 1),
            'l2' (length from fundamental frequency 2), 'both_l' (both l1 and l2), 'full' (motion + depth), 
            or 'all' (motion + data + depth).
        start_date_str (str): Start date in YYYYMMDD, YYYYMMDDHH, or YYYYMMDDHHMM format.
        end_date_str (str): End date in YYYYMMDD, YYYYMMDDHH, or YYYYMMDDHHMM format.
        box_number (int, optional): Specific box number to plot. If None, plots all boxes.
        x_lim (tuple (str, str), optional): Left and right limit of the x axis for the plot. Use "YYYYMMDD", "YYYYMMDDHH", or "YYYYMMDDHHMM".
        y_lim (tuple (float, float), optional): Bottom and top limit of the y axis for the plot (main parameter).
        overlap (list, optional): List of additional parameters to overlay on a separate y-axis. 
            Example: overlap=['motion', 'depth'] to add motion and depth on the right y-axis.
        overlap_y_lim (tuple (float, float), optional): Bottom and top limit of the right y axis for overlap parameters.
        
    Example use:
        peek_pdata("temp", "202505050000", "202505251200", box_number=8, x_lim=("202505010000","202506010000"), y_lim=(-40,20))
        peek_pdata("motion", "20250505", "20250525")
        peek_pdata("data", "20250505", "20250525", y_lim=(0.8,1))
        peek_pdata("f1", "20250505", "20250525", y_lim=(0,50))
        peek_pdata("f2", "20250505", "20250525", y_lim=(0,100))
        peek_pdata("both_f", "20250505", "20250525", y_lim=(0,100))
        peek_pdata("l1", "20250505", "20250525", y_lim=(0.8,1.2))
        peek_pdata("l2", "20250505", "20250525", y_lim=(0.8,1.2))
        peek_pdata("both_l", "20250505", "20250525", y_lim=(0.8,1.2))
        peek_pdata("full", "20250505", "20250525", y_lim=(0,5))
        peek_pdata("all", "2025050500", "2025052512", y_lim=(0,5))
        peek_pdata("motion", "20250505", "20250525", overlap=['depth', 'data'], y_lim=(0,5), overlap_y_lim=(0,3))
        peek_pdata("temp", "20250505", "20250525", overlap=['motion'], y_lim=(-40,20), overlap_y_lim=(0.8,1.2))
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
            "label": "Computed Length",
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
            "color": "orange"
        },
        "l1": {
            "val_col": 12,  # L1 length value column (from F1)
            "err_col": 13,  # L1 length error column
            "label": "Length from F1",
            "unit": "m",
            "color": "dodgerblue"
        },
        "l2": {
            "val_col": 14,  # L2 length value column (from F2)
            "err_col": 15,  # L2 length error column
            "label": "Length from F2",
            "unit": "m",
            "color": "fuchsia"
        }
    }

    # Validate parameter
    valid_parameters = ["temp", "motion", "data", "depth", "f1", "f2", "both_f", "l1", "l2", "both_l", "all", "full"]
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
                
            # Convert timestamps to datetime - keep original for all parameters
            original_timestamps = [
                datetime.datetime.strptime(str(int(ts)), "%Y%m%d%H%M")
                for ts in filtered_data[:, 0]
            ]
            
            fig, ax1 = plt.subplots(figsize=(12, 6))
            ax2 = None  # Initialize ax2 as None at the top level
            
            # Function to plot a parameter and return statistics
            def plot_parameter(param_name, timestamps, data, axis, alpha=1.0, show_stats=False):
                if param_name not in param_config:
                    print(f"Warning: Unknown parameter '{param_name}' in overlap list")
                    return None, None, None
                    
                config = param_config[param_name]
                vals = data[:, config["val_col"]].copy()
                errs = data[:, config["err_col"]].copy() if config["err_col"] != -1 else None
                
                # Create copies of timestamps for this parameter
                param_timestamps = timestamps.copy()
                
                # For frequency parameters, filter out NaN and zero values
                if param_name in ["f1", "f2"]:
                    valid_mask = ~(np.isnan(vals) | (vals == 0))
                    vals = vals[valid_mask]
                    if errs is not None:
                        errs = errs[valid_mask]
                    param_timestamps = [param_timestamps[i] for i in range(len(param_timestamps)) if valid_mask[i]]
                
                # For length measurements (including L1 and L2), filter out large error values
                if param_name in ["motion", "data", "depth", "l1", "l2"] and errs is not None:
                    large_err_mask = errs > 0.1
                    vals[large_err_mask] = np.nan
                    errs[large_err_mask] = np.nan
                
                label = f"{config['label']} ({config['unit']})"
                axis.errorbar(param_timestamps, vals, yerr=errs, fmt='o', 
                           color=config['color'], capsize=5, label=label, alpha=alpha)
                
                # Calculate statistics
                mean_val = np.nanmean(vals)
                std_val = np.nanstd(vals)
                n_valid = np.sum(~np.isnan(vals))
                err_on_mean = std_val / np.sqrt(n_valid) if n_valid > 0 else np.nan
                
                stats = {
                    'mean': mean_val,
                    'std': std_val,
                    'err_on_mean': err_on_mean,
                    'n_points': n_valid
                }
                
                if show_stats:
                    print(f"[Box {box}] {param_name} - Mean: {mean_val:.4f}, Std: {std_val:.4f}, Error on mean: {err_on_mean:.4f}, N: {n_valid}")
                
                return vals, param_timestamps, stats
            
            # Dictionary to store all statistics for printing
            all_stats = {}
            
            if parameter == "all":
                # Plot data, motion, and depth on the same plot (left axis)
                for param in ["depth", "motion", "data"]:
                    vals, timestamps, stats = plot_parameter(param, original_timestamps, filtered_data, ax1, alpha=0.7, show_stats=False)
                    if stats is not None:
                        all_stats[param] = stats
                
                # Create second y-axis for overlap parameters if specified
                if overlap:
                    ax2 = ax1.twinx()
                    for overlap_param in overlap:
                        vals, timestamps, stats = plot_parameter(overlap_param, original_timestamps, filtered_data, ax2, alpha=0.4, show_stats=False)
                        if stats is not None:
                            all_stats[overlap_param] = stats
                
                # Set plot title and labels
                if overlap:
                    valid_overlap = [p for p in overlap if p in param_config]
                    if valid_overlap:
                        overlap_names = ", ".join([param_config[p]['label'] for p in valid_overlap])
                        ax1.set_title(f"Lengths as Measured by BRACHIOSAURUS with {overlap_names} - Box {box}", fontsize=title_size)
                        
                        # Set y-axis labels
                        ax1.set_ylabel("Length (m)", size=title_size)
                        
                        # For right y-axis, use specific label for single parameter or generic for multiple
                        if len(valid_overlap) == 1:
                            # Single overlap parameter - use its specific label
                            overlap_param = valid_overlap[0]
                            ax2.set_ylabel(f"{param_config[overlap_param]['label']} ({param_config[overlap_param]['unit']})", size=title_size)
                        else:
                            # Multiple overlap parameters - check if units are the same
                            overlap_units = [param_config[p]['unit'] for p in valid_overlap]
                            if len(set(overlap_units)) == 1:
                                ax2.set_ylabel(f"Value ({overlap_units[0]})", size=title_size)
                            else:
                                ax2.set_ylabel("Overlap Parameters", size=title_size)
                        
                        # Color the y-axis tick labels to match their data
                        ax1.tick_params(axis='y', labelcolor='black')
                        ax2.tick_params(axis='y', labelcolor='black')
                    else:
                        ax1.set_title(f"Lengths as Measured by BRACHIOSAURUS Through Time - Box {box}", fontsize=title_size)
                        ax1.set_ylabel("Length (m)", size=title_size)
                else:
                    ax1.set_title(f"Lengths as Measured by BRACHIOSAURUS Through Time - Box {box}", fontsize=title_size)
                    ax1.set_ylabel("Length (m)", size=title_size)
                
                ax1.set_xlabel("UTC Time", size=title_size)
                
            elif parameter == "full":
                # Plot motion and depth on the same plot (left axis)
                for param in ["depth", "motion"]:
                    vals, timestamps, stats = plot_parameter(param, original_timestamps, filtered_data, ax1, alpha=0.7, show_stats=False)
                    if stats is not None:
                        all_stats[param] = stats
                
                # Create second y-axis for overlap parameters if specified
                if overlap:
                    ax2 = ax1.twinx()
                    for overlap_param in overlap:
                        vals, timestamps, stats = plot_parameter(overlap_param, original_timestamps, filtered_data, ax2, alpha=0.4, show_stats=False)
                        if stats is not None:
                            all_stats[overlap_param] = stats
                
                # Set plot title and labels
                if overlap:
                    valid_overlap = [p for p in overlap if p in param_config]
                    if valid_overlap:
                        overlap_names = ", ".join([param_config[p]['label'] for p in valid_overlap])
                        ax1.set_title(f"Computed Length and Depth Sensor Measurements\nwith {overlap_names} - Box {box}", fontsize=title_size)
                        
                        # Set y-axis labels
                        ax1.set_ylabel("Length (m)", size=title_size)
                        
                        # For right y-axis, use specific label for single parameter or generic for multiple
                        if len(valid_overlap) == 1:
                            # Single overlap parameter - use its specific label
                            overlap_param = valid_overlap[0]
                            ax2.set_ylabel(f"{param_config[overlap_param]['label']} ({param_config[overlap_param]['unit']})", size=title_size)
                        else:
                            # Multiple overlap parameters - check if units are the same
                            overlap_units = [param_config[p]['unit'] for p in valid_overlap]
                            if len(set(overlap_units)) == 1:
                                ax2.set_ylabel(f"Value ({overlap_units[0]})", size=title_size)
                            else:
                                ax2.set_ylabel("Overlap Parameters", size=title_size)
                        
                        # Color the y-axis tick labels to match their data
                        ax1.tick_params(axis='y', labelcolor='black')
                        ax2.tick_params(axis='y', labelcolor='black')
                    else:
                        ax1.set_title(f"Motion and Depth Measurements - Box {box}", fontsize=title_size)
                        ax1.set_ylabel("Length (m)", size=title_size)
                else:
                    ax1.set_title(f"Motion and Depth Measurements - Box {box}", fontsize=title_size)
                    ax1.set_ylabel("Length (m)", size=title_size)
                
                ax1.set_xlabel("UTC Time", size=title_size)
                
            elif parameter == "both_f":
                # Plot both f1 and f2 frequencies on the same plot (left axis)
                for param in ["f1", "f2"]:
                    vals, timestamps, stats = plot_parameter(param, original_timestamps, filtered_data, ax1, alpha=0.7, show_stats=False)
                    if stats is not None:
                        all_stats[param] = stats
                
                # Create second y-axis for overlap parameters if specified
                if overlap:
                    ax2 = ax1.twinx()
                    for overlap_param in overlap:
                        vals, timestamps, stats = plot_parameter(overlap_param, original_timestamps, filtered_data, ax2, alpha=0.4, show_stats=False)
                        if stats is not None:
                            all_stats[overlap_param] = stats
                
                # Set plot title and labels
                if overlap:
                    valid_overlap = [p for p in overlap if p in param_config]
                    if valid_overlap:
                        overlap_names = ", ".join([param_config[p]['label'] for p in valid_overlap])
                        ax1.set_title(f"Fundamental Frequencies with {overlap_names} - Box {box}", fontsize=title_size)
                        
                        # Set y-axis labels
                        ax1.set_ylabel("Frequency (Hz)", size=title_size)
                        
                        # For right y-axis, use specific label for single parameter or generic for multiple
                        if len(valid_overlap) == 1:
                            # Single overlap parameter - use its specific label
                            overlap_param = valid_overlap[0]
                            ax2.set_ylabel(f"{param_config[overlap_param]['label']} ({param_config[overlap_param]['unit']})", size=title_size)
                        else:
                            # Multiple overlap parameters - check if units are the same
                            overlap_units = [param_config[p]['unit'] for p in valid_overlap]
                            if len(set(overlap_units)) == 1:
                                ax2.set_ylabel(f"Value ({overlap_units[0]})", size=title_size)
                            else:
                                ax2.set_ylabel("Overlap Parameters", size=title_size)
                        
                        # Color the y-axis tick labels to match their data
                        ax1.tick_params(axis='y', labelcolor='black')
                        ax2.tick_params(axis='y', labelcolor='black')
                    else:
                        ax1.set_title(f"Fundamental Frequencies - Box {box}", fontsize=title_size)
                        ax1.set_ylabel("Frequency (Hz)", size=title_size)
                else:
                    ax1.set_title(f"Fundamental Frequencies - Box {box}", fontsize=title_size)
                    ax1.set_ylabel("Frequency (Hz)", size=title_size)
                
                ax1.set_xlabel("UTC Time", size=title_size)
                
            elif parameter == "both_l":
                # Plot both l1 and l2 lengths on the same plot (left axis)
                for param in ["l1", "l2"]:
                    vals, timestamps, stats = plot_parameter(param, original_timestamps, filtered_data, ax1, alpha=0.7, show_stats=False)
                    if stats is not None:
                        all_stats[param] = stats
                
                # Create second y-axis for overlap parameters if specified
                if overlap:
                    ax2 = ax1.twinx()
                    for overlap_param in overlap:
                        vals, timestamps, stats = plot_parameter(overlap_param, original_timestamps, filtered_data, ax2, alpha=0.4, show_stats=False)
                        if stats is not None:
                            all_stats[overlap_param] = stats
                
                # Set plot title and labels
                if overlap:
                    valid_overlap = [p for p in overlap if p in param_config]
                    if valid_overlap:
                        overlap_names = ", ".join([param_config[p]['label'] for p in valid_overlap])
                        ax1.set_title(f"Lengths from Fundamental Frequencies with {overlap_names} - Box {box}", fontsize=title_size)
                        
                        # Set y-axis labels
                        ax1.set_ylabel("Length (m)", size=title_size)
                        
                        # For right y-axis, use specific label for single parameter or generic for multiple
                        if len(valid_overlap) == 1:
                            # Single overlap parameter - use its specific label
                            overlap_param = valid_overlap[0]
                            ax2.set_ylabel(f"{param_config[overlap_param]['label']} ({param_config[overlap_param]['unit']})", size=title_size)
                        else:
                            # Multiple overlap parameters - check if units are the same
                            overlap_units = [param_config[p]['unit'] for p in valid_overlap]
                            if len(set(overlap_units)) == 1:
                                ax2.set_ylabel(f"Value ({overlap_units[0]})", size=title_size)
                            else:
                                ax2.set_ylabel("Overlap Parameters", size=title_size)
                        
                        # Color the y-axis tick labels to match their data
                        ax1.tick_params(axis='y', labelcolor='black')
                        ax2.tick_params(axis='y', labelcolor='black')
                    else:
                        ax1.set_title(f"Lengths from Fundamental Frequencies - Box {box}", fontsize=title_size)
                        ax1.set_ylabel("Length (m)", size=title_size)
                else:
                    ax1.set_title(f"Lengths from Fundamental Frequencies - Box {box}", fontsize=title_size)
                    ax1.set_ylabel("Length (m)", size=title_size)
                
                ax1.set_xlabel("UTC Time", size=title_size)
                
            else:
                # Plot main parameter on left axis
                main_vals, main_timestamps, main_stats = plot_parameter(parameter, original_timestamps, filtered_data, ax1, alpha=1.0, show_stats=True)
                if main_stats is not None:
                    all_stats[parameter] = main_stats
                
                # Create second y-axis for overlap parameters if specified
                if overlap:
                    ax2 = ax1.twinx()
                    for overlap_param in overlap:
                        vals, timestamps, stats = plot_parameter(overlap_param, original_timestamps, filtered_data, ax2, alpha=0.4, show_stats=False)
                        if stats is not None:
                            all_stats[overlap_param] = stats
                
                # Set plot title and labels
                if overlap:
                    # Filter out invalid overlap parameters
                    valid_overlap = [p for p in overlap if p in param_config]
                    if valid_overlap:
                        overlap_names = ", ".join([param_config[p]['label'] for p in valid_overlap])
                        ax1.set_title(f"Processed {param_config[parameter]['label']} with {overlap_names} - Box {box}", fontsize=title_size)
                        
                        # Set y-axis labels
                        ax1.set_ylabel(f"{param_config[parameter]['label']} ({param_config[parameter]['unit']})", 
                                      size=title_size)
                        
                        # For right y-axis, use specific label for single parameter or generic for multiple
                        if len(valid_overlap) == 1:
                            # Single overlap parameter - use its specific label
                            overlap_param = valid_overlap[0]
                            ax2.set_ylabel(f"{param_config[overlap_param]['label']} ({param_config[overlap_param]['unit']})", size=title_size)
                        else:
                            # Multiple overlap parameters - check if units are the same
                            overlap_units = [param_config[p]['unit'] for p in valid_overlap]
                            if len(set(overlap_units)) == 1:
                                ax2.set_ylabel(f"Value ({overlap_units[0]})", size=title_size)
                            else:
                                ax2.set_ylabel("Overlap Parameters", size=title_size)
                        
                        # Color the y-axis labels to match their data
                        ax1.tick_params(axis='y')
                        ax2.tick_params(axis='y', labelcolor='black')
                        
                    else:
                        ax1.set_title(f"Processed {param_config[parameter]['label']} - Box {box}", fontsize=title_size)
                        ax1.set_ylabel(f"{param_config[parameter]['label']} ({param_config[parameter]['unit']})", size=title_size)
                else:
                    ax1.set_title(f"Processed {param_config[parameter]['label']} - Box {box}", fontsize=title_size)
                    ax1.set_ylabel(f"{param_config[parameter]['label']} ({param_config[parameter]['unit']})", size=title_size)
                
                ax1.set_xlabel("UTC Time", size=title_size)
            
            # Print statistics for all parameters
            print(f"\n[Box {box}] Statistics Summary:")
            print("-" * 110)
            for param_name, stats in all_stats.items():
                param_label = param_config[param_name]['label']
                param_unit = param_config[param_name]['unit']
                print(f"{param_label:30s} | Mean: {stats['mean']:8.4f} {param_unit:3s} | Std: {stats['std']:8.4f} {param_unit:3s} | Error on mean: {stats['err_on_mean']:8.4f} {param_unit:3s} | N: {stats['n_points']:4d}")
            print("-" * 110)
            
            # Apply x and y limits
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
                ax1.set_xlim(x_lim_dt)
            
            if y_lim:
                ax1.set_ylim(y_lim)
                
            if overlap_y_lim and ax2 is not None:
                ax2.set_ylim(overlap_y_lim)
            
            # Combine legends from both axes if we have overlap
            if overlap and ax2 is not None:
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=15, loc='upper left')
            else:
                ax1.legend(fontsize=15, loc='best')
            
            ax1.grid()
            ax1.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            
            # Save plot - include overlap info in filename if used
            if overlap:
                valid_overlap = [p for p in overlap if p in param_config]
                overlap_str = "_with_" + "_".join(valid_overlap) if valid_overlap else ""
                plot_name = f"box_{box}_processed_{parameter}{overlap_str}_dual_axis.png"
            else:
                plot_name = f"box_{box}_processed_{parameter}_with_errors.png"
            
            plot_path = os.path.join(plot_dir, plot_name)
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()
            print(f"[Box {box}] Saved: {plot_path}\n")
            
        except Exception as e:
            print(f"Error processing box {box}: {str(e)}")
            
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
                    else:
                        raise ValueError("x_lim dates must be YYYYMMDD or YYYYMMDDHH format")
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
    # Validate time_str
    if not (isinstance(time_str, str) and len(time_str) == 10 and time_str.isdigit()):
        print(f"ERROR: time_str must be a 10-digit string 'YYYYMMDDHH'; got '{time_str}'.")
        return

    # Parse the requested hour as UTC
    try:
        dt_req = datetime.datetime.strptime(time_str, "%Y%m%d%H")
    except ValueError:
        print(f"ERROR: Could not parse '{time_str}' as YYYYMMDDHH.")
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
                f"Motion Frequency Spectrum - Box {box} at {time_chosen_str} UTC"
                #f"\nAll samples of wake #{chosen_wake_iter} (n={len(group)})"
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
                #f"Sample #{sample} of wake #{chosen_wake_iter}"
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
    # Validate time_str
    if not (isinstance(time_str, str) and len(time_str) == 10 and time_str.isdigit()):
        print(f"ERROR: time_str must be a 10-digit string 'YYYYMMDDHH'; got '{time_str}'.")
        return

    # Parse the requested hour as UTC
    try:
        dt_req = datetime.datetime.strptime(time_str, "%Y%m%d%H")
    except ValueError:
        print(f"ERROR: Could not parse '{time_str}' as YYYYMMDDHH.")
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
    # Validate time_str
    if not (isinstance(time_str, str) and len(time_str) == 10 and time_str.isdigit()):
        print(f"ERROR: time_str must be a 10-digit string 'YYYYMMDDHH'; got '{time_str}'.")
        return

    # Parse the requested hour as UTC
    try:
        dt_req = datetime.datetime.strptime(time_str, "%Y%m%d%H")
    except ValueError:
        print(f"ERROR: Could not parse '{time_str}' as YYYYMMDDHH.")
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


from matplotlib.patches import ConnectionPatch

def peek_zoom(parameter, start_date_str, end_date_str, x1, y1, x2, y2, box_number=None, 
              x_lim=False, y_lim=False, title_size=12, overlap=None, overlap_y_lim=False, param_zoom=None):
    """
    Plots processed data with a zoomed inset window showing detail of a selected region.

    Parameters:
        parameter (str): The parameter to plot in the main window. Options: 'temp', 'motion', 'data', 'depth', 
            'f1', 'f2', 'both_f', 'l1', 'l2', 'both_l', 'full', or 'all'.
        start_date_str (str): Start date in YYYYMMDD, YYYYMMDDHH, or YYYYMMDDHHMM format.
        end_date_str (str): End date in YYYYMMDD, YYYYMMDDHH, or YYYYMMDDHHMM format.
        x1 (str): Left x-limit of zoom window in YYYYMMDD, YYYYMMDDHH, or YYYYMMDDHHMM format.
        y1 (float): Bottom y-limit of zoom window.
        x2 (str): Right x-limit of zoom window in YYYYMMDD, YYYYMMDDHH, or YYYYMMDDHHMM format.
        y2 (float): Top y-limit of zoom window.
        box_number (int, optional): Specific box number to plot. If None, plots all boxes.
        x_lim (tuple (str, str), optional): Left and right limit of main plot x axis.
        y_lim (tuple (float, float), optional): Bottom and top limit of main plot y axis.
        overlap (list, optional): List of additional parameters to overlay on main plot.
        overlap_y_lim (tuple (float, float), optional): Bottom and top limit of right y axis.
        param_zoom (str, optional): Parameter to plot in zoom window. If None, uses same as main parameter.
        
    Example use:
        peek_zoom("full", "2025051412", "2025052103", "20250517", 1.98, "20250519", 2.03,
                  overlap=['temp'], overlap_y_lim=(-20,20), param_zoom="both_l", title_size=20)
        peek_zoom("motion", "20250505", "20250525", "20250510", 0.85, "20250515", 0.95, 
                  box_number=8, y_lim=(0.8,1.0))
        peek_zoom("temp", "20250505", "20250525", "20250510", -5, "20250515", 5,
                  overlap=['motion'], overlap_y_lim=(0.8,1.2))
        peek_zoom("full", "20250505", "20250525", "20250510", 10, "20250515", 30,
                  param_zoom="both_f", box_number=8)
    """
    # Parameter configuration (same as peek_pdata)
    param_config = {
        "temp": {"val_col": 1, "err_col": -1, "label": "Temperature", "unit": "°C", "color": "purple"},
        "motion": {"val_col": 2, "err_col": 3, "label": "Computed Length", "unit": "m", "color": "deepskyblue"},
        "data": {"val_col": 4, "err_col": 5, "label": "Microcontroller Analysis Length", "unit": "m", "color": "royalblue"},
        "depth": {"val_col": 6, "err_col": 7, "label": "Depth Sensor Length", "unit": "m", "color": "blue"},
        "f1": {"val_col": 8, "err_col": 9, "label": "Fundamental Frequency 1", "unit": "Hz", "color": "red"},
        "f2": {"val_col": 10, "err_col": 11, "label": "Fundamental Frequency 2", "unit": "Hz", "color": "orange"},
        "l1": {"val_col": 12, "err_col": 13, "label": "Length from F1", "unit": "m", "color": "dodgerblue"},
        "l2": {"val_col": 14, "err_col": 15, "label": "Length from F2", "unit": "m", "color": "fuchsia"}
    }

    # Validate parameter
    valid_parameters = ["temp", "motion", "data", "depth", "f1", "f2", "both_f", "l1", "l2", "both_l", "all", "full"]
    if parameter not in valid_parameters:
        raise ValueError(f"Invalid parameter. Use: {', '.join(valid_parameters)}")
    
    # Set zoom parameter to main parameter if not specified
    if param_zoom is None:
        param_zoom = parameter
    
    # Validate zoom parameter
    if param_zoom not in valid_parameters:
        raise ValueError(f"Invalid param_zoom. Use: {', '.join(valid_parameters)}")

    # Create plots directory
    plot_dir = os.path.join(script_directory, PROCESSED_DATA_PLOTS_FOLDER)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Convert zoom window dates to datetime
    def parse_date_str(date_str):
        if len(date_str) == 8:  # YYYYMMDD
            return datetime.datetime.strptime(date_str, "%Y%m%d")
        elif len(date_str) == 10:  # YYYYMMDDHH
            return datetime.datetime.strptime(date_str, "%Y%m%d%H")
        elif len(date_str) == 12:  # YYYYMMDDHHMM
            return datetime.datetime.strptime(date_str, "%Y%m%d%H%M")
        else:
            raise ValueError("Date must be YYYYMMDD, YYYYMMDDHH, or YYYYMMDDHHMM format")
    
    zoom_x1_dt = parse_date_str(x1)
    zoom_x2_dt = parse_date_str(x2)
    
    # Get Unix timestamp range
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
            
            if processed_data.ndim == 1:
                processed_data = processed_data.reshape(1, -1)

            data_timestamps = processed_data[:, 0].astype(np.int64)
            time_mask = (data_timestamps >= start_int) & (data_timestamps <= end_int)
            filtered_data = processed_data[time_mask]
            
            if len(filtered_data) == 0:
                print(f"No data for box {box} in range")
                continue
                
            # Convert timestamps to datetime
            original_timestamps = [
                datetime.datetime.strptime(str(int(ts)), "%Y%m%d%H%M")
                for ts in filtered_data[:, 0]
            ]
            
            # Create figure with two subplots (zoom on top, main on bottom)
            fig = plt.figure(figsize=(16, 10))
            ax_zoom = plt.subplot2grid((2, 1), (0, 0))
            ax_main = plt.subplot2grid((2, 1), (1, 0), rowspan=2)
            ax2 = None
            
            # Helper function to plot parameter
            def plot_parameter(param_name, timestamps, data, axis, alpha=1.0, show_stats=False):
                if param_name not in param_config:
                    print(f"Warning: Unknown parameter '{param_name}'")
                    return None, None, None
                    
                config = param_config[param_name]
                vals = data[:, config["val_col"]].copy()
                errs = data[:, config["err_col"]].copy() if config["err_col"] != -1 else None
                
                param_timestamps = timestamps.copy()
                
                # Filter for frequency parameters
                if param_name in ["f1", "f2"]:
                    valid_mask = ~(np.isnan(vals) | (vals == 0))
                    vals = vals[valid_mask]
                    if errs is not None:
                        errs = errs[valid_mask]
                    param_timestamps = [param_timestamps[i] for i in range(len(param_timestamps)) if valid_mask[i]]
                
                # Filter large errors for length measurements
                if param_name in ["motion", "data", "depth", "l1", "l2"] and errs is not None:
                    large_err_mask = errs > 0.1
                    vals[large_err_mask] = np.nan
                    errs[large_err_mask] = np.nan
                
                label = f"{config['label']} ({config['unit']})"
                axis.errorbar(param_timestamps, vals, yerr=errs, fmt='o', 
                           color=config['color'], capsize=5, label=label, alpha=alpha)
                
                mean_val = np.nanmean(vals)
                std_val = np.nanstd(vals)
                n_valid = np.sum(~np.isnan(vals))
                err_on_mean = std_val / np.sqrt(n_valid) if n_valid > 0 else np.nan
                
                stats = {
                    'mean': mean_val,
                    'std': std_val,
                    'err_on_mean': err_on_mean,
                    'n_points': n_valid
                }
                
                if show_stats:
                    print(f"[Box {box}] {param_name} - Mean: {mean_val:.4f}, Std: {std_val:.4f}, N: {n_valid}")
                
                return vals, param_timestamps, stats
            
            all_stats = {}
            ax_zoom2 = None  # Initialize second y-axis for zoom
            
            # Plot zoom parameter in zoom window
            if param_zoom in ["all", "full", "both_f", "both_l"]:
                # Multi-parameter plots for zoom
                if param_zoom == "all":
                    params_zoom = ["depth", "motion", "data"]
                    ylabel_zoom = "Length (m)"
                elif param_zoom == "full":
                    params_zoom = ["depth", "motion"]
                    ylabel_zoom = "Length (m)"
                elif param_zoom == "both_f":
                    params_zoom = ["f1", "f2"]
                    ylabel_zoom = "Frequency (Hz)"
                else:  # both_l
                    params_zoom = ["l1", "l2"]
                    ylabel_zoom = "Length (m)"
                
                # Plot in zoom window
                for param in params_zoom:
                    plot_parameter(param, original_timestamps, filtered_data, ax_zoom, alpha=0.7)
            else:
                # Single parameter plot for zoom
                plot_parameter(param_zoom, original_timestamps, filtered_data, ax_zoom, alpha=1.0)
                ylabel_zoom = f"{param_config[param_zoom]['label']} ({param_config[param_zoom]['unit']})"
            
            # Add overlap parameters to zoom window
            if overlap:
                ax_zoom2 = ax_zoom.twinx()
                for overlap_param in overlap:
                    plot_parameter(overlap_param, original_timestamps, filtered_data, ax_zoom2, alpha=0.4)
            
            # Plot main parameter in main window
            if parameter in ["all", "full", "both_f", "both_l"]:
                # Multi-parameter plots
                if parameter == "all":
                    params_to_plot = ["depth", "motion", "data"]
                    ylabel = "Length (m)"
                elif parameter == "full":
                    params_to_plot = ["depth", "motion"]
                    ylabel = "Length (m)"
                elif parameter == "both_f":
                    params_to_plot = ["f1", "f2"]
                    ylabel = "Frequency (Hz)"
                else:  # both_l
                    params_to_plot = ["l1", "l2"]
                    ylabel = "Length (m)"
                
                # Plot in main window
                for param in params_to_plot:
                    vals, timestamps, stats = plot_parameter(param, original_timestamps, filtered_data, ax_main, alpha=0.7)
                    if stats is not None:
                        all_stats[param] = stats
                        
            else:
                # Single parameter plot
                # Plot in main window
                main_vals, main_timestamps, main_stats = plot_parameter(parameter, original_timestamps, filtered_data, ax_main, alpha=1.0, show_stats=True)
                if main_stats is not None:
                    all_stats[parameter] = main_stats
                
                ylabel = f"{param_config[parameter]['label']} ({param_config[parameter]['unit']})"
            
            # Add overlap parameters to main plot only
            if overlap:
                ax2 = ax_main.twinx()
                for overlap_param in overlap:
                    vals, timestamps, stats = plot_parameter(overlap_param, original_timestamps, filtered_data, ax2, alpha=0.4)
                    if stats is not None:
                        all_stats[overlap_param] = stats
            
            # Set zoom window limits
            ax_zoom.set_xlim(zoom_x1_dt, zoom_x2_dt)
            ax_zoom.set_ylim(y1, y2)
            ax_zoom.set_ylabel(ylabel_zoom, size=title_size)
            ax_zoom.grid(True)
            ax_zoom.tick_params(axis='x', rotation=45)
            
            # Set overlap y-limits for zoom if specified
            if overlap_y_lim and ax_zoom2 is not None:
                ax_zoom2.set_ylim(overlap_y_lim)
            
            # Set right y-axis label for zoom window if overlap exists
            if overlap and ax_zoom2 is not None:
                valid_overlap = [p for p in overlap if p in param_config]
                if len(valid_overlap) == 1:
                    overlap_param = valid_overlap[0]
                    ax_zoom2.set_ylabel(f"{param_config[overlap_param]['label']} ({param_config[overlap_param]['unit']})", 
                                       size=title_size-2)
                else:
                    overlap_units = [param_config[p]['unit'] for p in valid_overlap]
                    if len(set(overlap_units)) == 1:
                        ax_zoom2.set_ylabel(f"Value ({overlap_units[0]})", size=title_size-2)
                    else:
                        ax_zoom2.set_ylabel("Overlap Parameters", size=title_size-2)
                ax_zoom2.tick_params(axis='y', labelcolor='black')
            
            # Set title for zoom window
            if param_zoom in ["all", "full", "both_f", "both_l"]:
                # Multi-parameter zoom titles
                if param_zoom == "all":
                    zoom_title_base = "Computed Length, Microcontroller Analysis Length, and Depth Sensor Length"
                elif param_zoom == "full":
                    zoom_title_base = "Computed Length and Depth Sensor Length"
                elif param_zoom == "both_f":
                    zoom_title_base = "Fundamental Frequency 1 and Fundamental Frequency 2"
                else:  # both_l
                    zoom_title_base = "Length from F1 and Length from F2"
            else:
                zoom_title_base = param_config[param_zoom]['label']
            
            if overlap:
                valid_overlap = [p for p in overlap if p in param_config]
                if valid_overlap:
                    overlap_names = " and ".join([param_config[p]['label'] for p in valid_overlap])
                    ax_zoom.set_title(f"Zoomed View: {zoom_title_base} with {overlap_names} - Box {box}", fontsize=title_size)
                else:
                    ax_zoom.set_title(f"Zoomed View: {zoom_title_base} - Box {box}", fontsize=title_size)
            else:
                ax_zoom.set_title(f"Zoomed View: {zoom_title_base} - Box {box}", fontsize=title_size)
            
            # Set main plot limits
            if x_lim:
                x_lim_dt = [parse_date_str(d) for d in x_lim]
                ax_main.set_xlim(x_lim_dt)
            
            if y_lim:
                ax_main.set_ylim(y_lim)
                
            if overlap_y_lim and ax2 is not None:
                ax2.set_ylim(overlap_y_lim)
            
            # Labels for main plot
            ax_main.set_xlabel("UTC Time", size=title_size)
            ax_main.set_ylabel(ylabel if parameter not in ["all", "full", "both_f", "both_l"] else 
                              ("Length (m)" if parameter in ["all", "full", "both_l"] else "Frequency (Hz)"), 
                              size=title_size)
            
            # Set title for main plot
            if parameter in ["all", "full", "both_f", "both_l"]:
                # Multi-parameter main titles (matching peek_pdata style)
                if parameter == "all":
                    if overlap:
                        valid_overlap = [p for p in overlap if p in param_config]
                        if valid_overlap:
                            overlap_names = " and ".join([param_config[p]['label'] for p in valid_overlap])
                            ax_main.set_title(f"Lengths as Measured by BRACHIOSAURUS with {overlap_names} - Box {box}", fontsize=title_size)
                        else:
                            ax_main.set_title(f"Lengths as Measured by BRACHIOSAURUS Through Time - Box {box}", fontsize=title_size)
                    else:
                        ax_main.set_title(f"Lengths as Measured by BRACHIOSAURUS Through Time - Box {box}", fontsize=title_size)
                elif parameter == "full":
                    if overlap:
                        valid_overlap = [p for p in overlap if p in param_config]
                        if valid_overlap:
                            overlap_names = " and ".join([param_config[p]['label'] for p in valid_overlap])
                            ax_main.set_title(f"Computed Length and Depth Sensor Measurements\nwith {overlap_names} - Box {box}", fontsize=title_size)
                        else:
                            ax_main.set_title(f"Motion and Depth Measurements - Box {box}", fontsize=title_size)
                    else:
                        ax_main.set_title(f"Motion and Depth Measurements - Box {box}", fontsize=title_size)
                elif parameter == "both_f":
                    if overlap:
                        valid_overlap = [p for p in overlap if p in param_config]
                        if valid_overlap:
                            overlap_names = " and ".join([param_config[p]['label'] for p in valid_overlap])
                            ax_main.set_title(f"Fundamental Frequencies with {overlap_names} - Box {box}", fontsize=title_size)
                        else:
                            ax_main.set_title(f"Fundamental Frequencies - Box {box}", fontsize=title_size)
                    else:
                        ax_main.set_title(f"Fundamental Frequencies - Box {box}", fontsize=title_size)
                else:  # both_l
                    if overlap:
                        valid_overlap = [p for p in overlap if p in param_config]
                        if valid_overlap:
                            overlap_names = " and ".join([param_config[p]['label'] for p in valid_overlap])
                            ax_main.set_title(f"Lengths from Fundamental Frequencies with {overlap_names} - Box {box}", fontsize=title_size)
                        else:
                            ax_main.set_title(f"Lengths from Fundamental Frequencies - Box {box}", fontsize=title_size)
                    else:
                        ax_main.set_title(f"Lengths from Fundamental Frequencies - Box {box}", fontsize=title_size)
            else:
                # Single parameter titles
                param_label = param_config[parameter]['label']
                if overlap:
                    valid_overlap = [p for p in overlap if p in param_config]
                    if valid_overlap:
                        overlap_names = " and ".join([param_config[p]['label'] for p in valid_overlap])
                        ax_main.set_title(f"Processed {param_label} with {overlap_names} - Box {box}", fontsize=title_size)
                    else:
                        ax_main.set_title(f"Processed {param_label} - Box {box}", fontsize=title_size)
                else:
                    ax_main.set_title(f"Processed {param_label} - Box {box}", fontsize=title_size)
            
            if overlap and ax2 is not None:
                valid_overlap = [p for p in overlap if p in param_config]
                if len(valid_overlap) == 1:
                    overlap_param = valid_overlap[0]
                    ax2.set_ylabel(f"{param_config[overlap_param]['label']} ({param_config[overlap_param]['unit']})", 
                                 size=title_size)
                else:
                    overlap_units = [param_config[p]['unit'] for p in valid_overlap]
                    if len(set(overlap_units)) == 1:
                        ax2.set_ylabel(f"Value ({overlap_units[0]})", size=title_size)
                    else:
                        ax2.set_ylabel("Overlap Parameters", size=title_size)
            
            ax_main.grid(True)
            ax_main.tick_params(axis='x', rotation=45)
            
            # Draw rectangle on main plot showing zoom region AFTER plotting data
            from matplotlib.patches import Rectangle
            rect = Rectangle((zoom_x1_dt, y1), zoom_x2_dt - zoom_x1_dt, y2 - y1,
                           linewidth=2, edgecolor='red', facecolor='none', linestyle='--', zorder=10)
            ax_main.add_patch(rect)
            
            # Add connection patches from top corners of rectangle to bottom corners of zoom plot
            # Get the y-axis limits to position connections at the top of the main plot
            main_ylim = ax_main.get_ylim()
            
            # Connect from top-left corner of rectangle to bottom-left of zoom
            con1 = ConnectionPatch(xyA=(zoom_x1_dt, y2), xyB=(zoom_x1_dt, y1), 
                                  coordsA="data", coordsB="data",
                                  axesA=ax_main, axesB=ax_zoom, 
                                  color="red", linewidth=1.5, linestyle='--', zorder=10, alpha=0.6)
            # Connect from top-right corner of rectangle to bottom-right of zoom
            con2 = ConnectionPatch(xyA=(zoom_x2_dt, y2), xyB=(zoom_x2_dt, y1),
                                  coordsA="data", coordsB="data",
                                  axesA=ax_main, axesB=ax_zoom, 
                                  color="red", linewidth=1.5, linestyle='--', zorder=10, alpha=0.6)
            ax_main.add_artist(con1)
            ax_main.add_artist(con2)
            
            # Legends
            if overlap and ax2 is not None:
                lines1, labels1 = ax_main.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax_main.legend(lines1 + lines2, labels1 + labels2, fontsize=12, loc='upper left')
            else:
                ax_main.legend(fontsize=20, loc='best')
            
            # Zoom window legend
            if overlap and ax_zoom2 is not None:
                lines1_zoom, labels1_zoom = ax_zoom.get_legend_handles_labels()
                lines2_zoom, labels2_zoom = ax_zoom2.get_legend_handles_labels()
                ax_zoom.legend(lines1_zoom + lines2_zoom, labels1_zoom + labels2_zoom, fontsize=10, loc='upper left')
            else:
                ax_zoom.legend(fontsize=20, loc='best')
            
            ax_main.grid(True)
            ax_main.tick_params(axis='x', rotation=45)
            
            # Print statistics
            print(f"\n[Box {box}] Statistics Summary:")
            print("-" * 110)
            for param_name, stats in all_stats.items():
                param_label = param_config[param_name]['label']
                param_unit = param_config[param_name]['unit']
                print(f"{param_label:30s} | Mean: {stats['mean']:8.4f} {param_unit:3s} | Std: {stats['std']:8.4f} {param_unit:3s} | N: {stats['n_points']:4d}")
            print("-" * 110)
            
            # Save plot
            if overlap:
                valid_overlap = [p for p in overlap if p in param_config]
                overlap_str = "_with_" + "_".join(valid_overlap) if valid_overlap else ""
                zoom_str = f"_zoom{param_zoom}" if param_zoom != parameter else ""
                plot_name = f"box_{box}_zoom_{parameter}{zoom_str}{overlap_str}.png"
            else:
                zoom_str = f"_zoom{param_zoom}" if param_zoom != parameter else ""
                plot_name = f"box_{box}_zoom_{parameter}{zoom_str}.png"
            
            plt.tight_layout()
            plot_path = os.path.join(plot_dir, plot_name)
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()
            print(f"[Box {box}] Saved: {plot_path}\n")
            
        except Exception as e:
            print(f"Error processing box {box}: {str(e)}")
    



