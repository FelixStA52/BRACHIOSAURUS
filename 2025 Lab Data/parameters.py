""" Parameters file for BRACHIOSAURUS data processing
second_peak: True or False, also consider the second peak of the frequency spectrum
dirr: "directory/to/data/and/motion/files"
processing: if you want the data and motion files to be processed (True or False)
"""
import numpy as np
IGNORE_RUNTIME_WARNINGS = True
PROCESSING = True
SECOND_PEAK = True
DIR = "/Users/felix/Documents/McGill_Bsc/Radio Lab/singing poles/20250725_calib"

# physical constants of the pole for theoretical approximations
YOUNG = 69e9#7.35025312e+10#69e9 # young modulus of the pole in GPa
R_O = 1.277/100 # outer radius of the pole in m
R_I = 1.021/100 # inner radius of the pole in m
DENSITY = 2700 # density of the pole in kg/m^3 #3071
MASS = 0.677#0.638#0.677 # mass of the BRACHIOSAURUS in kg

TOL = 0.05 # tolerence for checking between peaks
L_TOL = 2*TOL # larger tolerence for checking between peaks
MIN_FREQ = 2.0 # discard below this frequency to avoid DC (Hz)

# Local and global noise thresholds
LOCAL_NOISE_THRESH = 10
GLOBAL_NOISE_THRESH = 5

ARRAY_SIZE = 2048 # size of the sampling arrays

SAMP_FREQ = 200 # sampling frequency

POLE_MAX_L = 5 # maximum length of the pole
POLE_MIN_L = 0.1 # minimum length of the pole

# Tip mass dimensions (meters)
TIP_LENGTH_X = 0.14  # Along beam direction
TIP_HEIGHT_Z = 0.04   # In bending direction
TIP_WIDTH_Y = 0.12    # Lateral dimension

# Eccentricity from pole centerline to box center of mass
ECCENTRICITY = R_O + TIP_HEIGHT_Z/2  # outer radius + half box height

# Moment of inertia about box's own center (for bending about y-axis)  
I_box_center = MASS * (TIP_LENGTH_X**2 + TIP_HEIGHT_Z**2) / 12

# Total moment of inertia about pole centerline (parallel axis theorem)
I_T = I_box_center + MASS * ECCENTRICITY**2

# Add to parameters.py
CALIB_BOX = {
    52: 1.995,
    53: 1.93,
    54: 1.86,
    55: 1.795,
    56: 1.725,
    57: 1.675,
    58: 1.59,
    59: 1.525,
    60: 1.47,
    62: 1.35,
    63: 1.28,
    64: 1.195,
    65: 1.12,
    66: 1.02,
    67: 0.93,
    68: 0.85,
    69: 1.065,
    70: 0.97,
}

# Temperature (ºC) v. Stiffness (GPa)
STIFF_V_TEMP = np.array([[-73, 72.4e9],
                         [21, 68.9e9],
                         [93, 66.2e9]])

# Guess for the temperature at a given time (in case no temperature was measured)
JAN_TEMP = -30
FEB_TEMP = -30
MAR_TEMP = -30
APR_TEMP = -25
MAY_TEMP = -15
JUN_TEMP = -5
JUL_TEMP = 0
AUG_TEMP = -5
SEP_TEMP = -10
OCT_TEMP = -20
NOV_TEMP = -25
DEC_TEMP = -25
TEMP_GUESS = [JAN_TEMP,FEB_TEMP,MAR_TEMP,APR_TEMP,MAY_TEMP,JUN_TEMP,JUL_TEMP,AUG_TEMP,SEP_TEMP,OCT_TEMP,NOV_TEMP,DEC_TEMP]

tEI = 182.64026108 # the 3EI factor found from calibrating the pole
c = 0 # the damping factor as found from calibrating the pole

UNIFIED_FOLDER = "unified_data"
PROCESSED_FOLDER = "processed_data"
RAW_DATA_PLOTS_FOLDER = "raw_data_plots"
PROCESSED_DATA_PLOTS_FOLDER = "processed_data_plots"
MOTION_AMP_FOLDER = "motion_amplitude_plots"
SPECTRUM_FOLDER = "motion_spectrum_plots"
ACCELERATION_FOLDER = "motion_acceleration_plots"











































