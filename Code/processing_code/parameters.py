""" Parameters file for BRACHIOSAURUS data processing
second_peak: True or False, also consider the second peak of the frequency spectrum
dirr: "directory/to/data/and/motion/files"
processing: if you want the data and motion files to be processed (True or False)
"""
import numpy as np
IGNORE_RUNTIME_WARNINGS = True
PROCESSING = True
SECOND_PEAK = True
DIR = "/Users/felix/Documents/McGill_Bsc/Radio Lab/singing poles/Glacier stake data"

# physical constants of the pole for theoretical approximations
YOUNG = 69e9 # young modulus of the pole in GPa
R_O = 0.015875# outer radius of the pole in m
R_I = 0.0128# inner radius of the pole in m
DENSITY = 2700 # density of the pole in kg/m^3
MASS = 0.677 # mass of the BRACHIOSAURUS in kg

TOL = 0.05 # tolerence for checking between peaks
L_TOL = 2*TOL # larger tolerence for checking between peaks
MIN_FREQ = 0.5 # discard below this frequency to avoid DC (Hz)

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

# Local and global noise thresholds
LOCAL_NOISE_THRESH = 4
GLOBAL_NOISE_THRESH = 3

ARRAY_SIZE = 2048 # size of the sampling arrays

SAMP_FREQ = 200 # sampling frequency

CALIB_BOX = {
    16: 2,
    17: 3,
    18: 0.99
}

ORDER_LENGTH = 5
ORDER_TEMP = 5

POLE_MAX_L = 5 # maximum length of the pole
POLE_MIN_L = 0.1 # minimum length of the pole
POLE_POINTS = 300

TEMP_MIN = -60
TEMP_MAX = 30
TEMP_POINTS = 30

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











































