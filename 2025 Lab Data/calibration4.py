import numpy as np
import matplotlib.pyplot as plt
from parameters import *
import os
import scipy.optimize as sp
from freq_estimator import *

# Get Young's modulus at reference temperature (20°C)
E_T0 = youngs_modulus(20)  # Keep existing implementation

# Prepare data structures
all_lengths = []
all_f1_meas = []
all_f1_err = []
all_f1_theory = []

# Process each calibrated box
for box, true_length in CALIB_BOX.items():
    print(box)
    # Load processed data
    data_file = os.path.join(PROCESSED_FOLDER, f'processed_data_{box}.csv')
    try:
        data = np.genfromtxt(data_file, delimiter=',', skip_header=1)
        if data.ndim == 1:
            data = data.reshape(1, -1)
    except OSError:
        print(f"Warning: Missing data file for box {box} - {data_file}")
        continue

    # Extract relevant columns (F1 only)
    f1_meas = data[:, 8]        # Measured F1
    f1_err = data[:, 9]         # F1 error
    f1_meas = np.array(f1_meas)
    f1_err = np.array(f1_err)
    
    f1_err = f1_err[f1_meas>0]
    f1_meas = f1_meas[f1_meas>0]

    # Calculate theoretical frequency at true length using exact frequencies
    f1_th, _ = exact_frequencies(true_length, E_T0, R_O, R_I, DENSITY, MASS, I_T)
    
    # Store results
    all_lengths.extend([true_length] * len(f1_meas))
    all_f1_meas.extend(f1_meas)
    all_f1_err.extend(f1_err)
    all_f1_theory.extend([f1_th] * len(f1_meas))

# Generate theoretical curve using exact frequencies
lengths_curve = np.linspace(POLE_MIN_L, POLE_MAX_L, 100)
f1_curve = []
for L in lengths_curve:
    f1_exact, _ = exact_frequencies(L, YOUNG, R_O, R_I, DENSITY, MASS, I_T)
    f1_curve.append(f1_exact)

# Create plots with residuals
fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 12), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

def fit_func(L, a, b):
    """Wrapper function with fixed E, ρ, and tip_mass using exact frequencies"""
    E_fixed = E_T0
    rho_fixed = DENSITY
    m_tip_fixed = MASS
    r_o_fixed = R_O
    r_i_fixed = R_I
    # Handle both scalar and array inputs
    if np.isscalar(L):
        f1, _ = exact_frequencies(L, E_T0, r_o_fixed, r_i_fixed, rho_fixed, m_tip_fixed, I_T)
        return f1 - np.exp(a*L+b)
    else:
        result = np.zeros_like(L)
        for i, L_val in enumerate(L):
            f1, _ = exact_frequencies(L_val, E, r_o_fixed, r_i_fixed, rho_fixed, m_tip_fixed, I_T)
            result[i] = f1
        return result - np.exp(a*L+b)

def exact_freq_wrapper(L, E, r_o, r_i, rho, m_tip=MASS):
    """Wrapper for exact_frequencies to match curve_fit signature"""
    # Handle both scalar and array inputs
    if np.isscalar(L):
        f1, _ = exact_frequencies(L, E, r_o, r_i, rho, m_tip, I_T)
        return f1
    else:
        result = np.zeros_like(L)
        for i, L_val in enumerate(L):
            f1, _ = exact_frequencies(L_val, E, r_o, r_i, rho, m_tip, I_T)
            result[i] = f1
        return result

all_lengths = np.array(all_lengths)
all_f1_meas = np.array(all_f1_meas)
mask = np.isfinite(all_f1_meas)
all_lengths = all_lengths[mask]
all_f1_meas = all_f1_meas[mask]

param2, cov2 = sp.curve_fit(fit_func, all_lengths, all_f1_meas, p0=[1,1], maxfev = 1000000)

print("PARAM2",param2)

param, cov = sp.curve_fit(exact_freq_wrapper, all_lengths, all_f1_meas, p0=[69e9, R_O, R_I, DENSITY], maxfev = 1000000)

print("PARAM",param)

ax0.plot(lengths_curve, fit_func(lengths_curve, *param2), '-', color="#1E88E5", alpha=0.7,linewidth=4, label='Physical Prediction + Exponential Correction')
#ax0.plot(lengths_curve, exact_freq_wrapper(lengths_curve, *param), 'r:', linewidth=2, label='Fit')

ax0.tick_params(axis='both', which='major', labelsize=15)

# Convert to arrays for vector operations
all_lengths = np.array(all_lengths)
all_f1_meas = np.array(all_f1_meas)
all_f1_err = np.array(all_f1_err)
all_f1_theory = np.array(all_f1_theory)

# ========================================================================
# Group measurements by actual length and compute weighted averages
# ========================================================================
from collections import defaultdict

def compute_grouped_results(lengths_actual, lengths_measured, errors_measured):
    """Helper function to group and average results for any model"""
    length_groups = defaultdict(list)
    for i in range(len(lengths_actual)):
        length_groups[lengths_actual[i]].append({
            'measured_length': lengths_measured[i],
            'length_error': errors_measured[i]
        })
    unique_lengths = []
    avg_measured_lengths = []
    avg_length_errors = []
    for actual_length, measurements in length_groups.items():
        n_meas = len(measurements)
        meas_lengths = np.array([m['measured_length'] for m in measurements])
        length_errors = np.array([m['length_error'] for m in measurements])
        # Weighted average of measured lengths
        weights = 1.0 / (length_errors**2 + 1e-12)
        weights = weights / np.sum(weights)
        avg_meas_length = np.sum(weights * meas_lengths)
        avg_length_err = 1.0 / np.sqrt(np.sum(1.0 / (length_errors**2 + 1e-12)))
        unique_lengths.append(actual_length)
        avg_measured_lengths.append(avg_meas_length)
        avg_length_errors.append(avg_length_err)
    return (
        np.array(unique_lengths),
        np.array(avg_measured_lengths),
        np.array(avg_length_errors)
    )

# First, group and average the results
unique_lengths, avg_f1_meas, avg_f1_err = compute_grouped_results(
    all_lengths,      # actual lengths
    all_f1_meas,      # measured F1 values
    all_f1_err        # F1 errors
)

# Calculate residuals
#residuals = all_f1_meas - fit_func(all_lengths, *param2)
#residuals = all_f1_meas - exact_freq_wrapper(all_lengths, *param)
residuals_exact = avg_f1_meas - [exact_frequencies(L, E_T0, R_O, R_I, DENSITY, MASS, I_T)[0] for L in unique_lengths]
residuals_fit = avg_f1_meas - [fit_func(L, *param2) for L in unique_lengths]


avg_f1_err = np.array(avg_f1_err)
avg_f1_err[avg_f1_err<0.05] = 0.05

all_f1_err = np.array(all_f1_err)
all_f1_err = all_f1_err[mask]
all_f1_err[all_f1_err<0.05] = 0.05

# Then plot the averaged results
ax0.errorbar(
    unique_lengths, avg_f1_meas, yerr=avg_f1_err,
    fmt='o', color='k', markersize=8, capsize=8, capthick=1, label='Measured Data'
)

# Main plot: Frequency vs Length
#ax0.set_title('First Frequency Mode as a Function of Pole Length', fontsize=24)
ax0.plot(lengths_curve, f1_curve, '-', color="#D81B60", alpha=0.7, linewidth=4, label='Physical Prediction')
ax0.set_ylabel('Frequency (Hz)', fontsize=20)
ax0.legend(fontsize=15)
ax0.grid(True)
ax0.set_ylim(0, 15)

# Residual plot
ax1.errorbar(
    unique_lengths, residuals_exact, yerr=avg_f1_err,
    fmt='o', color='#D81B60', capsize=8, alpha=0.7, markersize=6, label='Physical Prediction Residuals'
)
ax1.errorbar(
    unique_lengths, residuals_fit, yerr=avg_f1_err,
    fmt='o', color='#1E88E5', capsize=8, alpha=0.7, markersize=6, label='Physical Prediction + Exponential Correction Residuals'
)
ax1.axhline(0, color='black', linestyle='--')
ax1.set_xlabel('Length (m)', fontsize=20)
ax1.set_ylabel('Residuals (Hz)', fontsize=20)
ax1.legend(fontsize=15)
ax1.grid(True)
#ax1.set_ylim(-0.5, 0.5)
ax1.set_ylim(-1, 1)

# Set common x-axis limits
x_min = max(POLE_MIN_L, np.min(all_lengths) - 0.1)
x_max = min(POLE_MAX_L, np.max(all_lengths) + 0.1)
ax0.set_xlim(x_min, x_max)
ax1.tick_params(axis='both', which='major', labelsize=15)

# Add parameter info to plot
param_text = (
    f"Parameters:\n"
    f"$R_o$ = {R_O*100:.2f} cm\n"
    f"$R_i$ = {R_I*100:.2f} cm\n"
    f"$E$ = {E_T0/1e9:.0f} GPa\n"
    f"$\\rho$ = {DENSITY} kg/m³\n"
    f"Tip mass = {MASS} kg"
)
#plt.figtext(0.15, 0.01, param_text, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)  # Make space for parameter box
plt.savefig('calibration_results.png', dpi=300)
plt.show()








# Function to compute theoretical length from measured frequency with error estimate
def theoretical_length_from_freq_with_error(f_measured, f_error, initial_guess):
    """Find length that gives the measured frequency using theoretical parameters
    Returns: (length, length_error)"""
    
    def freq_diff(L):
        f_theoretical, _ = exact_frequencies(L, E_T0, R_O, R_I, DENSITY, MASS, I_T)
        return f_theoretical - f_measured
    
    # Find the central solution
    solution = sp.fsolve(freq_diff, x0=initial_guess)
    L_central = solution[0]
    
    # Compute numerical derivative df/dL at the solution
    dL = 1e-6  # Small perturbation
    f_plus, _ = exact_frequencies(L_central + dL, E_T0, R_O, R_I, DENSITY, MASS, I_T)
    f_minus, _ = exact_frequencies(L_central - dL, E_T0, R_O, R_I, DENSITY, MASS, I_T)
    df_dL = (f_plus - f_minus) / (2 * dL)
    
    # Error propagation: if f = f(L), then σ_L = σ_f / |df/dL|
    L_error = f_error / abs(df_dL) if abs(df_dL) > 1e-12 else np.inf
    
    return L_central, L_error

# ========================================================================
# Function to compute theoretical length from measured frequency with error estimate
# FOR CORRECTED MODEL (with exponential factor)
# ========================================================================
def corrected_length_from_freq_with_error(f_measured, f_error, initial_guess, a, b):
    """Find length using corrected model (exact frequencies minus exponential term)"""
    def corrected_freq_diff(L):
        f_exact, _ = exact_frequencies(L, E_T0, R_O, R_I, DENSITY, MASS, I_T)
        corrected_f = f_exact - np.exp(a*L+b)
        return corrected_f - f_measured
    
    # Find the central solution
    solution = sp.fsolve(corrected_freq_diff, x0=initial_guess)
    L_central = solution[0]
    
    # Compute numerical derivative df/dL at the solution
    dL = 1e-6  # Small perturbation
    f_plus_exact, _ = exact_frequencies(L_central + dL, E_T0, R_O, R_I, DENSITY, MASS, I_T)
    f_minus_exact, _ = exact_frequencies(L_central - dL, E_T0, R_O, R_I, DENSITY, MASS, I_T)
    f_plus = f_plus_exact - np.exp(a*(L_central+dL)+b)
    f_minus = f_minus_exact - np.exp(a*(L_central-dL)+b)
    df_dL = (f_plus - f_minus) / (2 * dL)
    
    # Error propagation: σ_L = σ_f / |df/dL|
    L_error = f_error / abs(df_dL) if abs(df_dL) > 1e-12 else np.inf
    
    return L_central, L_error

# ========================================================================
# Compute theoretical lengths using both models
# ========================================================================
# unique_lengths, avg_f1_meas, yerr=avg_f1_err
#all_f1_meas = avg_f1_meas
#all_f1_err = avg_f1_err
#all_lengths = unique_lengths

# Original exact model
results_exact = [theoretical_length_from_freq_with_error(all_f1_meas[i], all_f1_err[i], all_lengths[i]) 
                 for i in range(len(all_lengths))]
len1_exact = np.array([result[0] for result in results_exact])
len1_err_exact = np.array([result[1] for result in results_exact])

# New corrected model (using param2 from curve_fit)
results_corrected = [corrected_length_from_freq_with_error(all_f1_meas[i], all_f1_err[i], all_lengths[i], param2[0], param2[1]) 
                     for i in range(len(all_lengths))]
len1_corrected = np.array([result[0] for result in results_corrected])
len1_err_corrected = np.array([result[1] for result in results_corrected])

# Process both models
(unique_lengths_exact, avg_measured_exact, avg_errors_exact) = compute_grouped_results(
    all_lengths, len1_exact, len1_err_exact
)

(unique_lengths_corrected, avg_measured_corrected, avg_errors_corrected) = compute_grouped_results(
    all_lengths, len1_corrected, len1_err_corrected
)

# ========================================================================
# Plot both sets of results
# ========================================================================
plt.figure(figsize=(12, 8))

# Convert differences to cm
exact_diff_cm = (avg_measured_exact - unique_lengths_exact) * 100
exact_diff_err_cm = avg_errors_exact * 100

corrected_diff_cm = (avg_measured_corrected - unique_lengths_corrected) * 100
corrected_diff_err_cm = avg_errors_corrected * 100

# Plot exact model results (blue)
plt.errorbar(unique_lengths_exact, exact_diff_cm, yerr=exact_diff_err_cm, 
             fmt='o', alpha=0.8, capsize=5, markersize=6, 
             color='#D81B60', label='Physical Model')

# Plot corrected model results (green)
plt.errorbar(unique_lengths_corrected, corrected_diff_cm, yerr=corrected_diff_err_cm, 
             fmt='s', alpha=0.8, capsize=5, markersize=6, 
             color='#1E88E5', label='Physical + Correction')

print(f"Max err = {np.max(corrected_diff_err_cm)}")

# Formatting
plt.tick_params(axis='both', which='major', labelsize=15)
plt.xlabel('Measured Length (m)', size=20)
plt.ylabel('Length Difference (cm)', size=20)
#plt.title('Difference between Computed and Measured Lengths', size=24)
plt.grid()
plt.ylim(-5, 5)
plt.xlim(0.8, 2.1)
plt.axhline(0, color='k', linestyle='--', alpha=0.7)
plt.legend(fontsize=20)

# Add statistics for both models
#stats_exact = f"Physical Model:\nMean: {np.mean(exact_diff_cm):.2f} ± {np.std(exact_diff_cm):.2f} cm\nRMS: {np.sqrt(np.mean(exact_diff_cm**2)):.2f} cm"
#stats_corrected = f"Physical+Correction:\nMean: {np.mean(corrected_diff_cm):.2f} ± {np.std(corrected_diff_cm):.2f} cm\nRMS: {np.sqrt(np.mean(corrected_diff_cm**2)):.2f} cm"

#plt.figtext(0.15, 0.02, stats_exact + "\n\n" + stats_corrected,
           #bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Calculate RMS and MAE errors for lengths above and below 1.4m threshold
threshold = 1.2

# For exact model
mask_below_exact = unique_lengths_exact < threshold
mask_above_exact = unique_lengths_exact >= threshold

rms_below_exact = np.sqrt(np.mean(exact_diff_cm[mask_below_exact]**2))
rms_above_exact = np.sqrt(np.mean(exact_diff_cm[mask_above_exact]**2))
mae_below_exact = np.mean(np.abs(exact_diff_cm[mask_below_exact]))
mae_above_exact = np.mean(np.abs(exact_diff_cm[mask_above_exact]))

# For corrected model
mask_below_corrected = unique_lengths_corrected < threshold
mask_above_corrected = unique_lengths_corrected >= threshold

rms_below_corrected = np.sqrt(np.mean(corrected_diff_cm[mask_below_corrected]**2))
rms_above_corrected = np.sqrt(np.mean(corrected_diff_cm[mask_above_corrected]**2))
mae_below_corrected = np.mean(np.abs(corrected_diff_cm[mask_below_corrected]))
mae_above_corrected = np.mean(np.abs(corrected_diff_cm[mask_above_corrected]))

# Print results
print("\n" + "="*70)
print("Performance Metrics for Length Measurements")
print("="*70)
print(f"\nPhysical Prediction Model:")
print(f"  L < {threshold} m:  RMSE = {rms_below_exact:.3f} cm  |  MAE = {mae_below_exact:.3f} cm")
print(f"  L >= {threshold} m: RMSE = {rms_above_exact:.3f} cm  |  MAE = {mae_above_exact:.3f} cm")
print(f"\nPhysical Prediction + Correction Model:")
print(f"  L < {threshold} m:  RMSE = {rms_below_corrected:.3f} cm  |  MAE = {mae_below_corrected:.3f} cm")
print(f"  L >= {threshold} m: RMSE = {rms_above_corrected:.3f} cm  |  MAE = {mae_above_corrected:.3f} cm")
print("="*70 + "\n")



plt.tight_layout()
plt.show()



























