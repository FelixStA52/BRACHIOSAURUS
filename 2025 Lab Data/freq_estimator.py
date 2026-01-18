##PDE + Physics
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.optimize import root
from scipy.optimize import root_scalar
import csv
from parameters import *

# Given properties (radii in meters)
r_outer = R_O  # Outer radius (1.277 cm)
r_inner = R_I  # Inner radius (1.021 cm)
E = YOUNG           # Young's modulus in Pa
rho = DENSITY         # Density in kg/m³
m_tip = MASS      # Tip mass in kg

# Cross-sectional properties
A = np.pi * (r_outer**2 - r_inner**2)  # Area in m²
I = np.pi * (r_outer**4 - r_inner**4) / 4  # Second moment of area in m⁴

# Length range (1.0 to 4.0 meters)
L_min = POLE_MIN_L
L_max = POLE_MAX_L
num_points = 4900
L_values = np.linspace(L_min, L_max, num_points)

# Initialize arrays for frequencies
f1_analytical = np.zeros(num_points)

def estimate_mode2_frequency(L, E, I_val, m, M_t):
    """
    Estimate second mode frequency for cases where iterative method fails.
    Uses a simple approximation based on beam without tip mass, then adjusts.
    """
    # Second mode for cantilever without tip mass
    lam2_no_tip = 4.694
    f2_no_tip = (lam2_no_tip**2 * np.sqrt(E * I_val / (m * L**4))) / (2 * np.pi)
    
    # Crude adjustment for tip mass effect (reduces frequency)
    mass_ratio = M_t / (m * L)
    reduction_factor = 1 / (1 + 0.1 * mass_ratio)  # Empirical - adjust as needed
    
    #return f2_no_tip * reduction_factor
    return np.nan

def solve_mode(char_eq, mode, mass_ratio):
    """
    Find root for specified vibration mode using robust bracketing method.
    
    Parameters:
        char_eq: Characteristic equation function
        mode: Mode number (1 or 2)
        mass_ratio: Mass ratio μ = M_eff / m_beam
    
    Returns:
        lam: Eigenvalue λ
    """
    from scipy.optimize import root_scalar, root
    
    if mode == 1:
        # Mode 1: eigenvalue typically between 0.5 and 4.0
        brackets = [
            [0.1, 2.5],   # Light tip mass
            [0.01, 1.5],  # Heavy tip mass
            [0.001, 1.0]  # Very heavy tip mass
        ]
        
        for i, bracket in enumerate(brackets):
            try:
                sol = root_scalar(char_eq, bracket=bracket, method='brentq')
                return sol.root
            except ValueError:
                #if i == len(brackets) - 1:  # Last bracket failed
                    #print(f"FALLBACK: Mode {mode} bracketing methods failed, trying Newton-Raphson")
                continue
        
        # Fallback to approximation + Newton
        initial_guess = max(0.1, (3/(mass_ratio + 0.24))**0.25)
        try:
            sol = root(char_eq, initial_guess, method='lm')
            if sol.success and sol.x[0] > 0:
                #print(f"FALLBACK: Mode {mode} using Newton-Raphson approximation")
                return sol.x[0]
        except:
            pass
            
    elif mode == 2:
        # Mode 2: eigenvalue typically between 3.5 and 8.0
        brackets = [
            [3.5, 7.0],   # Most common range
            [3.0, 8.0],   # Wider range
            [2.5, 10.0]   # Very wide range
        ]
        
        for i, bracket in enumerate(brackets):
            try:
                sol = root_scalar(char_eq, bracket=bracket, method='brentq')
                return sol.root
            except ValueError:
                #if i == len(brackets) - 1:  # Last bracket failed
                    #print(f"FALLBACK: Mode {mode} bracketing methods failed, trying Newton-Raphson")
                continue
        
        # Fallback to approximation + Newton
        initial_guess = max(4.0, 4.694 / (1 + 0.05*mass_ratio))
        try:
            sol = root(char_eq, initial_guess, method='lm')
            if sol.success and sol.x[0] > 3.0:  # Mode 2 should be > 3
                #print(f"FALLBACK: Mode {mode} using Newton-Raphson approximation")
                return sol.x[0]
        except:
            pass
    
    raise ValueError(f"FALLBACK FAILED: Could not find eigenvalue for mode {mode} with mass ratio {mass_ratio:.3f}")

def exact_frequencies(L, E, r_o=R_O, r_i=R_I, p=DENSITY, M_t=MASS, I_t=I_T):
    """
    Calculate exact natural frequencies (Hz) for both modes using iterative effective mass.
    
    Parameters:
        L: Beam length (m)
        E: Young's modulus (Pa)
        r_o: Outer radius (m)
        r_i: Inner radius (m)
        p: Density (kg/m³)
        M_t: Tip mass (kg)
        I_t: Tip rotary inertia (kg·m²)
    
    Returns:
        f1, f2: First and second natural frequencies (Hz)
    """
    # Ensure L is scalar
    if not np.isscalar(L):
        L_scalar = float(L)
    else:
        L_scalar = L
    
    # Cross-sectional properties
    I_val = np.pi * (r_o**4 - r_i**4) / 4  # Second moment of area (m⁴)
    m = p * np.pi * (r_o**2 - r_i**2)      # Mass per unit length (kg/m)
    
    # Check if configuration supports second mode
    mass_ratio = M_t / (m * L_scalar)
    
    # For very heavy tip masses relative to beam mass, mode 2 may not exist
    # This is a heuristic - you may need to adjust the threshold
    mode2_exists = mass_ratio < 50.0# and L_scalar > 1.18  # Adjust these thresholds as needed
    
    #if not mode2_exists:
        #print(f"FALLBACK AT LENGTH {L_scalar}: Mode 2 doesn't exist (mass_ratio={mass_ratio:.1f}), using approximation")
    
    # Calculate mode 1 (always exists)
    try:
        M_eff_1, lam1, _ = iterative_Meff(L_scalar, E, r_o, r_i, p, M_t, I_t, mode=1)
        f1 = (lam1**2 * np.sqrt(E * I_val / (m * L_scalar**4))) / (2 * np.pi)
    except:
        #print(f"FALLBACK AT LENGTH {L_scalar}: Error calculating mode 1, using Rayleigh approximation")
        M_eff_1 = M_t
        f1 = (1/(2*np.pi)) * np.sqrt(3*E*I_val/(L_scalar**3*(M_eff_1 + 0.24*m*L_scalar)))
    
    # Calculate mode 2 (only if it should exist)
    if mode2_exists:
        try:
            M_eff_2, lam2, phi2 = iterative_Meff(L_scalar, E, r_o, r_i, p, M_t, I_t, mode=2)
            
            # Additional check: verify that phi(L) is not near zero
            if abs(phi2(L_scalar)) < 1e-6:
                #print(f"FALLBACK AT LENGTH {L_scalar}: Mode 2 has node at tip, using approximation")
                f2 = estimate_mode2_frequency(L_scalar, E, I_val, m, M_t)
            else:
                f2 = (lam2**2 * np.sqrt(E * I_val / (m * L_scalar**4))) / (2 * np.pi)
        except:
            #print(f"FALLBACK AT LENGTH {L_scalar}: Error calculating mode 2, using approximation")
            f2 = estimate_mode2_frequency(L_scalar, E, I_val, m, M_t)
    else:
        # Mode 2 doesn't exist for this configuration - use approximation
        f2 = estimate_mode2_frequency(L_scalar, E, I_val, m, M_t)
    
    return f1, f2

# Initialize arrays for exact solutions
f1_exact = np.zeros(num_points)
f2_exact = np.zeros(num_points)


# =====================================================
# 1. Analytical solution for F1 (verified formula)
# =====================================================
def compute_Meff(L, d, M_tip, phi):
    """
    Compute the equivalent point mass via integration over the tip mass distribution.
    Meff = ∫_{L-d}^L φ(x)² (M_tip/d) dx
    
    Parameters:
        L: Beam length (m)
        d: Length of tip mass distribution (m)
        M_tip: Total tip mass (kg)
        phi: Mode shape function (normalized so φ(L) = 1)
    
    Returns:
        Meff: Effective tip mass (kg)
    """
    # Integration points over the tip mass region
    n_points = 200
    x_start = max(0, L - d)  # Don't go below x=0
    xs = np.linspace(x_start, L, n_points)
    
    # Integrand: φ(x)² × (mass density)
    phi_vals = phi(xs)
    integrand = phi_vals**2 * (M_tip / d)
    
    # Numerical integration
    M_eff = np.trapz(integrand, xs)
    
    return M_eff

def iterative_Meff(L, E, r_o, r_i, rho, M_tip, I_t=0, d=0.14, tol=1e-4, maxiter=25, mode=1):
    """
    Iteratively refine Meff for the specified mode:
      1. Start with Meff = M_tip
      2. Solve for φ(x) with that Meff
      3. Compute new Meff from φ(x) over the tip region
      4. Repeat until convergence
    
    Parameters:
        L: Beam length (m) - must be scalar
        E: Young's modulus (Pa)
        r_o: Outer radius (m)
        r_i: Inner radius (m)
        rho: Density (kg/m³)
        M_tip: Physical tip mass (kg)
        I_t: Tip rotary inertia (kg·m²)
        d: Length of tip mass distribution (m)
        tol: Convergence tolerance
        maxiter: Maximum iterations
        mode: Mode number (1 or 2)
    
    Returns:
        M_eff: Converged effective mass (kg)
        lam: Final eigenvalue
        phi: Final mode shape function
    """
    # Ensure L is scalar
    if not np.isscalar(L):
        L_scalar = float(L)
    else:
        L_scalar = L
    
    # Geometric & material constants
    I_beam = np.pi * (r_o**4 - r_i**4) / 4
    m_beam = rho * (np.pi * (r_o**2 - r_i**2)) * L_scalar

    # Initial guess - start with physical tip mass
    M_eff = M_tip
    
    # Store convergence history for debugging
    M_eff_history = [M_eff]

    for iteration in range(1, maxiter + 1):
        try:
            # Solve for mode shape and eigenvalue with current M_eff
            lam, phi = solve_mode_shape(L_scalar, E, I_beam, m_beam, M_eff, I_t, mode)
            
            # Compute updated M_eff from the mode shape over tip region
            M_eff_new = compute_Meff(L_scalar, d, M_tip, phi)
            
            # Store history
            M_eff_history.append(M_eff_new)
            
            # Check for convergence
            relative_change = abs(M_eff_new - M_eff) / max(abs(M_eff), abs(M_eff_new), 1e-10)
            
            if relative_change < tol:
                #print(f"Mode {mode} converged in {iteration} iterations: M_eff = {M_eff_new:.4f} kg")
                return M_eff_new, lam, phi
            
            # Check for oscillation (simple detection)
            if iteration > 3:
                recent_values = M_eff_history[-4:]
                if abs(recent_values[-1] - recent_values[-3]) < tol and \
                   abs(recent_values[-2] - recent_values[-4]) < tol:
                    # Oscillating - take average
                    M_eff_avg = (M_eff + M_eff_new) / 2
                    #print(f"FALLBACK AT LENGTH {L_scalar}: Mode {mode} oscillating, using average: {M_eff_avg:.4f} kg")
                    return M_eff_avg, lam, phi
            
            # Update for next iteration with damping for stability
            if iteration > 10:
                damping = 0.7
                #print(f"FALLBACK AT LENGTH {L_scalar}: Mode {mode} using convergence damping (factor={damping})")
            else:
                damping = 1.0
            M_eff = M_eff + damping * (M_eff_new - M_eff)
            
        except Exception as e:
            if "node at tip" in str(e):
                # Mode doesn't exist for this configuration
                raise ValueError(f"FALLBACK AT LENGTH {L_scalar}: Mode {mode} not physical for this configuration")
            else:
                #print(f"FALLBACK AT LENGTH {L_scalar}: Error in iteration {iteration} for mode {mode}: {e}")
                # Fall back to physical tip mass
                return M_tip, None, None

    #print(f"FALLBACK AT LENGTH {L_scalar}: Mode {mode} did not converge after {maxiter} iterations")
    #print(f"Final M_eff = {M_eff:.4f} kg, last change = {abs(M_eff_new - M_eff):.6f}")
    return M_eff, lam, phi


def solve_mode_shape(L, E, I_beam, m_beam, M_eff, I_t=0, mode=1):
    """
    Solve for the specified mode shape and eigenvalue of a cantilever beam with tip mass.
    Uses the complete formulation from equations (C.16), (C.19), and (C.20).
    
    Parameters:
        L: Beam length (m)
        E: Young's modulus (Pa)
        I_beam: Second moment of area (m⁴)
        m_beam: Total beam mass (kg)
        M_eff: Effective tip mass (kg)
        I_t: Tip rotary inertia (kg·m²)
        mode: Mode number (1 or 2)
    
    Returns:
        lam: Eigenvalue λ
        phi: Mode shape function normalized so φ(L) = 1
    """
    # Ensure L is scalar
    if not np.isscalar(L):
        raise ValueError("L must be scalar in solve_mode_shape")
    
    # Mass per unit length
    m = m_beam / L  # kg/m
    
    def char_eq(lam):
        """Characteristic equation (C.18)"""
        cosl = np.cos(lam)
        sinl = np.sin(lam)
        coshl = np.cosh(lam)
        sinhl = np.sinh(lam)
        
        term1 = 1 + cosl * coshl
        term2 = lam * (M_eff/(m*L)) * (cosl*sinhl - sinl*coshl)
        term3 = - (lam**3 * I_t/(m*L**3)) * (coshl*sinl + sinhl*cosl)
        term4 = (lam**4 * M_eff * I_t/(m**2 * L**4)) * (1 - cosl*coshl)
        
        return term1 + term2 + term3 + term4

    # Solve characteristic equation for the specified mode λ
    try:
        lam = solve_mode(char_eq, mode, M_eff/m_beam)
    except ValueError:
        raise ValueError(f"Could not find eigenvalue for mode {mode}")

    # Calculate trigonometric/hyperbolic values at λ
    cos_lam = np.cos(lam)
    sin_lam = np.sin(lam)
    cosh_lam = np.cosh(lam)
    sinh_lam = np.sinh(lam)
    
    # Calculate ζᵣ using equation (C.20)
    numerator = sin_lam - sinh_lam + lam * (M_eff/(m*L)) * (cos_lam - cosh_lam)
    denominator = cos_lam + cosh_lam - lam * (M_eff/(m*L)) * (sin_lam - sinh_lam)
    
    if abs(denominator) < 1e-8:
        # Handle degenerate case
        zeta_r = 0
        #print(f"FALLBACK AT LENGTH {L}: ζᵣ denominator near zero for mode {mode}, setting ζᵣ=0")
    else:
        zeta_r = numerator / denominator

    # Define mode shape using equation (C.19)
    def phi_raw(x):
        """Unnormalized mode shape φᵣ(x)"""
        # Ensure x is array-like for vectorized operations
        x = np.atleast_1d(x)
        scaled_x = x * lam / L
        
        cos_term = np.cos(scaled_x) - np.cosh(scaled_x)
        sin_term = np.sin(scaled_x) - np.sinh(scaled_x)
        
        result = cos_term + zeta_r * sin_term
        
        # Return scalar if input was scalar
        if result.size == 1:
            return float(result[0])
        return result

    # Calculate normalization factor (value at x=L)
    phi_L_val = phi_raw(L)
    """
    if mode == 2:
        x_debug = np.linspace(0.5*L, L, 200)  # Skip x=0
        phi_debug = phi_raw(x_debug)
        node_idx = np.argmin(np.abs(phi_debug))
        node_location = x_debug[node_idx]
        print(f"L={L}, node at x={node_location:.3f}m ({node_location/L*100:.1f}%), φ(L)={phi_L_val:.6f}")#"""
        
    # Handle potential division by zero
    if abs(phi_L_val) < 1e-8:
        # This indicates the mode shape has a node at the tip
        # This is non-physical for a cantilever with tip mass
        raise ValueError(f"FALLBACK AT LENGTH {L}: Mode {mode} has node at tip (φ(L) ≈ 0) - mode may not exist for this configuration")
    else:
        def phi(x):
            return phi_raw(x) / phi_L_val

    return lam, phi


def second_moment(ro, ri):
    return np.pi / 4 * (ro**4 - ri**4)

def freq_estimator(L, E=E, r_o=r_outer, r_i=r_inner, p=rho, m=m_tip):
    # Handle both scalar and array inputs
    if np.isscalar(L):
        # Single value case
        m_pole = np.pi*(r_o**2 - r_i**2)*L*p
        M_eff, _, _ = iterative_Meff(L, E, r_o, r_i, p, m, d=TIP_LENGTH_X)
        return (1/(2*np.pi))*np.sqrt(3*E*second_moment(r_o, r_i)/(L**3*(M_eff + 0.24*m_pole)))
    else:
        # Array case - process each element individually
        result = np.zeros_like(L)
        for i, L_val in enumerate(L):
            m_pole = np.pi*(r_o**2 - r_i**2)*L_val*p
            M_eff, _, _ = iterative_Meff(L_val, E, r_o, r_i, p, m, d=TIP_LENGTH_X)
            result[i] = (1/(2*np.pi))*np.sqrt(3*E*second_moment(r_o, r_i)/(L_val**3*(M_eff + 0.24*m_pole)))
        return result


def make_f_file():
    # Compute exact frequencies
    for i, L in enumerate(L_values):
        f1_exact[i], f2_exact[i] = exact_frequencies(L, E, r_outer, r_inner, rho, m_tip, I_T)
        #print(f1_exact[i], f2_exact[i])

    # Compute analytical
    for i, L in enumerate(L_values):
        f1_analytical[i] = freq_estimator(L)

    #"""
    # open once, write header
    csvfile = open('freq_vs_length.csv', 'w', newline='')
    writer = csv.writer(csvfile)
    writer.writerow(['L', 'F1_analytical', 'F1_exact', 'F2_exact'])

    for idx in range(len(L_values)):
        # write this row to the CSV
        writer.writerow([L_values[idx], f1_analytical[idx], f1_exact[idx], f2_exact[idx]])

    # close csv file
    csvfile.close()
    print("Wrote freq_vs_length.csv\n")#"""


def youngs_modulus(T):
    """
    Return the Young's modulus (in the same units as STIFF_V_TEMP[:,1])
    at temperature T by linear interpolation/extrapolation of STIFF_V_TEMP.
    
    STIFF_V_TEMP should be an (N,2) array: [[T0, E0], [T1, E1], …].
    """
    # Extract and sort by temperature
    temps = STIFF_V_TEMP[:, 0]
    mods  = STIFF_V_TEMP[:, 1]
    order = np.argsort(temps)
    temps_sorted = temps[order]
    mods_sorted  = mods[order]
    
    # Choose the two points for interpolation/extrapolation
    if T <= temps_sorted[0]:
        T1, E1 = temps_sorted[0], mods_sorted[0]
        T2, E2 = temps_sorted[1], mods_sorted[1]
    elif T >= temps_sorted[-1]:
        T1, E1 = temps_sorted[-2], mods_sorted[-2]
        T2, E2 = temps_sorted[-1], mods_sorted[-1]
    else:
        # Find interval [T1, T2] containing T
        idx = np.searchsorted(temps_sorted, T) - 1
        T1, E1 = temps_sorted[idx],   mods_sorted[idx]
        T2, E2 = temps_sorted[idx+1], mods_sorted[idx+1]
    
    # Linear formula: E = E1 + slope * (T - T1)
    slope = (E2 - E1) / (T2 - T1)
    return E1 + slope * (T - T1)


if __name__ == "__main__":
    make_f_file()
    # =====================================================
    # Plot results
    # =====================================================
    plt.figure(figsize=(14, 8))

    # Plot analytical F1
    plt.plot(L_values, f1_analytical, '-', color="deepskyblue",linewidth=5, label='F1 Analytical')
    plt.plot(L_values, f1_exact, ':', color="blue",linewidth=5, label='F1 Exact')
    plt.plot(L_values, f2_exact, '-', color="royalblue",linewidth=5, label='F2 Exact')

    #plt.plot(L_values, f1_analytical*4, 'k:', linewidth=2, label='Trouble line')

    plt.xlabel('Beam Length (m)', fontsize=35)
    plt.ylabel('Natural Frequency (Hz)', fontsize=35)
    plt.title('Natural Frequencies vs. Beam Length', fontsize=35)
    plt.grid(True)
    plt.legend(fontsize=20)
    plt.xlim(L_min, L_max)
    plt.ylim(0, 50)  # Adjusted for frequency range
    plt.tight_layout()

    # Add text box with parameters
    param_text = (
        f"Parameters:\n"
        f"Outer Radius = {r_outer*100:.2f} cm\n"
        f"Inner Radius = {r_inner*100:.2f} cm\n"
        f"E = {E/1e9:.0f} GPa\n"
        f"ρ = {rho} kg/m³\n"
        f"Tip Mass = {m_tip} kg"
    )
    #plt.annotate(param_text, xy=(0.02, 0.02), xycoords='axes fraction', 
                 #bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                 #fontsize=20)

    #plt.savefig('cantilever_frequencies.png', dpi=300)
    plt.show()
