import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import warnings
import sys

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning)

# --- Model Parameters (Typical Human-Derived Values) ---
# These parameters are based on the Kronauer/Jewett, Forger, Kronauer (JFK) 1999 model.
tau_x0 = 24.2    # Intrinsic period of the SCN oscillator (hours)
mu = 0.13        # Stiffness parameter of the van der Pol oscillator, affecting limit cycle shape
k_B = 0.0035     # Parameter influencing the amplitude and stability of xc
lambda_n = 60 #0.013 # Rate constant for the dynamics of light adaptation state n
beta_n = 0.0075  # Rate constant for the decay of light adaptation state n
G = 33.0         # Gain factor for the circadian light stimulus B(t)
b_x = 0.02       # Coefficient for the gating of light input by state variable x
b_xc = 0.05      # Coefficient for the gating of light input by state variable xc
alpha_0_val = 0.05 # Baseline effective photic input rate
I_0 = 9500.0      # Threshold light intensity for photic input saturation (lux)
q_factor = 0.99669 # Period correction factor used in the xc dynamics
q = 1.0          # Fixed q value
stabilisation = 10
# --- Helper Functions ---

def format_time_label(x, pos):
    """
    Format time label to show time of day in HH:MM format.
    Args:
        x (float): Time in days
        pos: Position parameter (required by matplotlib formatter)
    Returns:
        str: Formatted time string
    """
    total_hours = x * 24  # Convert days to hours
    hours = int(total_hours % 24)  # Get hours within day
    minutes = int((total_hours * 60) % 60)  # Get minutes
    return f"{hours:02d}:{minutes:02d}"

def light_schedule(t, light_on_time=7.0, light_dark_time=9.5, light_off_time=20.0, bright_lux=25000.0, mid_lux = 10000, dim_lux=50.0):
    """
    Determines the environmental light intensity L(t) based on a square wave light-dark cycle.
    Args:
        t (float): Current time in hours.
        light_on_time (float): Time of day when bright light turns on (hours from midnight).
        light_dark_time (float): Time of day in cubicle (hours from midnight).
        light_off_time (float): Time of day when bright light turns off (hours from midnight).
        bright_lux (float): Light intensity during the bright phase (lux).
        dim_lux (float): Light intensity during the dim phase (lux).
    Returns:
        float: Light intensity L(t) at time t.
    """
    t_in_day = t % 24.0  # Time within a 24-hour cycle
    if light_on_time <= t_in_day < light_dark_time:
        return bright_lux
    elif light_dark_time <= t_in_day < light_off_time:
        return mid_lux
    else:
        return dim_lux

def effective_photic_input(L_val, alpha_0_param, I_0_param):
    """
    Calculates the effective photic input rate alpha(L(t)).
    This function translates environmental light intensity L(t) into an effective rate
    that drives the light adaptation process and the circadian stimulus.
    Args:
        L_val (float): Current light intensity L(t) (lux).
        alpha_0_param (float): Baseline effective photic input rate.
        I_0_param (float): Threshold light intensity for saturation.
    Returns:
        float: Effective photic input rate alpha(L(t)).
    """
    if L_val <= 0:
        return 0.0  # No photic input if light is zero or negative
    
    ratio = L_val / I_0_param
    if L_val <= I_0_param:
        #p = 1.0  # Linear response below I_0
        p = 0.5
    else:
        p = 0.45 # Compressive response above I_0 (saturation effect)
    return alpha_0_param * (ratio ** p)

# --- ODE System Definition (SCN Model) ---

def scn_model(Y, t, *params):
    # ---- START DEBUG PRINT ----
    if 'debug_printed' not in scn_model.__dict__: # Print only once or a few times
        print("--- Running scn_model with (Y, t, *params) signature ---")
        print(f"Type of Y: {type(Y)}, Type of t: {type(t)}")
        if isinstance(Y, np.ndarray):
            print(f"Shape of Y: {Y.shape}")
        print(f"Value of t at first call (or early call): {t}")
        scn_model.debug_printed = True 
    # ---- END DEBUG PRINT ----
    
    """
    Defines the system of Ordinary Differential Equations (ODEs) for the SCN model.
    Args:
        t (float): Current time (hours).
        Y (list or np.array): Array of current state variable values [x, xc, n].
        t (float): Current time (hours).
        params (tuple): Tuple of model parameters.
    Returns:
        list: List of derivatives [dx/dt, dxc/dt, dn/dt].
    """
    x, xc, n = Y
    
    # Unpack parameters
    tau_x0_p = params[0]
    mu_p = params[1]
    k_B_p = params[2]  # This is k_B from the global parameters
    lambda_n_p = params[3]
    beta_n_p = params[4]
    G_p = params[5]
    b_x_p = params[6]
    b_xc_p = params[7]
    alpha_0_p = params[8]
    I_0_p = params[9]
    q_factor_p = params[10]

    # Debug print parameters (only once)
    if not hasattr(scn_model, 'params_printed'):
        print("\nModel Parameters:")
        print(f"tau_x0: {tau_x0_p}")
        print(f"mu: {mu_p}")
        print(f"k_B: {k_B_p}")
        print(f"lambda_n: {lambda_n_p}")
        print(f"beta_n: {beta_n_p}")
        print(f"G: {G_p}")
        print(f"b_x: {b_x_p}")
        print(f"b_xc: {b_xc_p}")
        print(f"alpha_0: {alpha_0_p}")
        print(f"I_0: {I_0_p}")
        print(f"q_factor: {q_factor_p}")
        scn_model.params_printed = True

    # 1. Determine current light intensity L(t)
    L_val = light_schedule(t)

    # 2. Calculate effective photic input alpha(L(t)) (Process L component)
    alpha_val = effective_photic_input(L_val, alpha_0_p, I_0_p)

    # 3. Calculate circadian stimulus B(t)
    # This term represents how light information, gated by the SCN's own state (x, xc)
    # and the photoreceptor adaptation state (n), drives the oscillator.
    B_t = G_p * alpha_val * (1 - n) * (1 - b_x_p * x - b_xc_p * xc)

    # Debug print for key variables (only print occasionally to avoid spam)
    if t % 24 < 0.01:  # Print roughly once per day
        print(f"\nDebug at t={t:.2f}:")
        print(f"L_val: {L_val:.2f}")
        print(f"alpha_val: {alpha_val:.6f}")
        print(f"B_t: {B_t:.6f}")
        print(f"x: {x:.6f}, xc: {xc:.6f}, n: {n:.6f}")

    # 4. SCN Oscillator Equations (Process P)
    # Modified van der Pol term with proper bounds
    x_clipped = np.clip(x, -3, 3)  # Clip x to reasonable range
    vdp_term = mu_p * (x_clipped/3 + (4/3)*np.clip(x_clipped**3, -3, 3))
    dxdt = (np.pi / tau_x0_p) * (xc + vdp_term + B_t)
    
    # dxc/dt: Dynamics of state variable xc
    # Modified x_coeff calculation with better damping
    base_coeff = -(24 / (q_factor_p * tau_x0_p))**2
    x_coeff = base_coeff + k_B_p * B_t
    dxc_dt = (np.pi / tau_x0_p) * (q*B_t * xc + x_coeff * x)

    # 5. Light Adaptation Equation (Process L)
    # dn/dt: Dynamics of the photoreceptor adaptation state n.
    #        alpha_val * (1-n) represents light-driven activation.
    #        beta_n * n represents the decay or recovery of adaptation.
    dndt = lambda_n_p * (alpha_val * (1 - n) - beta_n_p * n)

    return [dxdt, dxc_dt, dndt]

# --- Simulation Setup ---
total_days = stabilisation + 3 #15
total_hours = total_days * 24.0
dt = 0.1  # Increased time step for better stability
t_eval = np.arange(0, total_hours, dt)

# Initial conditions - increased to help with stability
x_init = 0.5    # Increased from 0.1
xc_init = 0.5   # Increased from 0.1
n_init = 0.5    # Kept the same
Y0 = [x_init, xc_init, n_init]

# Pack parameters for odeint
model_params = (tau_x0, mu, k_B, lambda_n, beta_n, G, b_x, b_xc, alpha_0_val, I_0, q_factor)

# --- Perform Simulation ---
print(f"Starting simulation for {total_days} days...")
try:
    # Print initial derivatives to check if they're reasonable
    initial_derivatives = scn_model(Y0, 0, *model_params)
    print("\nInitial derivatives at t=0:")
    print(f"dx/dt: {initial_derivatives[0]:.6f}")
    print(f"dxc/dt: {initial_derivatives[1]:.6f}")
    print(f"dn/dt: {initial_derivatives[2]:.6f}")
    
    # Use LSODA solver with more conservative parameters
    from scipy.integrate import solve_ivp
    
    def scn_model_wrapper(t, Y):
        return scn_model(Y, t, *model_params)
    
    solution = solve_ivp(scn_model_wrapper, 
                        t_span=(0, total_hours),
                        y0=Y0,
                        t_eval=t_eval,
                        method='LSODA',
                        rtol=1e-4,
                        atol=1e-4)
    
    if not solution.success:
        print(f"\nSolver failed: {solution.message}")
        sys.exit(1)
        
    solution = solution.y.T  # Transpose to match odeint format
    
    print("Simulation complete.")
except Exception as e:
    print(f"Error during simulation: {str(e)}")
    sys.exit(1)

# Check for numerical stability
if not np.all(np.isfinite(solution)):
    print("Warning: Numerical instability detected in solution")
    # Find where the solution becomes non-finite
    bad_indices = np.where(~np.isfinite(solution))[0]
    if len(bad_indices) > 0:
        print(f"First occurrence of non-finite value at t={t_eval[bad_indices[0]]:.2f}")
    sys.exit(1)

# Check if solution is stuck at zero
if np.allclose(solution, 0, atol=1e-10):
    print("Warning: Solution appears to be stuck at zero")
    print("This might indicate a problem with the initial conditions or parameters")
    sys.exit(1)

x_sol = solution[:, 0]
xc_sol = solution[:, 1]
n_sol = solution[:, 2] # n is not directly plotted but is part of the system

# --- Derive SCN Internal Phase (theta) ---
# theta(t) = atan2(xc(t), x(t))
# atan2 returns phase in radians between -pi and +pi.
# np.unwrap handles jumps greater than pi to make the phase continuous.
theta_t_raw = np.arctan2(xc_sol, x_sol)
theta_t_unwrapped = np.unwrap(theta_t_raw)

# --- Generate Plots ---
try:
    # Set matplotlib to use a non-interactive backend
    plt.switch_backend('Agg')
    
    # Calculate indices for post-stabilization period
    stabilization_hours = stabilisation * 24
    simulation_hours = (total_days - stabilisation) * 24
    post_stab_indices = (t_eval >= stabilization_hours) & (t_eval < total_days * 24)
    
    # 1. SCN Oscillator Phase Space (simulation period only)
    plt.figure(figsize=(8, 7))
    plt.plot(x_sol[post_stab_indices], xc_sol[post_stab_indices], color='dodgerblue')
    plt.xlabel('SCN State Variable $x$', fontsize=12)
    plt.ylabel('SCN State Variable $x_c$', fontsize=12)
    plt.title(f'SCN Oscillator Phase Space (After {stabilisation} Days Stabilization)', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('scn_phase_space.png')
    plt.close()
    
    # 2. SCN Internal Phase (theta) over Time (simulation period only)
    plt.figure(figsize=(12, 10))
    
    # 2a. Wrapped Phase Plot (top)
    plt.subplot(2, 1, 1)
    plt.plot(t_eval[post_stab_indices] / 24.0, theta_t_raw[post_stab_indices], color='red')
    plt.xlabel('Time (days)', fontsize=12)
    plt.ylabel('SCN Internal Phase $\\theta(t)$ (radians)', fontsize=12)
    plt.title(f'Wrapped SCN Internal Phase (After {stabilisation} Days Stabilization)', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.axhline(y=np.pi, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=-np.pi, color='gray', linestyle='--', alpha=0.5)
    plt.ylim(-np.pi-0.1, np.pi+0.1)
    
    # Add secondary x-axis with 24-hour clock format
    ax2 = plt.gca().twiny()
    ax2.set_xlim(plt.gca().get_xlim())
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(format_time_label))
    ax2.set_xlabel('Time of Day (HH:MM)', fontsize=12)
    ax2.xaxis.set_major_locator(plt.MultipleLocator(1/24))  # Show every hour
    
    # 2b. Unwrapped Phase Plot (bottom)
    plt.subplot(2, 1, 2)
    plt.plot(t_eval[post_stab_indices] / 24.0, theta_t_unwrapped[post_stab_indices], color='green')
    plt.xlabel('Time (days)', fontsize=12)
    plt.ylabel('Unwrapped SCN Internal Phase $\\theta(t)$ (radians)', fontsize=12)
    plt.title(f'Unwrapped SCN Internal Phase (After {stabilisation} Days Stabilization)', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # Add secondary x-axis with 24-hour clock format
    ax2 = plt.gca().twiny()
    ax2.set_xlim(plt.gca().get_xlim())
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(format_time_label))
    ax2.set_xlabel('Time of Day (HH:MM)', fontsize=12)
    ax2.xaxis.set_major_locator(plt.MultipleLocator(1/24))  # Show every hour
    
    plt.tight_layout()
    plt.savefig('scn_phase_comparison.png')
    plt.close()
    
    # 3. Optional: Plot Light Schedule (simulation period only)
    L_values_over_time = np.array([light_schedule(t_val) for t_val in t_eval])

    plt.figure(figsize=(12, 4))
    plt.plot(t_eval[post_stab_indices] / 24.0, L_values_over_time[post_stab_indices], color='orange')
    plt.xlabel('Time (days)', fontsize=12)
    plt.ylabel('Light Intensity $L(t)$ (lux)', fontsize=12)
    plt.title(f'Environmental Light Schedule (After {stabilisation} Days Stabilization)', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.yscale('log')
    plt.ylim(bottom=1)
    
    # Add secondary x-axis with 24-hour clock format
    ax2 = plt.gca().twiny()
    ax2.set_xlim(plt.gca().get_xlim())
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(format_time_label))
    ax2.set_xlabel('Time of Day (HH:MM)', fontsize=12)
    ax2.xaxis.set_major_locator(plt.MultipleLocator(1/24))  # Show every hour
    
    plt.tight_layout()
    plt.savefig('scn_light_schedule.png')
    plt.close()
    
    # 4. Plot xc over time
    plt.figure(figsize=(12, 4))
    plt.plot(t_eval[post_stab_indices] / 24.0, xc_sol[post_stab_indices], color='purple')
    plt.xlabel('Time (days)', fontsize=12)
    plt.ylabel('SCN State Variable $x_c$', fontsize=12)
    plt.title(f'SCN State Variable $x_c$ Over Time (After {stabilisation} Days Stabilization)', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # Add secondary x-axis with 24-hour clock format
    ax2 = plt.gca().twiny()
    ax2.set_xlim(plt.gca().get_xlim())
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(format_time_label))
    ax2.set_xlabel('Time of Day (HH:MM)', fontsize=12)
    ax2.xaxis.set_major_locator(plt.MultipleLocator(1/24))  # Show every hour
    
    plt.tight_layout()
    plt.savefig('scn_xc_time.png')
    plt.close()
    
    print("\nPlots have been saved as:")
    print("1. scn_phase_space.png")
    print("2. scn_phase_comparison.png (shows both wrapped and unwrapped phase)")
    print("3. scn_light_schedule.png")
    print("4. scn_xc_time.png")
    print(f"\nNote: All plots show {total_days - stabilisation} days of simulation data after {stabilisation} days of stabilization")
    print("      All time-based plots include 24-hour clock format on top axis")
    
except Exception as e:
    print(f"Error during plotting: {str(e)}")
    sys.exit(1)

print("\nScript finished. Check the saved plot files.")

# Add debug printing for key variables
print("\nDebug Values (showing every 24 hours after stabilization):")
print("Time (hrs) | x_sol | xc_sol | theta_raw")
print("-" * 50)

# Calculate indices for post-stabilization period
stabilization_hours = stabilisation * 24
post_stab_indices = (t_eval >= stabilization_hours) & (t_eval < total_days * 24)

# Print values every 24 hours
for i in range(len(t_eval)):
    if post_stab_indices[i] and i % 2400 == 0:  # 2400 = 24 hours * 100 (since dt = 0.01)
        print(f"{t_eval[i]:8.2f} | {x_sol[i]:6.3f} | {xc_sol[i]:6.3f} | {theta_t_raw[i]:6.3f}")

# Print final values
print("\nFinal Values:")
print(f"Final x_sol: {x_sol[-1]:.6f}")
print(f"Final xc_sol: {xc_sol[-1]:.6f}")
print(f"Final theta_raw: {theta_t_raw[-1]:.6f}")