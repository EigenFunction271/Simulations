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
lambda_n = 0.013 # Rate constant for the dynamics of light adaptation state n
beta_n = 0.0075  # Rate constant for the decay of light adaptation state n
G = 33.0         # Gain factor for the circadian light stimulus B(t)
b_x = 0.02       # Coefficient for the gating of light input by state variable x
b_xc = 0.05      # Coefficient for the gating of light input by state variable xc
alpha_0_val = 0.0001 # Baseline effective photic input rate
I_0 = 450.0      # Threshold light intensity for photic input saturation (lux)
q_factor = 0.99669 # Period correction factor used in the xc dynamics
stabilisation = 10
# --- Helper Functions ---

def light_schedule(t, light_on_time=8.0, light_off_time=20.0, bright_lux=5000.0, dim_lux=50.0):
    """
    Determines the environmental light intensity L(t) based on a square wave light-dark cycle.
    Args:
        t (float): Current time in hours.
        light_on_time (float): Time of day when bright light turns on (hours from midnight).
        light_off_time (float): Time of day when bright light turns off (hours from midnight).
        bright_lux (float): Light intensity during the bright phase (lux).
        dim_lux (float): Light intensity during the dim phase (lux).
    Returns:
        float: Light intensity L(t) at time t.
    """
    t_in_day = t % 24.0  # Time within a 24-hour cycle
    if light_on_time <= t_in_day < light_off_time:
        return bright_lux
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
        p = 1.0  # Linear response below I_0
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
    k_B_p = params[2]
    lambda_n_p = params[3]
    beta_n_p = params[4]
    G_p = params[5]
    b_x_p = params[6]
    b_xc_p = params[7]
    alpha_0_p = params[8]
    I_0_p = params[9]
    q_factor_p = params[10]

    # 1. Determine current light intensity L(t)
    L_val = light_schedule(t)

    # 2. Calculate effective photic input alpha(L(t)) (Process L component)
    alpha_val = effective_photic_input(L_val, alpha_0_p, I_0_p)

    # 3. Calculate circadian stimulus B(t)
    # This term represents how light information, gated by the SCN's own state (x, xc)
    # and the photoreceptor adaptation state (n), drives the oscillator.
    B_t = G_p * alpha_val * (1 - n) * (1 - b_x_p * x - b_xc_p * xc)

    # 4. SCN Oscillator Equations (Process P)
    # These equations describe a van der Pol-type oscillator, representing the core SCN pacemaker.
    # dx/dt: Dynamics of state variable x. Includes coupling to xc, a van der Pol term for
    #        limit cycle behavior, and the light stimulus B(t).
    dxdt = (np.pi / tau_x0_p) * (xc + mu_p * ((1/3)*x + (4/3)*(x**3)) + B_t)
    
    # dxc/dt: Dynamics of state variable xc. Coupled to x and includes a term to maintain
    #         oscillation amplitude and period, adjusted by q_factor and k_B.
    # The term (24 / (q_factor_p * tau_x0_p)) is a period/amplitude scaling factor.
    x_coeff = -(24 / (q_factor_p * tau_x0_p))**2
    dxc_dt = (np.pi / tau_x0_p) * (k_B_p * xc + x_coeff * x)

    # 5. Light Adaptation Equation (Process L)
    # dn/dt: Dynamics of the photoreceptor adaptation state n.
    #        alpha_val * (1-n) represents light-driven activation.
    #        beta_n * n represents the decay or recovery of adaptation.
    dndt = lambda_n_p * (alpha_val * (1 - n) - beta_n_p * n)

    return [dxdt, dxc_dt, dndt]

# --- Simulation Setup ---
total_days = stabilisation + 2 #15
total_hours = total_days * 24.0
dt = 0.01  # Time step for output (hours)
t_eval = np.arange(0, total_hours, dt) # Time array for simulation

# Initial conditions
x_init = 0.1   # Initial state for x
xc_init = 0.1  # Initial state for xc
n_init = 0.5   # Initial state for light adaptation n (0 to 1)
Y0 = [x_init, xc_init, n_init]

# Pack parameters for odeint
model_params = (tau_x0, mu, k_B, lambda_n, beta_n, G, b_x, b_xc, alpha_0_val, I_0, q_factor)

# --- Perform Simulation ---
print(f"Starting simulation for {total_days} days...")
try:
    # Use rtol and atol parameters for better numerical stability
    solution = odeint(scn_model, Y0, t_eval, 
                     args=model_params,
                     rtol=1e-6, atol=1e-6)
    print("Simulation complete.")
except Exception as e:
    print(f"Error during simulation: {str(e)}")
    sys.exit(1)

# Check for numerical stability
if not np.all(np.isfinite(solution)):
    print("Warning: Numerical instability detected in solution")
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
    
    # 1. SCN Oscillator Phase Space (last 2 days)
    hours_for_phase_plot = total_days * 24
    num_points_phase_plot = int(hours_for_phase_plot / dt)

    plt.figure(figsize=(8, 7)) # Increased height for better aspect ratio with title
    plt.plot(x_sol[-num_points_phase_plot:], xc_sol[-num_points_phase_plot:], color='dodgerblue')
    plt.xlabel('SCN State Variable $x$', fontsize=12)
    plt.ylabel('SCN State Variable $x_c$', fontsize=12)
    plt.title(f'SCN Oscillator Phase Space (Last {hours_for_phase_plot/24:.0f} Days)', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.axis('equal') # Ensures circular limit cycles appear circular
    plt.tight_layout()
    plt.savefig('scn_phase_space.png')
    plt.close()
    
    # 2. SCN Internal Phase (theta) over Time (last 5 days)
    hours_for_theta_plot = total_days * 24
    num_points_theta_plot = int(hours_for_theta_plot / dt)

    # Time axis in days for this plot
    time_in_days_for_theta_plot = t_eval[-num_points_theta_plot:] / 24.0

    # Create a figure with two subplots for wrapped and unwrapped phase
    plt.figure(figsize=(12, 10))
    
    # 2a. Wrapped Phase Plot (top)
    plt.subplot(2, 1, 1)
    plt.plot(time_in_days_for_theta_plot, theta_t_raw[-num_points_theta_plot:], color='red')
    plt.xlabel('Time (days)', fontsize=12)
    plt.ylabel('SCN Internal Phase $\\theta(t)$ (radians)', fontsize=12)
    plt.title('Wrapped SCN Internal Phase (Last 5 Days)', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.axhline(y=np.pi, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=-np.pi, color='gray', linestyle='--', alpha=0.5)
    plt.ylim(-np.pi-0.1, np.pi+0.1)
    
    # 2b. Unwrapped Phase Plot (bottom)
    plt.subplot(2, 1, 2)
    plt.plot(time_in_days_for_theta_plot, theta_t_unwrapped[-num_points_theta_plot:], color='green')
    plt.xlabel('Time (days)', fontsize=12)
    plt.ylabel('Unwrapped SCN Internal Phase $\\theta(t)$ (radians)', fontsize=12)
    plt.title(f'SCN Internal Phase $\\theta(t)$ over Time (Last {hours_for_theta_plot/24:.0f} Days)', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('scn_phase_comparison.png')
    plt.close()
    
    # 3. Optional: Plot Light Schedule (last 5 days for context with phase plot)
    L_values_over_time = np.array([light_schedule(t_val) for t_val in t_eval])

    plt.figure(figsize=(12, 4))
    plt.plot(time_in_days_for_theta_plot, L_values_over_time[-num_points_theta_plot:], color='orange')
    # Full schedule: plt.plot(t_eval / 24, L_values_over_time, color='orange')
    plt.xlabel('Time (days)', fontsize=12)
    plt.ylabel('Light Intensity $L(t)$ (lux)', fontsize=12)
    plt.title(f'Environmental Light Schedule (Last {hours_for_theta_plot/24:.0f} Days)', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.yscale('log') # Light intensity varies greatly, log scale can be useful
    plt.ylim(bottom=1) # Avoid log(0) issues, ensure dim light is visible
    plt.tight_layout()
    plt.savefig('scn_light_schedule.png')
    plt.close()
    
    print("\nPlots have been saved as:")
    print("1. scn_phase_space.png")
    print("2. scn_phase_comparison.png (shows both wrapped and unwrapped phase)")
    print("3. scn_light_schedule.png")
    
except Exception as e:
    print(f"Error during plotting: {str(e)}")
    sys.exit(1)

print("\nScript finished. Check the saved plot files.")