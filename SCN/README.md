# SCN Circadian Pacemaker Simulation

This folder contains a Python implementation of the human Suprachiasmatic Nucleus (SCN) circadian pacemaker model, based on the Kronauer/Jewett/Forger (KJF) 1999 model.

## Features
- Simulates the core SCN oscillator (x, xc) and light adaptation (n)
- Implements a homeostatic sleep drive (S) and process S-P
- Flexible light schedule with configurable day/night cycles
- Plots for phase space, internal phase, light schedule, xc, S, SP, and single-day overlays
- Debug output for key variables and model parameters

## Key Files
- `KJF.py`: Main simulation script
- `plot_single_day.py`: Plots S, xc, and SP for the last simulated day with 24-hour time axis

## Usage
1. **Run the main simulation:**
   ```bash
   python KJF.py
   ```
   This will generate simulation results and save plots in the current directory.

2. **Plot single-day overlays:**
   ```bash
   python plot_single_day.py
   ```
   This will generate `S_day.png`, `xc_day.png`, and `SP_day.png` for the last simulated day.

## Key Parameters
- `tau_x0`: Intrinsic period of the SCN oscillator (hours)
- `mu`: Nonlinearity parameter (affects amplitude)
- `k_B`: Damping parameter
- `G`: Light sensitivity/gain
- `alpha_0_val`: Baseline effective photic input rate (per hour)
- `I_0`: Light intensity threshold (lux)
- `r_w`, `r_s`, `S_max`: Homeostatic sleep drive parameters

## Light Schedule
- Configurable for bright, mid, and dim phases
- Defined in hours from midnight, repeats every 24 hours

## Output
- Plots are saved as PNG files:
  - `scn_phase_space.png`, `scn_phase_comparison.png`, `scn_light_schedule.png`
  - `scn_xc_time.png`, `scn_sp_time.png`, `scn_S_time.png`
  - `S_day.png`, `xc_day.png`, `SP_day.png`

## References
- Kronauer, R. E., Forger, D. B., & Jewett, M. E. (1999). Quantifying human circadian pacemaker response to brief, extended, and repeated light stimuli over the phototopic range. *Journal of Biological Rhythms*, 14(6), 500-515.

## Notes
- All time units are in hours unless otherwise specified.
- The model is sensitive to parameter choices; see comments in `KJF.py` for tuning advice. 