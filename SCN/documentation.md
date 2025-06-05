# SCN Circadian Pacemaker Simulation

## Overview
This project implements a numerical simulation of the human Suprachiasmatic Nucleus (SCN) circadian pacemaker, based on the Kronauer/Jewett, Forger, Kronauer (JFK) 1999 model. The SCN is the master circadian clock in mammals, responsible for regulating various biological rhythms.

## Model Description

### State Variables
- **x**: Primary SCN oscillator state variable
- **xc**: Secondary SCN oscillator state variable
- **n**: Light adaptation state (photoreceptor activation history)

### Key Equations

1. **SCN Oscillator (Process P)**:
   ```
   dx/dt = (π/τx0) * [xc + μ(x/3 - 4x³/3) + B(t)]
   dxc/dt = (π/τx0) * [-x + (24/(0.99669*τx0) + kB)*xc]
   ```

2. **Light Adaptation (Process L)**:
   ```
   dn/dt = λn * [α(L(t))(1-n) - βn*n]
   ```

3. **Circadian Stimulus**:
   ```
   B(t) = G * α(L(t)) * (1-n) * (1 - bx*x - bxc*xc)
   ```

4. **SCN Internal Phase**:
   ```
   θ(t) = atan2(xc(t), x(t))
   ```

### Parameters
- τx0 = 24.2 hours (intrinsic period)
- μ = 0.13 (nonlinearity parameter)
- kB = 0.0035 (damping parameter)
- λn = 0.013 (light adaptation rate)
- βn = 0.0075 (light adaptation decay)
- G = 33.0 (light sensitivity)
- bx = 0.02 (x feedback)
- bxc = 0.05 (xc feedback)
- α0 = 0.0001 (baseline light sensitivity)
- I0 = 450.0 lux (light intensity threshold)

## Implementation Details

### Numerical Integration
- Using `scipy.integrate.odeint` for solving the coupled ODEs
- Time step: 0.1 hours
- Simulation duration: 15 days
- Initial conditions: x = 0.1, xc = 0.1, n = 0.5

### Light Schedule
- 24-hour square wave cycle
- Bright light (5000 lux): 08:00-22:00
- Dim light (50 lux): 22:00-08:00

### Visualization
1. **Phase Space Plot**
   - x vs xc for last 2 days
   - Shows limit cycle behavior
   - Includes start/end markers

2. **Internal Phase Plot**
   - θ(t) vs time for last 5 days
   - Shows entrained rhythm
   - Phase unwrapped for continuous display

3. **Light Schedule Plot**
   - Light intensity vs time
   - Reference for entrainment

## Current Status

### Implemented Features
- Basic model structure
- ODE system implementation
- Light schedule function
- Phase calculation and unwrapping
- Basic plotting functionality

### Known Issues
1. Numerical instability in state variables
2. Incorrect phase behavior
3. Time axis scaling problems
4. Plot visibility and clarity issues

### Next Steps
1. Fix numerical stability issues
2. Implement proper error handling
3. Improve plotting clarity
4. Add comprehensive documentation
5. Add parameter validation
6. Implement adaptive time stepping if needed

## References
- Kronauer, R. E., Forger, D. B., & Jewett, M. E. (1999). Quantifying human circadian pacemaker response to brief, extended, and repeated light stimuli over the phototopic range. Journal of Biological Rhythms, 14(6), 500-515. 