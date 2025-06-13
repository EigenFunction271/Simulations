import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Assume these are imported or available from the main simulation
from KJF import S_sol, xc_sol, SP_sol, t_eval, stabilisation, total_days, format_time_label

# Find indices for the last simulated day (after stabilization)
last_day_start = (total_days - 1) * 24
last_day_end = total_days * 24
indices = (t_eval >= last_day_start) & (t_eval < last_day_end)

# Time of day in hours (0-24)
time_of_day = (t_eval[indices] % 24)

# Formatter for 24-hour time (e.g., 1800 for 6:00 PM)
def hour_formatter(x, pos):
    hour = int(x)
    minute = int((x - hour) * 60)
    return f"{hour:02d}{minute:02d}"
formatter = FuncFormatter(hour_formatter)

# Plot S(t) for the last day
plt.figure(figsize=(10, 4))
plt.plot(time_of_day, S_sol[indices], color='darkorange')
plt.xlabel('Time of Day (HHMM)', fontsize=12)
plt.ylabel('S(t)', fontsize=12)
plt.title('S(t) Over the Last Simulated Day')
plt.grid(True, linestyle=':', alpha=0.7)
plt.xlim(0, 24)
plt.gca().xaxis.set_major_formatter(formatter)
plt.xticks(np.arange(0, 25, 2))
plt.tight_layout()
plt.savefig('S_day.png')
plt.close()

# Plot xc(t) for the last day
plt.figure(figsize=(10, 4))
plt.plot(time_of_day, xc_sol[indices], color='purple')
plt.xlabel('Time of Day (HHMM)', fontsize=12)
plt.ylabel('$x_c$(t)', fontsize=12)
plt.title('$x_c$(t) Over the Last Simulated Day')
plt.grid(True, linestyle=':', alpha=0.7)
plt.xlim(0, 24)
plt.gca().xaxis.set_major_formatter(formatter)
plt.xticks(np.arange(0, 25, 2))
plt.tight_layout()
plt.savefig('xc_day.png')
plt.close()

# Plot SP(t) for the last day
plt.figure(figsize=(10, 4))
plt.plot(time_of_day, SP_sol[indices], color='teal')
plt.xlabel('Time of Day (HHMM)', fontsize=12)
plt.ylabel('SP(t)', fontsize=12)
plt.title('SP(t) Over the Last Simulated Day')
plt.grid(True, linestyle=':', alpha=0.7)
plt.xlim(0, 24)
plt.gca().xaxis.set_major_formatter(formatter)
plt.xticks(np.arange(0, 25, 2))
plt.tight_layout()
plt.savefig('SP_day.png')
plt.close()

print('Single-day plots saved as S_day.png, xc_day.png, and SP_day.png') 