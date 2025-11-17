import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress

# Sample data: time (in seconds) and distance (in meters)
time = np.array([0, 1, 2, 3, 4, 5])
distance = np.array([0, 2.1, 4.0, 6.1, 8.0, 10.2])
# Perform linear regression to find velocity
slope, intercept, r_value, p_value, std_err = linregress(time, distance)
# Slope represents velocity (m/s)
velocity = slope
# Generate fitted line for plotting
fitted_distance = intercept + slope * time
# Plotting the data and the fitted line
plt.scatter(time, distance, color="blue", label="Data Points")
plt.plot(time, fitted_distance, color="red", label="Fitted Line")
plt.title("Distance vs Time")
plt.xlabel("Time (s)")
plt.ylabel("Distance (m)")
plt.legend()
plt.grid()
plt.show()
# Output the calculated velocity
print(f"Calculated Velocity: {velocity:.2f} m/s")
# This script analyzes distance vs time data to calculate velocity using linear regression.
# It visualizes the data and the fitted line, and outputs the calculated velocity.
# This script analyzes distance vs time data to calculate velocity using linear regression.
# It visualizes the data and the fitted line, and outputs the calculated velocity.
# This script analyzes distance vs time data to calculate velocity using linear regression.
# It visualizes the data and the fitted line, and outputs the calculated velocity.
# This script analyzes distance vs time data to calculate velocity using linear regression.
# It visualizes the data and the fitted line, and outputs the calculated velocity.

# This script analyzes distance vs time data to calculate velocity using linear regression.
# It visualizes the data and the fitted line, and outputs the calculated velocity.
