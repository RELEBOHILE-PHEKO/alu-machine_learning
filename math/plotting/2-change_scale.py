#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 28651, 5730)
r = np.log(0.5)
t = 5730
y = np.exp((r / t) * x)

# plotting
plt.plot(x, y)
plt.xlabel("Time (years)")
plt.ylabel("Fraction Remaining")
plt.title("Exponential Decay of C-14")
plt.xlim(0, 28650)
plt.yscale('log')  # Logarithmic scale for y-axis
plt.show()
