#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Data setup
y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r = np.log(0.5)
t1 = 5730
y2 = np.exp((r / t1) * x2)

x3 = np.arange(0, 21000, 1000)
t2 = 1600
y3_c14 = np.exp((r / t1) * x3)
y3_ra226 = np.exp((r / t2) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

# Create figure and define layout
fig = plt.figure(constrained_layout=True, figsize=(10, 8))
gs = gridspec.GridSpec(3, 2, figure=fig)

# Plot 1: y = x³
ax0 = fig.add_subplot(gs[0, 0])
ax0.plot(np.arange(0, 11), y0)
ax0.set_title("y = x³", fontsize='x-small')
ax0.set_xlabel("x", fontsize='x-small')
ax0.set_ylabel("y", fontsize='x-small')

# Plot 2: Scatter plot
ax1 = fig.add_subplot(gs[0, 1])
ax1.scatter(x1, y1, color='magenta')
ax1.set_title("Men's Height vs Weight", fontsize='x-small')
ax1.set_xlabel("Height (in)", fontsize='x-small')
ax1.set_ylabel("Weight (lbs)", fontsize='x-small')

# Plot 3: Exponential decay (log scale)
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(x2, y2)
ax2.set_yscale('log')
ax2.set_title("Exponential Decay of C-14", fontsize='x-small')
ax2.set_xlabel("Time (years)", fontsize='x-small')
ax2.set_ylabel("Fraction Remaining", fontsize='x-small')
ax2.set_xlim(0, 28650)

# Plot 4: C-14 vs Ra-226
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(x3, y3_c14, 'r--', label='C-14')
ax3.plot(x3, y3_ra226, 'g-', label='Ra-226')
ax3.set_title("Exponential Decay of Radioactive Elements", fontsize='x-small')
ax3.set_xlabel("Time (years)", fontsize='x-small')
ax3.set_ylabel("Fraction Remaining", fontsize='x-small')
ax3.set_xlim(0, 20000)
ax3.set_ylim(0, 1)
ax3.legend(fontsize='x-small', loc='upper right')

# Plot 5: Histogram (takes full bottom row)
ax4 = fig.add_subplot(gs[2, :])
bins = np.arange(0, 110, 10)
ax4.hist(student_grades, bins=bins, edgecolor='black')
ax4.set_title("Project A", fontsize='x-small')
ax4.set_xlabel("Grades", fontsize='x-small')
ax4.set_ylabel("Number of Students", fontsize='x-small')
ax4.set_xticks(bins[1:])
ax4.set_yticks(np.arange(0, 35, 5))

# Main figure title
fig.suptitle("All in One", fontsize='x-small')
plt.show()

