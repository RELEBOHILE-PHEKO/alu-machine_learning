#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

bins = np.arange(0, 110, 10)  # Bins every 10 units from 0 to 100

# Plot with black-outlined bars
plt.hist(student_grades, bins=bins, edgecolor='black')

plt.xlabel('Grades')
plt.ylabel('Number of Students')
plt.title('Project A')

plt.xticks(np.arange(10, 110, 10))  # Show all x-axis ticks
plt.yticks(np.arange(0, 35, 5))     # Show y-axis ticks from 0 to 30

plt.show()

