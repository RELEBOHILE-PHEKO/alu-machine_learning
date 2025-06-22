"""
Visualization Function
"""
import matplotlib.pyplot as plt


def plot_results(history_m, history_b, history_error, num_iterations):
    """
    Visualizes the change in parameters and error over iterations.
    """
    iterations_range = range(num_iterations)
    
    # Create a figure with two subplots, side by side
    plt.figure(figsize=(14, 6))

    # Plot 1: Change in m and b
    plt.subplot(1, 2, 1)
    plt.plot(iterations_range, history_m[:-1], 'o-', label='m (slope)')
    plt.plot(iterations_range, history_b[:-1], 'o-', label='b (intercept)')
    plt.title('Change in Parameters (m & b) over Iterations', fontsize=14)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.legend()
    plt.grid(True)

    # Plot 2: Change in Error (MSE)
    plt.subplot(1, 2, 2)
    plt.plot(iterations_range, history_error, 'o-', color='red', label='Error (MSE)')
    plt.title('Change in Error (MSE) over Iterations', fontsize=14)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Error', fontsize=12)
    plt.legend()
    plt.grid(True)

    plt.tight_layout() # Adjusts plot to prevent labels from overlapping
    plt.show()
