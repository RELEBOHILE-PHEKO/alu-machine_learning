import numpy as np
import matplotlib.pyplot as plt
from gradient_descent import gradient_descent
from plotting import plot_results

# --- 3. Main Execution Block ---
if __name__ == "__main__":
    # --- Setup the Problem ---
    x_points = np.array([1, 3])
    y_points = np.array([3, 6])
    
    # Initial parameters from the instructions
    initial_m = -1
    initial_b = 1
    learning_rate = 0.1

    iterations = 10 

    # --- Run the Gradient Descent ---
    final_m, final_b, m_hist, b_hist, error_hist = gradient_descent(
        x=x_points,
        y=y_points,
        m_initial=initial_m,
        b_initial=initial_b,
        alpha=learning_rate,
        num_iterations=iterations
    )

    print("\n--- Gradient Descent Finished ---")
    print(f"Initial m: {initial_m}, Initial b: {initial_b}")
    print(f"Final m:   {final_m:.4f}, Final b:   {final_b:.4f}")
    
    print("\n--- Predictions using Final Parameters ---")
    for x_val, y_val in zip(x_points, y_points):
        prediction = final_m * x_val + final_b
        print(f"For point ({x_val}, {y_val}), model predicts: {prediction:.4f}")

    # --- Plotting ---
    plot_results(m_hist, b_hist, error_hist, iterations)