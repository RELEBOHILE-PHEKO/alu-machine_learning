"""
Gradient Descent Function
"""
import numpy as np


def gradient_descent(x, y, m_initial, b_initial, alpha, num_iterations):
    """
    Performs gradient descent to find the best m and b.
    """
    # Initialize our parameters
    m_current = m_initial
    b_current = b_initial
    n = float(len(x))

    # Create lists to store the history of our parameters for plotting
    history_m = [m_current]
    history_b = [b_current]
    history_error = []

    print("--- Starting Gradient Descent ---")
    print(f"Initial State: m={m_current:.4f}, b={b_current:.4f}\n")

    # This is the core loop of the algorithm
    for i in range(num_iterations):
        # 1. Compute predicted values (y_hat)
        # This is the y = mx + b calculation for all our points at once
        y_hat = m_current * x + b_current

        # 2. Compute the error (Mean Squared Error)
        error = (1/n) * np.sum((y - y_hat)**2)
        history_error.append(error)

        # 3. Compute the gradients
        grad_m = -(2/n) * np.sum(x * (y - y_hat))
        grad_b = -(2/n) * np.sum(y - y_hat)

        # 4. Update the parameters m and b
        m_current = m_current - alpha * grad_m
        b_current = b_current - alpha * grad_b
        
        # Store the updated parameters for our plots
        history_m.append(m_current)
        history_b.append(b_current)

        print(f"Iteration {i+1}:")
        print(f"  Error: {error:.4f}")
        print(f"  Gradient (m, b): ({grad_m:.4f}, {grad_b:.4f})")
        print(f"  New (m, b): ({m_current:.4f}, {b_current:.4f})\n")

    return m_current, b_current, history_m, history_b, history_error
