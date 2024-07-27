import numpy as np
import matplotlib.pyplot as plt

def magnitude(num):
    if num<0:
        num*= -1
    return num

def lms(x, d, stepSize, n_iterations):
    """
    
    Parameters:
    x (numpy array): Input signal.
    d (numpy array): Desired signal.
    mu (float): Step size.
    n_iterations (int): Number of iterations.
    
    Returns:
    numpy array: Final filter coefficients.
    numpy array: Error signal.
    """
    n = len(x)
    w = np.zeros(n_iterations)  # Initialize filter coefficients
    e = np.zeros(n_iterations)  # Initialize error signal
    mu  = stepSize*2
    for i in range(n_iterations):
        y = np.dot(w[:i+1], x[:i+1][::-1])  # Filter output
        e[i] = d[i] - y                     # Error signal
        w[:i+1] += mu * e[i] * x[:i+1][::-1]  # Update filter coefficients
    
    eSum = 0
    for i in range(e.size):
        eSum += magnitude(e[i])
    eAvgMag = eSum / e.size
    print(eAvgMag)
    return w, e

# Example usage
n = 1000
np.random.seed(0)
x = np.random.randn(n)  # Input signal
d = 10* x + np.random.randn(n) * 0.1  # Desired signal with some noise
mu = 0.01 # Step size
n_iterations = n

# Apply LMS algorithm
w, e = lms(x, d, mu, n_iterations)

# Plot results
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(d, label='Desired Signal')
plt.plot(np.convolve(x, w, 'same'), label='LMS Output')
plt.title('Desired Signal vs LMS Output')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(e, label='Error Signal')
plt.title('Error Signal')
plt.legend()

plt.tight_layout()
plt.show()
