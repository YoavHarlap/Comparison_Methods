import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, ifft


# print("\033[H\033[J")
# clear console

def phase(y):
    # Calculate the phase of the complex vector y
    magnitudes = np.abs(y)
    phase_y = np.where(magnitudes != 0, np.divide(y, magnitudes), 0)

    return phase_y


def PB(y, b):
    # Calculate the phase of the complex vector y
    phase_y = phase(y)

    # Point-wise multiplication between b and phase_y
    result = b * phase_y

    return result


def PB_for_p(x, b):
    # Calculate the phase of the complex vector y
    y = fft(x)
    result = PB(y, b)
    x = ifft(result)
    return x


def sparse_projection_on_vector(v, S):
    n = len(v)  # Infer the size of the DFT matrix from the length of y

    # Find indices of S largest elements in absolute values
    indices = np.argsort(np.abs(v))[-S:]

    # Create a sparse vector by zeroing out elements not in indices
    new_v = np.zeros(n, dtype='complex')
    new_v[indices] = np.array(v)[indices.astype(int)]

    return new_v


def step_RRR(S, b, p, beta):
    P_1 = sparse_projection_on_vector(p, S)
    P_2 = PB_for_p(2 * P_1 - p, b)
    p = p + beta * (P_2 - P_1)
    return p


# Hybrid Input-Output (HIO) algorithm step
def step_HIO(S, b, p, beta):
    P_1 = sparse_projection_on_vector(p, S)
    P_2 = PB_for_p((1 + beta) * P_1 - p, b)
    p = p + P_2 - beta * P_1  # Update using HIO formula
    return p


# Relaxed Averaged Alternating Reflections (RAAR) algorithm step
def step_RAAR(S, b, p, beta):
    P_1 = sparse_projection_on_vector(p, S)
    P_2 = PB_for_p(2 * P_1 - p, b)
    p = beta * (p + P_2) + (1 - 2 * beta) * P_1  # Update using RAAR formula
    return p


# Alternating Projection (AP) algorithm step
def step_AP(S, b, p):
    P_1 = sparse_projection_on_vector(p, S)
    P_2 = PB_for_p(P_1, b)
    p = P_2
    return P_2  # Return the updated result after projection


def mask_epsilon_values(p):
    # Separate real and imaginary parts
    real_part = p.real
    imag_part = p.imag

    epsilon = 0.5

    real_part = np.array(real_part)  # Make sure real_part is a NumPy array

    # Zero out elements with absolute values less than or equal to 1e-16 for real part
    real_part[np.abs(real_part) <= epsilon] = 0

    imag_part = np.array(imag_part)  # Make sure real_part is a NumPy array

    # Zero out elements with absolute values less than or equal to 1e-16 for imaginary part
    imag_part[np.abs(imag_part) <= epsilon] = 0

    # Combine real and imaginary parts back into the complex array
    result = real_part + 1j * imag_part

    # Printing the modified array
    # print(result)

    return result


def i_f(p):    
    squared_abs = np.abs(p) ** 2
    sum_squared_abs = np.sum(squared_abs)
    # print("sum_squared_abs",sum_squared_abs)

    if  np.real(i_s(p, S)) > np.real(sum_squared_abs):
        print(1394342)
    return sum_squared_abs


def i_s(p, S):
    p_sparse = sparse_projection_on_vector(p, S)
    squared_abs = np.abs(p_sparse) ** 2
    sum_squared_abs = np.sum(squared_abs)
    return sum_squared_abs


def power_p2_S(p, S):
    P_1 = sparse_projection_on_vector(p, S)
    # P_2 = PB_for_p(2 * P_1 - p, b)
    P_2 = PB_for_p(P_1, b)
    ratio = i_s(P_2, S) / i_f(P_2)

    # ratio = i_s(p, S) / i_f(p)
    # print("i_s(P_2, S) / i_f(P_2):", ratio)
    return ratio


def run_algorithm(S, b, p_init, algo, beta=None, max_iter=100, tolerance=1e-6):
    # Initialize y with the provided initial values
    p = p_init

    # Storage for plotting
    norm_diff_list = []
    norm_diff_min = 1000
    converged = -1

    for iteration in range(max_iter):
        if algo == "AP":
            p = step_AP(S, b, p)
        elif algo == "RRR":
            p = step_RRR(S, b, p, beta)
        elif algo == "RAAR":
            p = step_RAAR(S, b, p, beta)
        elif algo == "HIO":
            p = step_HIO(S, b, p, beta)
        else:
            raise ValueError(f"Unknown algorithm: {algo} :) ")

        # Calculate the i_s(P_2, S) / i_f(P_2) ratio:
        norm_diff = power_p2_S(p, S)

        # Store the norm difference for plotting
        norm_diff_list.append(norm_diff)
        # Check convergence
        if norm_diff > tolerance:
            print(f"{algo} Converged in {iteration + 1} iterations.")
            converged = iteration + 1
            break

    m_s_string = f"\nm = {m}, S = {S}, threshold = {tolerance}"
    
    
    
    # Plot the norm difference over iterations
    # plt.plot(norm_diff_list)
    # plt.xlabel('Iteration')
    # plt.ylabel(' i_s(P_2, S) / i_f(P_2) ratio')
    # plt.title(f' i_s(P_2, S) / i_f(P_2) ratio of {algo} Algorithm, threshold = {tolerance}' + m_s_string)
    # plt.show()

    # print("norm_diff_list:", norm_diff_list[-5:])
    return p, converged


beta = 0.5
max_iter = 10000
tolerance = 0.95
# Set dimensions
array_limit = 200
m_array = list(np.arange(10, array_limit + 1, 10))
S_array = list(np.arange(10, array_limit + 1, 10))
# S_array = np.arange(10, 70 + 1, 10)


m_array = list(np.arange(10, array_limit + 1, 50))
S_array = list(np.arange(10, array_limit + 1, 50))

m_array = [50,60,70,80]
S_array = [4,5]

m_array = [50]
S_array = [3]

m_S_average = []
algorithms = ["AP", "RRR", "RAAR", "HIO"]
sigma_values = np.linspace(0.01,2, 100)
sigma_values = [0,0.1]
convergence_values = []
# ppp = 10-(10-0.01)/200*6
# sigma_values = [10.0]
# Loop over different values of m and n

        
beta = 0.5
# Initialize data structures for storing convergence information
convergence_data = {algo: [] for algo in algorithms}
convergence_count = {algo: 0 for algo in algorithms}
index_of_operation = 0
total_trials = 10000
# total_trials = 3

sigma=0.5
sigma=0
m=50
S=5

for trial in range(total_trials):

    print(f"\nTrial {trial + 1}/{total_trials}")
    np.random.seed(trial)

    m_s_string = f"\nm = {m}, S = {S}, threshold = {tolerance}"
    print(f"m = {m}, S = {S}")
    x_sparse_real_true = sparse_projection_on_vector(np.random.randn(m), S)
    # print("x_sparse_real_true:", x_sparse_real_true[:5])

    # Calculate b = |fft(x)|
    b = np.abs(fft(x_sparse_real_true))

    x_sparse_real_init = np.random.randn(m)
    p_init = x_sparse_real_init
    convergence_values = []
    # Add Gaussian noise
    print(sigma)
    noise = np.random.normal(0, sigma, b.shape) 
    # noise = 0
    b_copy = b.copy() + noise

    for algo in algorithms:
        print(f"Running {algo}...")

        result_RRR, converged = run_algorithm(S, b_copy, p_init, algo=algo, beta=beta, max_iter=max_iter,
                                              tolerance=tolerance)
        
        # Store convergence data
        convergence_data[algo].append(converged if converged != -1 else None)
        
        # Count successful convergence
        if converged != -1:
            convergence_count[algo] += 1
        
        # Plot the result for this algorithm
        
    index_of_operation += 1

# # Convergence plots
# for algo in algorithms:
#     plt.semilogy(range(index_of_operation), convergence_data[algo], label=f'{algo} Converged')

# plt.xlabel('Scenario')
# plt.ylabel('Convergence - num of iterations')
# plt.title('Convergence Plot')
# plt.legend()
# plt.show()

plt.title(f'Percentage of Successful Convergences for Each Algorithm: m={m}, S={S}, sigma = {sigma}, trials = {total_trials}')
plt.show()

# Logarithmic convergence plot
colors = ['blue', 'green', 'red', 'purple']
markers = ['s-', 'o--', 'd-.', 'v:']
max_iter = 10000

for i, algo in enumerate(algorithms):
    convergence_data[algo] = [max_iter if x == None else x for x in convergence_data[algo]]

    plt.semilogy(range(index_of_operation), convergence_data[algo], markers[i], color=colors[i], label=f'{algo}')

plt.xlabel('Trial Index')
plt.ylabel('Number of Iterations (log scale)')
plt.axhline(y=max_iter, color='black', linestyle='--', label='Maximal Number of Iterations')
plt.legend()

# plt.title('Logarithmic Plot of Converged Values')
plt.grid(True, which="both", ls="--")
plt.show()

# Plotting the percentages of successful convergence
convergence_percentages = {algo: (convergence_count[algo] / total_trials) * 100 for algo in algorithms}

plt.figure(figsize=(10, 6))
bars = plt.bar(convergence_percentages.keys(), convergence_percentages.values(), color=['blue', 'green', 'red', 'purple'])
plt.xlabel('Algorithm')
plt.ylabel('Convergence Percentage (%)')
# plt.title(f'Percentage of Successful Convergences for Each Algorithm: m={m}, S={S}, sigma = {sigma}, trials = {total_trials}')
plt.ylim(0, 100)
    # Add percentage value text on each bar
max_height = max(convergence_percentages.values(), default=0)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, yval + max_height * (-.05), f'{yval:.2f}%', ha='center',
             va='bottom')
plt.show()

# Beep sound
import winsound
winsound.Beep(1000, 501)  # Frequency 1000 Hz, duration 500 ms












# Define the path
path = r"C:\Users\ASUS\Documents\code_images\overleaf_images\Numerical_experiments\dft_case\generic2" 
import os

import json

# # Ensure the directory exists
# os.makedirs(path, exist_ok=True)

# # Save the file as JSON
file_path = os.path.join(path, "iteration_counts.json")
with open(file_path, "w") as file:
    json.dump(convergence_data, file)

print(f"File saved to: {file_path}")



# # Load the file back
with open(file_path, "r") as file:
    loaded_iteration_counts = json.load(file)

print("Loaded data:", loaded_iteration_counts)


# loaded_iteration_counts = {algo: counts[100:125] for algo, counts in loaded_iteration_counts.items()}



# Logarithmic convergence plot
colors = ['blue', 'green', 'red', 'purple']
markers = ['s-', 'o--', 'd-.', 'v:']
max_iter = 10000

for i, algo in enumerate(algorithms):
    loaded_iteration_counts[algo] = [max_iter if x == None else x for x in loaded_iteration_counts[algo]]

    plt.semilogy(range(len(loaded_iteration_counts["RRR"])), loaded_iteration_counts[algo], markers[i], color=colors[i], label=f'{algo}',linestyle='None')

plt.xlabel('Trial Index')
plt.ylabel('Number of Iterations (log scale)')
plt.axhline(y=max_iter, color='black', linestyle='--', label='Maximal Number of Iterations')
plt.legend()

# plt.title('Logarithmic Plot of Converged Values')
plt.grid(True, which="both", ls="--")
plt.show()


loaded_iteration_counts = {algo: counts[100:125] for algo, counts in loaded_iteration_counts.items()}



# Logarithmic convergence plot
colors = ['blue', 'green', 'red', 'purple']
markers = ['s-', 'o--', 'd-.', 'v:']
max_iter = 10000

for i, algo in enumerate(algorithms):
    loaded_iteration_counts[algo] = [max_iter if x == None else x for x in loaded_iteration_counts[algo]]

    plt.semilogy(range(len(loaded_iteration_counts["RRR"])), loaded_iteration_counts[algo], markers[i], color=colors[i], label=f'{algo}')

plt.xlabel('Trial Index')
plt.ylabel('Number of Iterations (log scale)')
plt.axhline(y=max_iter, color='black', linestyle='--', label='Maximal Number of Iterations')
plt.legend()

# plt.title('Logarithmic Plot of Converged Values')
plt.grid(True, which="both", ls="--")
plt.show()





