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


def run_algorithm(S, b, p_init, algo, beta=None, max_iter=100, tolerance=1e-6,sigma=0):
    # Initialize y with the provided initial values
    p = p_init

    # Storage for plotting
    norm_diff_list = []
    norm_diff_min = 1000
    converged = None

    for iteration in range(max_iter):
        if algo == "Alternating Projections":
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

    m_s_string = f"\nn = {m}, S = {S}, threshold = {tolerance}"
    
    
    
    # Plot the norm difference over iterations
    plt.plot(norm_diff_list)
    plt.xlabel('Iteration')
    plt.ylabel(' i_s(P_2, S) / i_f(P_2) ratio')
    plt.title(f' i_s(P_2, S) / i_f(P_2) ratio of {algo} Algorithm, sigma = {sigma}' + m_s_string)
    plt.show()

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

m_array = [40]
S_array = [4]

m_S_average = []
algorithms = ["Alternating Projections", "RRR", "RAAR", "HIO"]
sigma_values = np.linspace(0,10, 300)
sigma_values = np.round(sigma_values, 2)
# sigma_values = [0]
convergence_values = []
# ppp = 10-(10-0.01)/200*6
# sigma_values = [10.0]
# Loop over different values of m and n
for m in m_array:  # Add more values as needed
    for S in S_array:  # Add more values as needed

        if S > 0.5 * m:
            break

        np.random.seed(44)  # For reproducibility

        m_s_string = f"\nn = {m}, S = {S}, threshold = {tolerance}"
        print(f"n = {m}, S = {S}")
        x_sparse_real_true = sparse_projection_on_vector(np.random.randn(m), S)
        # print("x_sparse_real_true:", x_sparse_real_true[:5])

        # Calculate b = |fft(x)|
        b = np.abs(fft(x_sparse_real_true))

        x_sparse_real_init = np.random.randn(m)
        p_init = x_sparse_real_init
        convergence_values = []

        for sigma in sigma_values:
            # Add Gaussian noise
            print(sigma)
            noise = np.random.normal(0, sigma, b.shape) 
            # noise = 0
            b_copy = b.copy() + noise
            result_RRR, converged = run_algorithm(S, b_copy, p_init, algo=algorithms[1], beta=beta, max_iter=max_iter,
                                                  tolerance=tolerance,sigma=sigma)
            convergence_values.append(converged)
            
        plt.plot(sigma_values, convergence_values, label=f'n={m}, S={S}', marker='H', linestyle='None')
        plt.title("Convergence Iteration Status Across Different Sigma Values")
        plt.xlabel("Sigma (Noise level)")
        
        # plt.xlim(sigma_values[0], sigma_values[-1])  # Force x-axis limits to include all x values
        # num_ticks = min(10, len(sigma_values) // 25)  # Show up to 10 ticks, adapt this if needed
        # ticks = sigma_values[::max(1, len(sigma_values) // num_ticks)]  # Base ticks selection
        # if sigma_values[-1] not in ticks:  # Ensure the last tick is included
        #     ticks = np.append(ticks, sigma_values[-1])  # Add the last x value if it's not already included
        
        # plt.xticks(ticks=np.sort(ticks))  # Sort the ticks to maintain order
        
        # plt.xticks(ticks=[0,0.5,1,1.5,2])
        plt.ylabel("Convergence Iteration (log scale)")
        # plt.yscale('log')
        plt.legend()
        plt.yscale('log')
        plt.grid(True)
        plt.show()
        # print("result_RRR:        ", result_RRR[:5])
        # print("x_sparse_real_true:", x_sparse_real_true[:5])
        
        plt.plot(np.abs(fft(x_sparse_real_true)), label='abs fft for Sparse Original Vector', color='blue')
        # plt.plot(np.abs((b_copy)), label='abs fft for noisy Original Vector', color='green')

        plt.plot(np.abs(fft(sparse_projection_on_vector(result_RRR, S))),
                  label='abs fft for result_RRR after sparse projection', color='red')
        # Add legend
        plt.legend()
        plt.title("abs fft for Sparse Original Vector And abs fft for result_RRR after sparse projection" + m_s_string)
        # Show the plot
        plt.show()
