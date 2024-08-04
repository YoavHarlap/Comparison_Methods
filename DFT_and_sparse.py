import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from scipy.fft import fft, ifft
##898909
print()

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
    # P_2 = mask_epsilon_values(P_2)
    # P_3cp =  sparse_projection_on_vector(P_2, S)
    p = p + beta * (P_2 - P_1)
    # P_4dp = sparse_projection_on_vector(p, S)
    return p


def mask_epsilon_values(p):
    # Separate real and imaginary parts
    real_part = p.real
    imag_part = p.imag

    epsilon = 2
    # Zero out elements with absolute values less than or equal to 1e-16 for real part
    real_part[np.abs(real_part) <= epsilon] = 0

    # Zero out elements with absolute values less than or equal to 1e-16 for imaginary part
    imag_part[np.abs(imag_part) <= epsilon] = 0

    # Combine real and imaginary parts back into the complex array
    result = real_part + 1j * imag_part

    # Printing the modified array
    # print(result)

    return result


def i_f(p):
    return sum(x ** 2 for x in p)


def i_s(p, S):
    p_sparse = sparse_projection_on_vector(p, S)
    return sum(x ** 2 for x in p_sparse)


def power_p2_S(p, S):
    P_1 = sparse_projection_on_vector(p, S)
    P_2 = PB_for_p(2 * P_1 - p, b)
    print("i_s(P_2, S) / i_f(P_2):", i_s(P_2, S) / i_f(P_2))

    return i_s(P_2, S) / i_f(P_2)


def run_algorithm(S, b, p_init, algo, beta=None, max_iter=100, tolerance=1e-6):
    # Initialize y with the provided initial values
    p = p_init

    # Storage for plotting
    norm_diff_list = []
    norm_diff_min = 1000
    converged = None

    if algo == "alternating_projections":
        print(f"{algo}")

    elif algo == "RRR_algorithm":
        for iteration in range(max_iter):
            # if iteration % 100 == 0:
            #     print("iteration:", iteration)
            p = step_RRR(S, b, p, beta)

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
    plt.plot(norm_diff_list)
    plt.xlabel('Iteration')
    plt.ylabel(' i_s(P_2, S) / i_f(P_2) ratio')
    plt.title(f' i_s(P_2, S) / i_f(P_2) ratio of {algo} Algorithm, threshold = {tolerance}' + m_s_string)
    plt.show()

    print("norm_diff_list:", norm_diff_list[-5:])

    return p, converged


def plot_m_S_average(m_S_average, max_iter, tolerance):
    m_S_average = np.array(m_S_average)
    max_iter_str = f", max_iter: {max_iter}, threshold = {tolerance}"

    # Extracting m, S, and average_changes
    m_values = m_S_average[:, 0]
    S_values = m_S_average[:, 1]
    average_changes_values = m_S_average[:, 2]
    converged_values = m_S_average[:, 3]

    # Set a threshold for constant color
    threshold_value = 2
    # Clip values greater than the threshold to the threshold
    clipped_values = np.clip(average_changes_values, None, threshold_value)

    # Creating a colormap based on clipped_values
    norm = plt.Normalize(clipped_values.min(), threshold_value)
    # cmap = cm.viridis  # Viridis colormap
    cmap = cm.RdYlGn_r  # Reverse Red-Yellow-Green colormap

    # Plotting
    scatter = plt.scatter(m_values, S_values, c=clipped_values, cmap=cmap, norm=norm)
    # Plotting
    for m, S, avg_change, converged in zip(m_values, S_values, clipped_values, converged_values):
        if converged is None:
            plt.scatter(m, S, marker='x', color='black')

    plt.xlabel('m')
    plt.ylabel('S')
    plt.title(
        'Scatter Plot for m and S with Color-Coded average_changes' + "\n" 'Plot "x" if not converged' + max_iter_str)
    plt.colorbar(scatter, label='average_changes (clipped)')
    plt.show()


beta = 0.5
max_iter = 10000
tolerance = 0.95
# Set dimensions
array_limit = 200
m_array = np.arange(10, array_limit + 1, 10)
S_array = np.arange(10, array_limit + 1, 10)
# S_array = np.arange(10, 70 + 1, 10)

m_array = [1000]
S_array = [2]

m_S_average = []

# Loop over different values of m and n
for m in m_array:  # Add more values as needed
    for S in S_array:  # Add more values as needed

        if S > m:
            break

        np.random.seed(44)  # For reproducibility

        m_s_string = f"\nm = {m}, S = {S}, threshold = {tolerance}"
        print(f"m = {m}, S = {S}")
        x_sparse_real_true = sparse_projection_on_vector(np.random.randn(m), S)
        print("x_sparse_real_true:", x_sparse_real_true[:5])

        # Calculate b = |fft(x)|
        b = np.abs(fft(x_sparse_real_true))
        x_sparse_real_init = np.random.randn(m)
        p_init = x_sparse_real_init

        # result_AP = run_algorithm(S, b, y_initial, algo="alternating_projections", max_iter=max_iter,
        #                           tolerance=tolerance)
        # print("result_AP:", np.abs(result_AP[:5]))
        # print("b:        ", b[:5])

        result_RRR, converged = run_algorithm(S, b, p_init, algo="RRR_algorithm", beta=beta, max_iter=max_iter,
                                              tolerance=tolerance)
        print("result_RRR:        ", result_RRR[:5])
        print("x_sparse_real_true:", x_sparse_real_true[:5])

        # Plot the data with specified colors and labels
        plt.plot(x_sparse_real_true, label=f"Sparse: S = {S}, Original Vector", color='blue')
        plt.plot(x_sparse_real_init, label='Random Initial Vector', color='green')
        plt.plot(result_RRR, label='Result RRR', color='red')
        # Add legend
        plt.legend()
        plt.title("The vectors values" + m_s_string)
        # Show the plot
        plt.show()

        # plt.plot(np.abs(fft(result_RRR)), label='abs fft for result_RRR', color='blue')
        # # Add legend
        # # plt.legend()
        # plt.title("abs fft for result_RRR"+m_s_string)
        # # Show the plot
        # plt.show()

        plt.plot(sparse_projection_on_vector(result_RRR, S), label='result_RRR after sparse projection', color='red')

        plt.plot(x_sparse_real_true, label='Sparse Original Vector', color='blue')
        # Add legend
        plt.legend()
        plt.title("Original Vector and result_RRR after sparse projection" + m_s_string)
        # Show the plot
        plt.show()

        plt.plot(np.abs(fft(x_sparse_real_true)), label='abs fft for Sparse Original Vector', color='blue')
        # Add legend
        # plt.legend()
        plt.title("abs fft for Sparse Original Vector" + m_s_string)
        # Show the plot
        plt.show()

        plt.plot(np.abs(fft(sparse_projection_on_vector(result_RRR, S))),
                 label='abs fft for result_RRR after sparse projection', color='blue')
        # Add legend
        # plt.legend()
        plt.title("abs fft for result_RRR after sparse projection" + m_s_string)
        # Show the plot
        plt.show()

        plt.plot(np.abs(fft(x_sparse_real_true)), label='abs fft for Sparse Original Vector', color='blue')
        plt.plot(np.abs(fft(sparse_projection_on_vector(result_RRR, S))),
                 label='abs fft for result_RRR after sparse projection', color='red')
        # Add legend
        plt.legend()
        plt.title("abs fft for Sparse Original Vector And abs fft for result_RRR after sparse projection" + m_s_string)
        # Show the plot
        plt.show()

        # Compute FFT for both vectors
        fft_a = np.abs(fft(sparse_projection_on_vector(result_RRR, S)))
        fft_b = np.abs(fft(x_sparse_real_true))
        average_changes = np.mean(np.abs(fft_a - fft_b))

        m_S_average.append([m, S, average_changes, converged])

        mean = np.round(average_changes, 2)
        # Plot the absolute difference of FFT magnitudes
        plt.figure(figsize=(10, 6))
        plt.plot(np.abs(fft_a - fft_b), label='||FFT(a)| - |FFT(b)||')
        plt.title('Difference of FFT Magnitudes' + m_s_string)
        plt.axhline(y=average_changes, color='g', linestyle=':', label=f"Horizontal Line at mean changes: {mean}")
        # plt.xlabel('Frequency')
        plt.ylabel('Magnitude Difference')
        plt.legend()
        plt.show()

plot_m_S_average(m_S_average, max_iter, tolerance)



