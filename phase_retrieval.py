import matplotlib.pyplot as plt
import numpy as np


# from print_to_txt_file import Tee


def phase(y):
    # Calculate the phase of the complex vector y
    # magnitudes = np.abs(y)
    # phase_y = np.where(magnitudes != 0, np.divide(y, magnitudes), 0)

    y1 = np.copy(y)
    # Find indices where t is not zero
    nonzero_indices = np.nonzero(y1)
    if not len(nonzero_indices) == 0 and not len(np.nonzero(np.all(np.abs(y1[nonzero_indices]) < 1e-5))) == 0:
        y1[nonzero_indices] /= np.abs(y1[nonzero_indices])

    for i, val in enumerate(y1):
        if np.isnan(val):
            print("NaN found at index", i)

    return y1


def PB(y, b):
    # Calculate the phase of the complex vector y
    phase_y = phase(y)
    # Point-wise multiplication between b and phase_y
    result = b * phase_y
    return result


def PA(y, A):
    # Calculate the pseudo-inverse of A
    A_dagger = np.linalg.pinv(A)
    # Matrix-vector multiplication: AA†y
    result = np.dot(A, np.dot(A_dagger, y))
    return result


def step_RRR(A, b, y, beta):
    P_Ay = PA(y, A)
    P_By = PB(y, b)
    PAPB_y = PA(P_By, A)
    result = y + beta * (2 * PAPB_y - P_Ay - P_By)
    return result


def step_AP(A, b, y):
    y_PB = PB(y, b)
    y_PA = PA(y_PB, A)
    resulr = y_PA
    return resulr


def run_algorithm(A, b, y_init, algo, beta=0.5, max_iter=100, tolerance=1e-6, alpha=0.5):
    y = y_init
    norm_diff_list = []
    converged = -1
    if algo == "alternating_projections":
        for iteration in range(max_iter):
            y = step_AP(A, b, y)
            norm_diff = np.linalg.norm(PB(y, b) - PA(y, A))
            norm_diff_list.append(norm_diff)

            # Check convergence
            if norm_diff < tolerance:
                print(f"{algo} Converged in {iteration + 1} iterations.")
                converged = iteration + 1
                break

    elif algo == "RRR_algorithm":
        for iteration in range(max_iter):

            y = step_RRR(A, b, y, beta)

            for i, val in enumerate(y):
                if np.isinf(val):
                    print("NaN found at index", i)

            norm_diff = np.linalg.norm(PB(y, b) - PA(y, A))
            norm_diff_list.append(norm_diff)

            # Check convergence
            if norm_diff < tolerance:
                print(f"{algo} Converged in {iteration + 1} iterations.  for beta = {beta}")
                converged = iteration + 1
                break

    # Plot the norm difference over iterations

    plt.plot(norm_diff_list)
    plt.xlabel('Iteration')
    plt.ylabel('|PB - PA|')
    plt.title(f'Convergence of {algo} Algorithm')
    plt.show()
    return y, converged


#
# log_file_path = os.path.join("texts", "RRR_and_GD.txt")
# log_file = open(log_file_path, "w")
# sys.stdout = Tee(sys.stdout, log_file)

beta = 0.5
max_iter = 3000
tolerance = 1e-4

array_limit = 200
m_array = np.arange(10, array_limit + 1, 10)
n_array = np.arange(10, array_limit + 1, 10)

m_array = [30, 31, 32, 33, 34, 35]
n_array = [6, 7, 8, 9, 10, 11, 12, 13]

m_array = [30]
n_array = [6, 7, 8]

# n_array = range(5, 14, 9)
# betas = np.linspace(0.3, 0.999, 10)
betas = [1]
AP_converged_list = []
RRR_converged_list = []
index_of_operation = 0
for m in m_array:  # Add more values as needed
    for n in n_array:  # Add more values as needed
        for beta in betas:
            np.random.seed(42)  # For reproducibility

            print(f"m = {m}, n = {n}")  # Restore the standard output after the loop

            A = np.random.randn(m, n) + 1j * np.random.randn(m, n)
            A_real = np.random.randn(m, n)

            x = np.random.randn(n) + 1j * np.random.randn(n)
            x_real = np.random.randn(n)

            # Calculate b = |Ax|
            b = np.abs(np.dot(A, x))
            b_real = np.abs(np.dot(A_real, x_real))

            y_true = np.dot(A, x)
            y_true_real = np.dot(A_real, x_real)

            # Initialize y randomly
            y_initial = np.random.randn(m) + 1j * np.random.randn(m)
            y_initial_real = np.random.randn(m)

            A = A_real
            b = b_real
            y_initial = y_initial_real
            y_true = y_true_real

            result_AP, AP_converged = run_algorithm(A, b, y_initial, algo="alternating_projections", max_iter=max_iter,
                                                    tolerance=tolerance)

            AP_converged_list.append(AP_converged)

            result_RRR, RRR_converged = run_algorithm(A, b, y_initial, algo="RRR_algorithm", beta=beta,
                                                      max_iter=max_iter, tolerance=tolerance)
            RRR_converged_list.append(RRR_converged)

            index_of_operation += 1

            plt.plot(abs(PA(result_AP, A)), label='result_AP')
            plt.plot(abs(PA(result_RRR, A)), label='result_RRR')

            plt.plot(b, label='b')
            plt.xlabel('element')
            plt.ylabel('value')
            plt.title('Plot of Terms')
            plt.legend()
            plt.show()

plt.plot(range(index_of_operation), AP_converged_list, label='AP Converged')
plt.plot(range(index_of_operation), RRR_converged_list, label='RRR Converged')

# Adding labels and title
plt.xlabel('Scenario')
plt.ylabel('Convergence - num of iterations')
plt.title('Convergence Plot')

# Adding legends
plt.legend()

# Displaying the plot
plt.show()
