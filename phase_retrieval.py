import matplotlib.pyplot as plt
import numpy as np


def phase(y):
    y1 = np.copy(y)
    nonzero_indices = np.nonzero(y1)
    if not len(nonzero_indices) == 0 and not len(np.nonzero(np.all(np.abs(y1[nonzero_indices]) < 1e-5))) == 0:
        y1[nonzero_indices] /= np.abs(y1[nonzero_indices])
    for i, val in enumerate(y1):
        if np.isnan(val):
            print("NaN found at index", i)
    return y1


def PB(y, b):
    phase_y = phase(y)
    result = b * phase_y
    return result


def PA(y, A):
    A_dagger = np.linalg.pinv(A)
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
    return y_PA


def step_RAAR(A, b, y, beta):
    P_Ay = PA(y, A)
    P_By = PB(y, b)
    PAPB_y = PA(2 * P_By - y, A)
    result = beta * (y + PAPB_y) + (1 - 2 * beta) * P_By
    return result


def step_HIO(A, b, y, beta):
    P_By = PB(y, b)
    P_Ay = PA((1 + beta) * P_By - y, A)
    result = y + P_Ay - beta * P_By
    return result


def run_algorithm(A, b, y_init, algo, beta=0.5, max_iter=100, tolerance=1e-6, alpha=0.5):
    y = y_init
    norm_diff_list = []
    converged = -1

    for iteration in range(max_iter):
        if algo == "alternating_projections":
            y = step_AP(A, b, y)
        elif algo == "RRR_algorithm":
            y = step_RRR(A, b, y, beta)
        elif algo == "RAAR_algorithm":
            y = step_RAAR(A, b, y, beta)
        elif algo == "HIO_algorithm":
            y = step_HIO(A, b, y, beta)
        else:
            raise ValueError(f"Unknown algorithm: {algo}")

        norm_diff = np.linalg.norm(PB(y, b) - PA(y, A))
        norm_diff_list.append(norm_diff)

        if norm_diff < tolerance:
            print(f"{algo} Converged in {iteration + 1} iterations. for beta = {beta}")
            converged = iteration + 1
            break

    # plt.plot(norm_diff_list)
    # plt.xlabel('Iteration')
    # plt.ylabel('|PB - PA|')
    # plt.title(f'Convergence of {algo} Algorithm')
    # plt.show()
    return y, converged


# beta = 1
max_iter = 10000
tolerance = 1e-4

m_array = [25,26,27,28,29,30,31,32,33,34,35,36]
m_array = [25,26,27]

n_array = [7, 8, 9,10,11,12,13]
n_array = [17, 18, 19,20,21,22,23]
n_array = [7, 8,9,10]


betas = [0.5]
AP_converged_list = []
RRR_converged_list = []
RAAR_converged_list = []
HIO_converged_list = []
index_of_operation = 0

for m in m_array:
    for n in n_array:
        for beta in betas:
            np.random.seed(42)

            print(f"m = {m}, n = {n}")

            A = np.random.randn(m, n) + 1j * np.random.randn(m, n)
            A_real = np.random.randn(m, n)

            x = np.random.randn(n) + 1j * np.random.randn(n)
            x_real = np.random.randn(n)

            b = np.abs(np.dot(A, x))
            b_real = np.abs(np.dot(A_real, x_real))

            y_true = np.dot(A, x)
            y_true_real = np.dot(A_real, x_real)

            y_initial = np.random.randn(m) + 1j * np.random.randn(m)
            y_initial_real = np.random.randn(m)

            # A = A_real
            # b = b_real
            # y_initial = y_initial_real
            # y_true = y_true_real

            result_AP, AP_converged = run_algorithm(A, b, y_initial, algo="alternating_projections", max_iter=max_iter,
                                                    tolerance=tolerance)
            AP_converged_list.append(AP_converged if AP_converged != -1 else None)

            result_RRR, RRR_converged = run_algorithm(A, b, y_initial, algo="RRR_algorithm", beta=beta,
                                                      max_iter=max_iter, tolerance=tolerance)
            RRR_converged_list.append(RRR_converged if RRR_converged != -1 else None)

            result_RAAR, RAAR_converged = run_algorithm(A, b, y_initial, algo="RAAR_algorithm", beta=beta,
                                                        max_iter=max_iter, tolerance=tolerance)
            RAAR_converged_list.append(RAAR_converged if RAAR_converged != -1 else None)

            result_HIO, HIO_converged = run_algorithm(A, b, y_initial, algo="HIO_algorithm", beta=beta,
                                                      max_iter=max_iter, tolerance=tolerance)
            HIO_converged_list.append(HIO_converged if HIO_converged != -1 else None)

            index_of_operation += 1

            plt.plot(abs(PA(result_AP, A)), label='result_AP')
            plt.plot(abs(PA(result_RRR, A)), label='result_RRR')
            plt.plot(abs(PA(result_RAAR, A)), label='result_RAAR')
            plt.plot(abs(PA(result_HIO, A)), label='result_HIO')

            plt.plot(b, label='b')
            plt.xlabel('element')
            plt.ylabel('value')
            plt.title('Plot of Terms')
            plt.legend()
            plt.show()

plt.semilogy(range(index_of_operation), AP_converged_list, '-o', label='AP Converged')
plt.semilogy(range(index_of_operation), RRR_converged_list, '-o', label='RRR Converged')
plt.semilogy(range(index_of_operation), RAAR_converged_list, '-o', label='RAAR Converged')
plt.semilogy(range(index_of_operation), HIO_converged_list, '-o', label='HIO Converged')

plt.xlabel('Scenario')
plt.ylabel('Convergence - num of iterations')
plt.title('Convergence Plot')
plt.legend()
plt.show()


# Plot
plt.semilogy(range(index_of_operation), AP_converged_list, 's-', color='blue', label='AP Converged')
plt.semilogy(range(index_of_operation), RRR_converged_list, 'o--', color='green', label='RRR Converged')
plt.semilogy(range(index_of_operation), RAAR_converged_list, 'd-.', color='red', label='RAAR Converged')
plt.semilogy(range(index_of_operation), HIO_converged_list, 'v:', color='purple', label='HIO Converged')

plt.xlabel('Index of Operation')
plt.ylabel('Converged Value (log scale)')
plt.legend()
plt.title('Logarithmic Plot of Converged Values')
plt.grid(True, which="both", ls="--")

plt.show()

import winsound
# Beep sound
winsound.Beep(1000, 500)  # Frequency 1000 Hz, duration 500 ms