# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 15:44:57 2024
no figgggg noooo
@author: ASUS
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import matrix_rank, svd

def initialize_matrix(n, r, q, seed=None):
    if seed is not None:
        np.random.seed(seed)  # Set seed for reproducibility

    # Initialize a random matrix of rank r
    true_matrix = np.random.rand(n, r) @ np.random.rand(r, n)
    hints_matrix = true_matrix.copy()
    # print("Original matrix rank:", matrix_rank(hints_matrix))

    # Set q random entries to NaN (missing entries)
    missing_entries = np.random.choice(n * n, q, replace=False)
    row_indices, col_indices = np.unravel_index(missing_entries, (n, n))
    # print(row_indices,col_indices)
    hints_matrix[row_indices, col_indices] = 0
    # print("Matrix rank after setting entries to zero:", matrix_rank(hints_matrix))
    hints_indices = np.ones_like(true_matrix, dtype=bool)
    hints_indices[row_indices, col_indices] = False

    # Ensure the rank is still r
    U, Sigma, Vt = svd(hints_matrix)
    Sigma[r:] = 0  # Zero out singular values beyond rank r
    new_matrix = U @ np.diag(Sigma) @ Vt
    # print("Matrix rank after preserving rank:", matrix_rank(new_matrix))
    initial_matrix = new_matrix

    return [true_matrix, initial_matrix, hints_matrix, hints_indices]


def proj_2(matrix, r):
    # Perform SVD and truncate to rank r
    u, s, v = np.linalg.svd(matrix, full_matrices=False)
    matrix_proj_2 = u[:, :r] @ np.diag(s[:r]) @ v[:r, :]

    if r != matrix_rank(matrix_proj_2):
        print(f"matrix_rank(matrix_proj_2): {matrix_rank(matrix_proj_2)}, not equal r: {r}")

    # # Ensure the rank is still r
    # U, Sigma, Vt = svd(matrix)
    # Sigma[r:] = 0  # Zero out singular values beyond rank r
    # new_matrix = U @ np.diag(Sigma) @ Vt

    return matrix_proj_2


def proj_1(matrix, hints_matrix, hints_indices):
    matrix_proj_1 = matrix.copy()
    # Set non-missing entries to the corresponding values in the initialization matrix
    matrix_proj_1[hints_indices] = hints_matrix[hints_indices]
    return matrix_proj_1


def plot_sudoku(matrix, colors, ax, title, missing_elements_indices):
    n = matrix.shape[0]

    # Hide the axes
    ax.set_xticks([])
    ax.set_yticks([])

    # Add a grid
    for i in range(n + 1):
        lw = 2 if i % 3 == 0 else 0.5
        ax.axhline(i, color='black', lw=lw)
        ax.axvline(i, color='black', lw=lw)

    # Calculate text size based on n
    text_size = -5 / 11 * n + 155 / 11

    # Fill the cells with the matrix values and color based on differences
    for i in range(n):
        for j in range(n):
            value = matrix[i, j]
            color = colors[i, j]
            if missing_elements_indices[i, j]:
                # Highlight specific cells with blue background
                ax.add_patch(plt.Rectangle((j, n - i - 1), 1, 1, fill=True, color='blue', alpha=0.3))
            if value != 0:
                ax.text(j + 0.5, n - i - 0.5, f'{value:.2f}', ha='center', va='center', color=color, fontsize=text_size)

    ax.set_title(title)


def hints_matrix_norm(matrix, hints_matrix, hints_indices):
    # Set non-missing entries to the corresponding values in the initialization matrix
    norm = np.linalg.norm(matrix[hints_indices] - hints_matrix[hints_indices])
    return norm


def plot_2_metrix(matrix1, matrix2, missing_elements_indices, iteration_number):
    # Set a threshold for coloring based on absolute differences
    threshold = 0
    # Calculate absolute differences between matrix1 and matrix2
    rounded_matrix1 = np.round(matrix1, 2)
    rounded_matrix2 = np.round(matrix2, 2)

    diff_matrix = np.abs(rounded_matrix2 - rounded_matrix1)
    colors = np.where(diff_matrix > threshold, 'red', 'green')
    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the initial matrix with the specified threshold
    plot_sudoku(matrix1, colors, axs[0], "True_matrix", missing_elements_indices)

    # Plot the matrix after setting entries to zero with the specified threshold
    plot_sudoku(matrix2, colors, axs[1], "iteration_number: " + str(iteration_number), missing_elements_indices)

    plt.show()


def step_RRR(matrix, r, hints_matrix, hints_indices, beta):
    matrix_proj_1 = proj_1(matrix, hints_matrix, hints_indices)
    matrix_proj_2 = proj_2(matrix, r)
    PAPB_y = proj_1(matrix_proj_2, hints_matrix, hints_indices)
    new_matrix = matrix + beta * (2 * PAPB_y - matrix_proj_1 - matrix_proj_2)
    return new_matrix


def step_RRR_original(matrix, r, hints_matrix, hints_indices, beta):
    matrix_proj_2 = proj_2(matrix, r)
    matrix_proj_1 = proj_1(2 * matrix_proj_2 - matrix, hints_matrix, hints_indices)
    new_matrix = matrix + beta * (matrix_proj_1 - matrix_proj_2)
    return new_matrix


def step_AP(matrix, r, hints_matrix, hints_indices):
    matrix_proj_2 = proj_2(matrix, r)
    matrix_proj_1 = proj_1(matrix_proj_2, hints_matrix, hints_indices)
    return matrix_proj_1


def run_algorithm_for_matrix_completion(true_matrix, initial_matrix, hints_matrix, hints_indices, r, algo, beta=None,
                                        max_iter=1000, tolerance=1e-6):
    matrix = initial_matrix.copy()
    missing_elements_indices = ~hints_indices

    # Storage for plotting
    norm_diff_list = []
    norm_diff_list2 = []
    norm_diff_list3 = []
    norm_diff_min = 1000
    n_iter = -1

    if algo == "alternating_projections":
        for iteration in range(max_iter):
            # plot_2_metrix(true_matrix, matrix, missing_elements_indices, iteration)
            # if iteration % 100 == 0:
            #     print("iteration:", iteration)

            matrix = step_AP(matrix, r, hints_matrix, hints_indices)
            # print("y:", y[:3])

            # Calculate the norm difference between PB - PA
            matrix_proj_2 = proj_2(matrix, r)
            matrix_proj_1 = proj_1(matrix, hints_matrix, hints_indices)
            norm_diff = np.linalg.norm(matrix_proj_2 - matrix_proj_1)
            norm_diff3 = hints_matrix_norm(matrix, hints_matrix, hints_indices)
            norm_diff_list3.append(norm_diff3)

            # Store the norm difference for plotting
            norm_diff_list.append(norm_diff)

            norm_diff2 = np.linalg.norm(matrix - true_matrix)
            norm_diff_list2.append(norm_diff2)

            # if norm_diff_min >= norm_diff:
            #     print(iteration, norm_diff)
            #     norm_diff_min = norm_diff

            # Check convergence
            if norm_diff < tolerance:
                print(f"{algo} Converged in {iteration + 1} iterations.")
                n_iter = iteration + 1
                break

    elif algo == "RRR_algorithm":
        for iteration in range(max_iter):
            # plot_2_metrix(true_matrix, matrix, missing_elements_indices, iteration)
            # if iteration % 100 == 0:
            #     print("iteration:", iteration)
            # matrix = step_RRR(matrix, r, hints_matrix, hints_indices, beta)
            matrix = step_RRR_original(matrix, r, hints_matrix, hints_indices, beta)

            # Calculate the norm difference between PB - PA
            matrix_proj_2 = proj_2(matrix, r)
            matrix_proj_1 = proj_1(matrix, hints_matrix, hints_indices)
            norm_diff = np.linalg.norm(matrix_proj_2 - matrix_proj_1)
            norm_diff3 = hints_matrix_norm(matrix, hints_matrix, hints_indices)
            norm_diff_list3.append(norm_diff3)

            # Store the norm difference for plotting
            norm_diff_list.append(norm_diff)

            norm_diff2 = np.linalg.norm(matrix - true_matrix)
            norm_diff_list2.append(norm_diff2)

            # if norm_diff_min >= norm_diff:
            #     print(iteration, norm_diff)
            #     norm_diff_min = norm_diff
            # Check convergence
            if norm_diff < tolerance:
                print(f"{algo} Converged in {iteration + 1} iterations.")
                n_iter = iteration + 1
                break

    # Plot the norm difference over iterations
    plt.plot(norm_diff_list)
    plt.xlabel('Iteration')
    plt.ylabel('|PB(y, b) - PA(y, A)|')
    plt.title(f'Convergence of {algo} Algorithm, |PB(y, b) - PA(y, A)|')
    plt.show()

    # # Plot the norm difference over iterations
    # plt.plot(norm_diff_list3)
    # plt.xlabel('Iteration')
    # plt.ylabel('|hints_matrix - iter_matrix| at hints indexes')
    # plt.title(f'Convergence of {algo} Algorithm, |hints_matrix - iter_matrix| at hints indexes')
    # plt.show()

    # Plot the norm difference over iterations
    plt.plot(norm_diff_list2)
    plt.xlabel('Iteration')
    plt.ylabel('|true_matrix - iter_matrix|')
    plt.title(f'Convergence of {algo} Algorithm, |true_matrix - iter_matrix|')
    plt.show()

    return matrix, n_iter


def run_experiment(n, r, q, max_iter=1000, tolerance=1e-6, beta=0.5):
    np.random.seed(42)  # For reproducibility

    print(f"n = {n}, r = {r}, q = {q}")

    [true_matrix, initial_matrix, hints_matrix, hints_indices] = initialize_matrix(n, r, q, seed=42)
    missing_elements_indices = ~hints_indices

    # Alternating Projections
    result_AP, AP_n_iter = run_algorithm_for_matrix_completion(true_matrix, initial_matrix, hints_matrix, hints_indices,
                                                               r,
                                                               algo="alternating_projections", max_iter=max_iter,
                                                               tolerance=tolerance)

    plot_2_metrix(true_matrix, result_AP, missing_elements_indices, f"_END_ AP, for n = {n}, r = {r}, q = {q}")

    # Randomized Rounding Relaxation
    result_RRR, RRR_n_iter = run_algorithm_for_matrix_completion(true_matrix, initial_matrix, hints_matrix,
                                                                 hints_indices, r,
                                                                 algo="RRR_algorithm", beta=beta, max_iter=max_iter,
                                                                 tolerance=tolerance)
    result_RRR = proj_1(result_RRR, hints_matrix, hints_indices)

    plot_2_metrix(true_matrix, result_RRR, missing_elements_indices, f"_END_ RRR, for n = {n}, r = {r}, q = {q}")

    return AP_n_iter, RRR_n_iter


n_values = np.linspace(10, 150, 5)
r_values = np.linspace(10, 150, 5)
q_values = np.linspace(10, 20 ** 2, 5)

# Convert to integer arrays
n_values_int = n_values.astype(int)
r_values_int = r_values.astype(int)
q_values_int = q_values.astype(int)

n_values_int = [100]
r_values_int = [50]
q_values_int = [100]

#
# # Example usage of the loop
# for n in n_values_int:  # Set your desired values for n
#     for r in r_values_int:  # Set your desired values for r
#         for q in q_values_int:  # Set your desired values for q
#             run_experiment(n, r, q, max_iter=max_iter, tolerance=tolerance, beta=beta)


# log_file_path = r"n_r_q_n_iter1.txt"
# # Create a log file to write to
# log_file = open(log_file_path, "w")
#
# # Redirect sys.stdout to the custom Tee object
# sys.stdout = Tee(sys.stdout, log_file)

beta = 0.5
max_iter = 100000
tolerance = 1e-6
np.random.seed(42)  # For reproducibility

n_r_q_n_iter = []
#
# # Example usage of the loop
# for n in range(2, 20, 3):  # Set your desired values for n
#     for r in range(1, min(n, 20), 1):  # Set your desired values for r
#         for q in range(1, min((n - r) ** 2, n ** 2) + 1, 5):  # Set your desired values for q
#             AP_n_iter, RRR_n_iter = run_experiment(n, r, q, max_iter=max_iter, tolerance=tolerance, beta=beta)
#             n_r_q_n_iter.append([n, r, q, AP_n_iter, RRR_n_iter])


n = 20

r = 3

# nr = (n-r)**2
# print(f"(n-r)**2 = {nr} / {n*n}")

# q_values = range(1, n ** 2 - 1, 5)

q_values = range(1, (n-r) ** 2-1, 5)

# q_values = [340]
for q in q_values:  # Set your desired values for q
    AP_n_iter, RRR_n_iter = run_experiment(n, r, q, max_iter=max_iter, tolerance=tolerance, beta=beta)
    n_r_q_n_iter.append([n, r, q, AP_n_iter, RRR_n_iter])





