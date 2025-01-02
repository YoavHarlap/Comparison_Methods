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
    print("Original matrix rank:", matrix_rank(hints_matrix))
    if(r!=matrix_rank(true_matrix)):
        true_matrix = np.random.rand(n, r) @ np.random.rand(r, n)
        hints_matrix = true_matrix.copy()

        
    if(r!=matrix_rank(true_matrix)):
        true_matrix = np.random.rand(n, r) @ np.random.rand(r, n)
        hints_matrix = true_matrix.copy()
        
    

    # Set q random entries to NaN (missing entries)
    missing_entries = np.random.choice(n * n, q, replace=False)
    row_indices, col_indices = np.unravel_index(missing_entries, (n, n))
    # print(row_indices,col_indices)
    hints_matrix[row_indices, col_indices] = 0
    # print("Matrix rank after setting entries to zero:", matrix_rank(hints_matrix))
    hints_indices = np.ones_like(true_matrix, dtype=bool)
    hints_indices[row_indices, col_indices] = False

    # # Ensure the rank is still r
    # U, Sigma, Vt = svd(hints_matrix)
    # Sigma[r:] = 0  # Zero out singular values beyond rank r
    # new_matrix = U @ np.diag(Sigma) @ Vt
    # # print("Matrix rank after preserving rank:", matrix_rank(new_matrix))
    # initial_matrix = new_matrix
    
    
    initial_matrix = np.random.rand(n, r) @ np.random.rand(r, n)
    missing_elements_indices = ~hints_indices
    hints_matrix = true_matrix.copy()
    hints_matrix[missing_elements_indices]=0
    # plot_2_metrix(true_matrix, hints_matrix , missing_elements_indices, "given matrix")

    # plot_2_metrix(true_matrix, initial_matrix , missing_elements_indices, "initial matrix")
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
    threshold = 0.005
    # Calculate absolute differences between matrix1 and matrix2
    rounded_matrix1 = np.round(matrix1, 2)
    rounded_matrix2 = np.round(matrix2, 2)

    diff_matrix = np.abs(rounded_matrix2 - rounded_matrix1)
    colors = np.where(diff_matrix > threshold, 'red', 'green')
    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the initial matrix with the specified threshold
    plot_sudoku(matrix1, colors, axs[0], "True matrix", missing_elements_indices)

    # Plot the matrix after setting entries to zero with the specified threshold
    plot_sudoku(matrix2, colors, axs[1], "Iteration number: " + str(iteration_number), missing_elements_indices)

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

def step_RAAR(matrix, r, hints_matrix, hints_indices, beta):
    matrix_proj_2 = proj_2(matrix, r)
    PAPB_y = proj_1(2 * matrix_proj_2 - matrix, hints_matrix, hints_indices)
    result = beta * (matrix + PAPB_y) + (1 - 2 * beta) * matrix_proj_2
    return result

def step_HIO(matrix, r, hints_matrix, hints_indices, beta):
    matrix_proj_2 = proj_2(matrix, r)
    P_Ay = proj_1((1 + beta) * matrix_proj_2 - matrix, hints_matrix, hints_indices)
    result = matrix + P_Ay - beta * matrix_proj_2
    return result

# Update the run_algorithm_for_matrix_completion function to include these new algorithms

def run_algorithm_for_matrix_completion(true_matrix, initial_matrix, hints_matrix, hints_indices, r, algo, beta=None,
                                        max_iter=1000, tolerance=1e-6):
    matrix = initial_matrix.copy()
    missing_elements_indices = ~hints_indices

    norm_diff_list = []
    norm_diff_list2 = []
    norm_diff_list3 = []
    norm_diff_min = 1000
    n_iter = -1

    for iteration in range(max_iter):
        if algo == "Alternating Projections":
            matrix = step_AP(matrix, r, hints_matrix, hints_indices)
        elif algo == "RRR":
            matrix = step_RRR_original(matrix, r, hints_matrix, hints_indices, beta)
        elif algo == "RAAR":
            matrix = step_RAAR(matrix, r, hints_matrix, hints_indices, beta)
        elif algo == "HIO":
            matrix = step_HIO(matrix, r, hints_matrix, hints_indices, beta)
        else:
            raise ValueError("Unknown algorithm specified")

        matrix_proj_2 = proj_2(matrix, r)
        matrix_proj_1 = proj_1(matrix, hints_matrix, hints_indices)
        norm_diff = np.linalg.norm(matrix_proj_2 - matrix_proj_1)
        norm_diff3 = hints_matrix_norm(matrix, hints_matrix, hints_indices)
        norm_diff_list3.append(norm_diff3)

        norm_diff_list.append(norm_diff)
        norm_diff2 = np.linalg.norm(matrix - true_matrix)
        norm_diff_list2.append(norm_diff2)

        if norm_diff < tolerance:
            print(f"{algo} Converged in {iteration + 1} iterations.")
            n_iter = iteration + 1
            break
        # if iteration == 0 or iteration==32:
         # plot_2_metrix(true_matrix,  proj_1(matrix, hints_matrix, hints_indices), missing_elements_indices, iteration+1)
    # important skill do not delete
    # plt.plot(norm_diff_list)
    # plt.xlabel('Iteration')
    # plt.ylabel('|PB(y, b) - PA(y, A)|')
    # plt.title(f'Convergence of {algo} Algorithm, |PB(y, b) - PA(y, A)|')
    # plt.show()

    # plt.plot(norm_diff_list2)
    # plt.xlabel('Iteration')
    # plt.ylabel('|true_matrix - iter_matrix|')
    # plt.title(f'Convergence of {algo} Algorithm, |true_matrix - iter_matrix|')
    # plt.show()

    return matrix, n_iter


def run_experiment(n, r, q, algorithms,max_iter=1000, tolerance=1e-6, beta=0.5):
    np.random.seed(42)  # For reproducibility

    print(f"n = {n}, r = {r}, q = {q}")

    [true_matrix, initial_matrix, hints_matrix, hints_indices] = initialize_matrix(n, r, q, seed=42)
    missing_elements_indices = ~hints_indices

    results = {}

    for algo in algorithms:
        print(f"\nRunning {algo}...")
        result_matrix, n_iter = run_algorithm_for_matrix_completion(
            true_matrix, initial_matrix, hints_matrix, hints_indices,
            r, algo=algo, beta=beta, max_iter=max_iter, tolerance=tolerance
        )
        # plot_2_metrix(true_matrix,proj_1(result_matrix, hints_matrix, hints_indices), missing_elements_indices, f"{n_iter} - {algo} Done!, n = {n}, r = {r}, q = {q}")
        # plot_2_metrix(true_matrix,proj_1(result_matrix, hints_matrix, hints_indices), missing_elements_indices, f" {n_iter} --> END")

        results[algo] = n_iter

    return results




def plot_n_r_q_n_iter(n, r, q_values, algorithms, max_iter=100000, tolerance=1e-6, beta=0.5, seed=42):
    np.random.seed(seed)  # For reproducibility
    n_r_q_n_iter = []

    for q in q_values:  # Loop over q values and run experiments
        print(f"\nRunning experiment for q={q}...")
        experiment_results = run_experiment(n, r, q, algorithms, max_iter=max_iter, tolerance=tolerance, beta=beta)
        n_r_q_n_iter.append([n, r, q] + [experiment_results[algo] for algo in experiment_results])

    # Convert the list to a numpy array for easier manipulation
    n_r_q_n_iter = np.array(n_r_q_n_iter)

    # Extract n, r, q
    n_values = n_r_q_n_iter[:, 0]
    r_values = n_r_q_n_iter[:, 1]
    q_values = n_r_q_n_iter[:, 2]

    # Prepare a dictionary to store iteration counts for each algorithm
    algo_iters = {}

    # Dynamically extract the iteration counts based on the algorithms provided
    for i, algo in enumerate(algorithms):
        algo_iters[algo] = n_r_q_n_iter[:, 3 + i]

    # Filter out points where the number of iterations is -1 for each algorithm
    valid_indices = {}
    q_values_valid = {}
    for algo in algorithms:
        valid_indices[algo] = algo_iters[algo] != -1
        q_values_valid[algo] = q_values[valid_indices[algo]]
        algo_iters[algo] = algo_iters[algo][valid_indices[algo]]

    # Plotting using semilogy
    plt.figure(figsize=(10, 6))
    colors = ['blue', 'green', 'red', 'purple']  # Add more colors if needed
    markers = ['s-', 'o--', 'd-.', 'v:']  # Add more markers if needed

    for idx, algo in enumerate(algorithms):
        plt.semilogy(q_values_valid[algo], algo_iters[algo], markers[idx], color=colors[idx], label=f'{algo} Converged')

    # Adding labels and title
    plt.xlabel('q (Number of Missing Entries)')
    plt.ylabel('Number of Iterations (Log Scale)')
    plt.title(f'Convergence Comparison for n={n_values[0]}, r={r_values[0]}')
    plt.legend()
    plt.grid(True, which="both", ls="--")

    # Show plot
    plt.show()



def run_randomized_experiment(n, r, q, algorithms, num_trials=10, max_iter=1000, tolerance=1e-6, beta=0.5):
    convergence_results = {algo: 0 for algo in algorithms}

    for trial in range(num_trials):
        print(f"\nTrial {trial + 1}/{num_trials}")
        [true_matrix, initial_matrix, hints_matrix, hints_indices] = initialize_matrix(n, r, q, seed=trial)
        missing_elements_indices = ~hints_indices

        for algo in algorithms:
            print(f"Running {algo}...")
            _, n_iter = run_algorithm_for_matrix_completion(
                true_matrix, initial_matrix, hints_matrix, hints_indices,
                r, algo=algo, beta=beta, max_iter=max_iter, tolerance=tolerance
            )

            ######## If the algorithm converged, increase the count
            if n_iter != -1:
                convergence_results[algo] += 1

    # Calculate convergence percentage
    convergence_percentage = {algo: (convergence_results[algo] / num_trials) * 100 for algo in algorithms}

    # Plot the convergence percentage
    plt.figure(figsize=(10, 6))
    bars = plt.bar(convergence_percentage.keys(), convergence_percentage.values(), color='skyblue')
    plt.xlabel('Algorithms')
    plt.ylabel('Convergence Percentage (%)')
    plt.title(f'Convergence Percentage for n={n}, r={r}, q={q} over {num_trials} Trials')
    plt.ylim(0, 100)
    plt.grid(True, which="both", ls="--")

    # Add percentage value text on each bar
    max_height = max(convergence_percentage.values(), default=0)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + max_height * (-.05), f'{yval:.2f}%', ha='center',
                 va='bottom')

    plt.show()

    return convergence_percentage



def run_randomized_experiment_and_iteration_counts(n, r, q, algorithms, num_trials=10, max_iter=1000, tolerance=1e-6, beta=0.5):
    convergence_results = {algo: 0 for algo in algorithms}
    iteration_counts = {algo: [] for algo in algorithms}

    for trial in range(num_trials):
        print(f"\nTrial {trial + 1}/{num_trials}")
        [true_matrix, initial_matrix, hints_matrix, hints_indices] = initialize_matrix(n, r, q, seed=trial)
        missing_elements_indices = ~hints_indices

        for algo in algorithms:
            print(f"Running {algo}...")
            result_matrix, n_iter = run_algorithm_for_matrix_completion(
                true_matrix, initial_matrix, hints_matrix, hints_indices,
                r, algo=algo, beta=beta, max_iter=max_iter, tolerance=tolerance
            )

            # Append the number of iterations (or -1 if not converged)
            iteration_counts[algo].append(n_iter)
            
            # plot_2_metrix(true_matrix, result_matrix, missing_elements_indices, f"_END_ {algo}, for n = {n}, r = {r}, q = {q}")


            # If the algorithm converged, increase the count
            if n_iter != -1:
                convergence_results[algo] += 1

    # Calculate convergence percentage
    convergence_percentage = {algo: (convergence_results[algo] / num_trials) * 100 for algo in algorithms}

    # Plot the convergence percentage
    plt.figure(figsize=(10, 6))
    bars = plt.bar(convergence_percentage.keys(), convergence_percentage.values(), color='skyblue')
    plt.xlabel('Algorithms')
    plt.ylabel('Convergence Percentage (%)')
    plt.title(f'Convergence Percentage for n={n}, r={r}, q={q} over {num_trials} Trials')
    plt.ylim(0, 100)
    plt.grid(True, which="both", ls="--")

    # Add percentage value text on each bar
    max_height = max(convergence_percentage.values(), default=0)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + max_height * (-.05), f'{yval:.2f}%', ha='center',
                 va='bottom')

    plt.show()

    # Plot the iteration counts per trial using semilogy
    plt.figure(figsize=(12, 8))
    colors = ['blue', 'green', 'red', 'purple']  # Add more colors if needed
    markers = ['s-', 'o--', 'd-.', 'v:']  # Add more markers if needed

    for idx, algo in enumerate(algorithms):
        # Filter out non-converged trials
        converged_indices = [i for i, x in enumerate(iteration_counts[algo]) if x != -1]
        converged_iterations = [iteration_counts[algo][i] for i in converged_indices]

        plt.semilogy([i + 1 for i in converged_indices], converged_iterations, markers[idx], color=colors[idx],
                     label=f'{algo}')

    plt.xlabel('Trial Number')
    plt.ylabel('Number of Iterations (Log Scale)')
    plt.title(f'Iterations per Algorithm for n={n}, r={r}, q={q}')
    plt.legend()
    plt.grid(True, which="both", axis="both", ls="--")
    # plt.xticks(np.arange(1, num_trials + 1, 1))  # Ensure x-ticks are at every trial number

    plt.show()

    return iteration_counts, convergence_percentage

algorithms = ["Alternating Projections", "RRR", "RAAR"]

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


beta = 0.5
max_iter = 100000
tolerance = 1e-6
np.random.seed(42)  # For reproducibility

############################
n = 20
r = 5
q_values = range(1, (n-r) ** 2 - 1, 5)
# q_values = [25]


plot_n_r_q_n_iter(n, r, q_values, algorithms, max_iter=100000, tolerance=1e-6, beta=0.5)

#########################
# Example usage:
# n = 20
# r = 3
# q = 50
# algorithms = ["alternating_projections", "RRR_algorithm", "RAAR_algorithm"]
# num_trials = 5

# convergence_percentage = run_randomized_experiment(n, r, q, algorithms, num_trials=num_trials, max_iter=1000, tolerance=1e-6, beta=0.5)
# print("Convergence percentage results:", convergence_percentage)
########################


#########################
# Example usage:
# n = 20
# r = 3
# q = 50

# num_trials = 10000

# iteration_counts,convergence_percentage = run_randomized_experiment_and_iteration_counts(n, r, q, algorithms, num_trials=num_trials, max_iter=1000, tolerance=1e-6, beta=0.5)
# print("Convergence percentage results:", convergence_percentage)
########################

import winsound
# Beep sound
winsound.Beep(1001, 500)  # Frequency 1000 Hz, duration 500 ms





# Define the path
path = r"C:\Users\ASUS\Documents\code_images\overleaf_images\Numerical_experiments\Matrix_Completion\10000_trials"
import os

import json

# Ensure the directory exists
# os.makedirs(path, exist_ok=True)

# # Save the file as JSON
file_path = os.path.join(path, "iteration_counts.json")
# with open(file_path, "w") as file:
#     json.dump(convergence_data, file)

# print(f"File saved to: {file_path}")



# # Load the file back
with open(file_path, "r") as file:
    loaded_iteration_counts = json.load(file)

print("Loaded data:", loaded_iteration_counts)


# loaded_iteration_counts = {algo: counts[110:125] for algo, counts in loaded_iteration_counts.items()}

# Plot the iteration counts per trial using semilogy
plt.figure(figsize=(12, 8))
algorithms = ["Alternating Projections", "RRR", "RAAR"]
colors = ['blue', 'green', 'red', 'purple']  # Add more colors if needed
markers = ['s-', 'o--', 'd-.', 'v:']  # Add more markers if needed
max_iter = 100000


for i, algo in enumerate(algorithms):
    # Filter out non-converged trials
    loaded_iteration_counts[algo] = [max_iter if x == -1 else x for x in loaded_iteration_counts[algo]]


    # plt.semilogy([i + 1 for i in converged_indices], converged_iterations, markers[idx], color=colors[idx],
    #               label=f'{algo}')
    str_value = "AP" if algo == "Alternating Projections" else algo
    plt.semilogy(range(len(loaded_iteration_counts["RRR"])), loaded_iteration_counts[algo], markers[i], color=colors[i], label=f'{str_value}',linestyle='None')


plt.axhline(y=max_iter, color='black', linestyle='--', label='maximal number of iterations')

plt.xlabel('trial index')
plt.ylabel('number of iterations (log scale)')
# plt.title(f'Iterations per Algorithm for n={n}, r={r}, q={q}')
plt.legend()
plt.grid(True, which="both", axis="both", ls="--")
# plt.xticks(np.arange(1, num_trials + 1, 1))  # Ensure x-ticks are at every trial number

plt.show()


loaded_iteration_counts = {algo: counts[110:125] for algo, counts in loaded_iteration_counts.items()}

# Plot the iteration counts per trial using semilogy
plt.figure(figsize=(12, 8))
algorithms = ["Alternating Projections", "RRR", "RAAR"]
colors = ['blue', 'green', 'red', 'purple']  # Add more colors if needed
markers = ['s-', 'o--', 'd-.', 'v:']  # Add more markers if needed
max_iter = 100000


for i, algo in enumerate(algorithms):
    # Filter out non-converged trials
    loaded_iteration_counts[algo] = [max_iter if x == -1 else x for x in loaded_iteration_counts[algo]]


    # plt.semilogy([i + 1 for i in converged_indices], converged_iterations, markers[idx], color=colors[idx],
    #               label=f'{algo}')
    str_value = "AP" if algo == "Alternating Projections" else algo
    plt.semilogy(range(len(loaded_iteration_counts["RRR"])), loaded_iteration_counts[algo], markers[i], color=colors[i], label=f'{str_value}',linestyle='None')


plt.axhline(y=max_iter, color='black', linestyle='--', label='maximal number of iterations')

plt.xlabel('trial index')
plt.ylabel('number of iterations (log scale)')
# plt.title(f'Iterations per Algorithm for n={n}, r={r}, q={q}')
plt.legend()
plt.grid(True, which="both", axis="both", ls="--")
# plt.xticks(np.arange(1, num_trials + 1, 1))  # Ensure x-ticks are at every trial number

plt.show()

