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
    # P_Ay = PA(y, A)
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
        if algo == "Alternating Projections":
            y = step_AP(A, b, y)
        elif algo == "RRR":
            y = step_RRR(A, b, y, beta)
        elif algo == "RAAR":
            y = step_RAAR(A, b, y, beta)
        elif algo == "HIO":
            y = step_HIO(A, b, y, beta)
        else:
            raise ValueError(f"Unknown algorithm: {algo} :) ")

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


max_iter = 10000
tolerance = 1e-6

# m_array = [25,26,27,28,29,30,31,32,33,34,35,36]
# m_array = [25,26,27]
m_array = [25]

# n_array = [7, 8, 9,10,11,12,13]
# n_array = [17, 18, 19,20,21,22,23]
n_array = [10]


betas = [0.5]



algorithms = ["Alternating Projections", "RRR", "RAAR", "HIO"]
convergence_data = {algo: [] for algo in algorithms}
index_of_operation = 0

for m in m_array:
    print(m)
    print(m)
    print(m)
    print(m)
    print(m)
    print(m)
    print(m)

    for n in n_array:
        for beta in betas:
            np.random.seed(42)
            print(f"m = {m}, n = {n}, beta = {beta}")

            # Initialize the problem
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

            for algo in algorithms:
                print(f"Running {algo}...")

                result, converged = run_algorithm(A, b, y_initial, algo=algo, beta=beta, max_iter=max_iter, tolerance=tolerance)
                
                # Store convergence data
                convergence_data[algo].append(converged if converged != -1 else None)
                
                # Plot the result for this algorithm
            #     plt.plot(abs(PA(result, A)), label=f'result_{algo}')

            # # Plot the observed data b
            # plt.plot(b, label='b')
            # plt.xlabel('Element')
            # plt.ylabel('Value')
            # plt.title(f'Plot of Terms for m={m}, n={n}, beta={beta}')
            # plt.legend()
            # plt.show()

            index_of_operation += 1

# Convergence plots
for algo in algorithms:
    plt.semilogy(range(index_of_operation), convergence_data[algo], label=f'{algo} Converged')

plt.xlabel('Scenario')
plt.ylabel('Convergence - num of iterations')
plt.title('Convergence Plot')
plt.legend()
plt.show()

# Logarithmic convergence plot
colors = ['blue', 'green', 'red', 'purple']
markers = ['s-', 'o--', 'd-.', 'v:']
for i, algo in enumerate(algorithms):
    plt.semilogy(range(index_of_operation), convergence_data[algo], markers[i], color=colors[i], label=f'{algo} Converged')

plt.xlabel('Index of Operation')
plt.ylabel('Converged Value (log scale)')
plt.legend()
plt.title('Logarithmic Plot of Converged Values')
plt.grid(True, which="both", ls="--")
plt.show()


############################## make percentages
# Experiment parameters
max_iter = 1000
tolerance = 1e-4
# m_array = [25, 26, 27]
# n_array = [10]
betas = [0.5]
algorithms = ["Alternating Projections", "RRR", "RAAR", "HIO"]

# Initialize data structures for storing convergence information
convergence_data = {algo: [] for algo in algorithms}
convergence_count = {algo: 0 for algo in algorithms}
index_of_operation = 0
total_trials = 1

for trial in range(total_trials):
    print(f"\nTrial {trial + 1}/{total_trials}")

    # Randomize input for each trial
    np.random.seed(trial)
    m = 25
    n = 8
    beta = 0.5
    
    # print(f"m = {m}, n = {n}, beta = {beta}")

    # Initialize the problem
    A = np.random.randn(m, n) + 1j * np.random.randn(m, n)
    x = np.random.randn(n) + 1j * np.random.randn(n)
    b = np.abs(np.dot(A, x))
    y_initial = np.random.randn(m) + 1j * np.random.randn(m)

    for algo in algorithms:
        print(f"Running {algo}...")

        result, converged = run_algorithm(A, b, y_initial, algo=algo, beta=beta, max_iter=max_iter, tolerance=tolerance)
        
        # Store convergence data
        convergence_data[algo].append(converged if converged != -1 else None)
        
        # Count successful convergence
        if converged != -1:
            convergence_count[algo] += 1
        
        # Plot the result for this algorithm
    #     plt.plot(abs(PA(result, A)), label=f'result_{algo}')

    # # Plot the observed data b
    # plt.plot(b, label='b')
    # plt.xlabel('Element')
    # plt.ylabel('Value')
    # plt.title(f'Plot of Terms for m={m}, n={n}, beta={beta}')
    # plt.legend()
    # plt.show()

    index_of_operation += 1

# Convergence plots
for algo in algorithms:
    plt.semilogy(range(index_of_operation), convergence_data[algo], label=f'{algo} Converged')

plt.xlabel('Scenario')
plt.ylabel('Convergence - num of iterations')
plt.title('Convergence Plot')
plt.legend()
plt.show()

# Logarithmic convergence plot
colors = ['blue', 'green', 'red', 'purple']
markers = ['s-', 'o--', 'd-.', 'v:']
for i, algo in enumerate(algorithms):
    plt.semilogy(range(index_of_operation), convergence_data[algo], markers[i], color=colors[i], label=f'{algo} Converged')

plt.xlabel('Index of Operation')
plt.ylabel('Converged Value (log scale)')
plt.legend()
plt.title('Logarithmic Plot of Converged Values')
plt.grid(True, which="both", ls="--")
plt.show()

# Plotting the percentages of successful convergence
convergence_percentages = {algo: (convergence_count[algo] / total_trials) * 100 for algo in algorithms}

plt.figure(figsize=(10, 6))
bars = plt.bar(convergence_percentages.keys(), convergence_percentages.values(), color=['blue', 'green', 'red', 'purple'])
plt.xlabel('Algorithm')
plt.ylabel('Convergence Percentage (%)')
plt.title('Percentage of Successful Convergences for Each Algorithm')
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
path = r"C:\Users\ASUS\Documents\code_images\overleaf_images\Numerical_experiments\phase_complex"
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

# print("Loaded data:", loaded_iteration_counts)


# loaded_iteration_counts = {algo: counts[110:125] for algo, counts in loaded_iteration_counts.items()}
print("Loaded data:", loaded_iteration_counts)

# Plot the iteration counts per trial using semilogy
plt.figure(figsize=(12, 8))
algorithms = ["Alternating Projections", "RRR", "RAAR","HIO"]
colors = ['blue', 'green', 'red', 'purple']  # Add more colors if needed
markers = ['s-', 'o--', 'd-.', 'v:']  # Add more markers if needed
max_iter = 10000


for i, algo in enumerate(algorithms):
    # Filter out non-converged trials
    loaded_iteration_counts[algo] = [max_iter if x == None else x for x in loaded_iteration_counts[algo]]


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
print("Loaded data:", loaded_iteration_counts)

# Plot the iteration counts per trial using semilogy
plt.figure(figsize=(12, 8))



for i, algo in enumerate(algorithms):
    # Filter out non-converged trials
    loaded_iteration_counts[algo] = [max_iter if x == None else x for x in loaded_iteration_counts[algo]]


    # plt.semilogy([i + 1 for i in converged_indices], converged_iterations, markers[idx], color=colors[idx],
    #               label=f'{algo}')
    str_value = "AP" if algo == "Alternating Projections" else algo
    plt.semilogy(range(len(loaded_iteration_counts["RRR"])), loaded_iteration_counts[algo], markers[i], color=colors[i], label=f'{str_value}')


plt.axhline(y=max_iter, color='black', linestyle='--', label='maximal number of iterations')

plt.xlabel('trial index')
plt.ylabel('number of iterations (log scale)')
# plt.title(f'Iterations per Algorithm for n={n}, r={r}, q={q}')
plt.legend()
plt.grid(True, which="both", axis="both", ls="--")
# plt.xticks(np.arange(1, num_trials + 1, 1))  # Ensure x-ticks are at every trial number

plt.show()
