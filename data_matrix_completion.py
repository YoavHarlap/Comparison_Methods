
import matplotlib.pyplot as plt
import numpy as np

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


plt.xlabel('Trial Index')
plt.ylabel('Number of Iterations (log scale)')
plt.axhline(y=max_iter, color='black', linestyle='--', label='Maximal Number of Iterations')
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
    plt.semilogy(range(len(loaded_iteration_counts["RRR"])), loaded_iteration_counts[algo], markers[i], color=colors[i], label=f'{str_value}')


plt.xlabel('Trial Index')
plt.ylabel('Number of Iterations (log scale)')
plt.axhline(y=max_iter, color='black', linestyle='--', label='Maximal Number of Iterations')
plt.legend()
plt.grid(True, which="both", axis="both", ls="--")
# plt.xticks(np.arange(1, num_trials + 1, 1))  # Ensure x-ticks are at every trial number

plt.show()



# Plotting the percentages of successful convergence
convergence_percentages = {"AP":97.11,"RRR":99.95,"RAAR":98.01}

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