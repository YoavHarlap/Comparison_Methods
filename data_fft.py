

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, ifft

# Plotting the percentages of successful convergence
convergence_percentages = {"AP":45.53,"RRR":99.99,"RAAR":54.67,"HIO":98.24}

plt.figure(figsize=(10, 6))
# bars = plt.bar(convergence_percentages.keys(), convergence_percentages.values(), color=['blue', 'green', 'red', 'purple'])
bars = plt.bar(
    convergence_percentages.keys(),
    convergence_percentages.values(),
    color=['skyblue', 'lightgreen', 'salmon', 'plum']
)

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


# Define the path
path = r"C:\Users\ASUS\Documents\code_images\overleaf_images\Numerical_experiments\dft_case\generic2"
import os

import json

# # Ensure the directory exists
# os.makedirs(path, exist_ok=True)

# # Save the file as JSON
file_path = os.path.join(path, "iteration_counts.json")
# with open(file_path, "w") as file:
#     json.dump(convergence_data, file)
#
# print(f"File saved to: {file_path}")



# # Load the file back
with open(file_path, "r") as file:
    loaded_iteration_counts = json.load(file)

print("Loaded data:", loaded_iteration_counts)


# loaded_iteration_counts = {algo: counts[100:125] for algo, counts in loaded_iteration_counts.items()}



# Logarithmic convergence plot
colors = ['blue', 'green', 'red', 'purple']
markers = ['s-', 'o--', 'd-.', 'v:']
max_iter = 10000
algorithms = ["AP", "RRR", "RAAR", "HIO"]

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
