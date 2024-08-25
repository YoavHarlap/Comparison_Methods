import matplotlib.pyplot as plt
import numpy as np


def run_randomized_experiment_and_iteration_counts(n, r, q, algorithms, num_trials=10, max_iter=1000, tolerance=1e-6, beta=0.5):
    convergence_results = {algo: 0 for algo in algorithms}
    iteration_counts = {algo: [] for algo in algorithms}

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

            # Append the number of iterations (or -1 if not converged)
            iteration_counts[algo].append(n_iter)

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
    plt.grid(True, which="both", ls="--")
    plt.show()

    return iteration_counts, convergence_percentage
