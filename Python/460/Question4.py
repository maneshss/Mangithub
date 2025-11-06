import pandas as pd
import numpy as np
import random

# === Step 1: Load cost matrix from Excel ===
# Ensure the Excel file has a 20x20 cost matrix starting from A1
df = pd.read_excel(
    "/Users/maneshss/Desktop/Study/Project/Mangithub/Python/460/Problems4.xlsx"
)

cost_matrix = df.values
n = cost_matrix.shape[0]


# === Step 2: Construction Heuristic (Greedy Assignment) ===
def greedy_construction(cost_matrix):
    n = cost_matrix.shape[0]
    assigned_employees = set()
    assigned_tasks = set()
    assignment = [-1] * n
    total_cost = 0

    while len(assigned_employees) < n:
        min_cost = float("inf")
        best_pair = None
        for i in range(n):
            if i in assigned_employees:
                continue
            for j in range(n):
                if j in assigned_tasks:
                    continue
                if cost_matrix[i][j] < min_cost:
                    min_cost = cost_matrix[i][j]
                    best_pair = (i, j)
        if best_pair:
            i, j = best_pair
            assigned_employees.add(i)
            assigned_tasks.add(j)
            assignment[i] = j
            total_cost += cost_matrix[i][j]
    return assignment, total_cost


# === Step 3: Destruction–Reconstruction Metaheuristic ===
def destruction_reconstruction(
    cost_matrix, initial_assign, iterations=500, destroy_rate=0.2
):
    n = cost_matrix.shape[0]
    best_assign = initial_assign.copy()
    best_cost = sum(cost_matrix[i][best_assign[i]] for i in range(n))

    for it in range(iterations):
        # Step 1: Destruction
        new_assign = best_assign.copy()
        num_destroy = int(n * destroy_rate)
        destroyed = random.sample(range(n), num_destroy)
        unassigned_tasks = set(new_assign[i] for i in destroyed)
        for i in destroyed:
            new_assign[i] = -1  # unassign

        # Step 2: Reconstruction
        available_tasks = list(set(range(n)) - set(t for t in new_assign if t != -1))
        for i in destroyed:
            # choose task with lowest cost among available ones
            best_task = min(available_tasks, key=lambda j: cost_matrix[i][j])
            new_assign[i] = best_task
            available_tasks.remove(best_task)

        # Step 3: Evaluate new cost
        new_cost = sum(cost_matrix[i][new_assign[i]] for i in range(n))

        # Step 4: Acceptance criterion
        if new_cost < best_cost:
            best_assign, best_cost = new_assign, new_cost

    return best_assign, best_cost


# === Step 4: Run both heuristics ===
initial_assign, initial_cost = greedy_construction(cost_matrix)
improved_assign, improved_cost = destruction_reconstruction(cost_matrix, initial_assign)

# === Step 5: Print results ===
print("Initial greedy solution cost:", initial_cost)
print("Improved metaheuristic solution cost:", improved_cost)
print(
    "Improvement: {:.2f}%".format(100 * (initial_cost - improved_cost) / initial_cost)
)

# === Step 6: Show final assignments ===
result = pd.DataFrame(
    {
        "Employee": range(1, n + 1),
        "Assigned_Task": [a + 1 for a in improved_assign],
        "Cost": [cost_matrix[i][improved_assign[i]] for i in range(n)],
    }
)
print("\nFinal assignment:\n", result)
