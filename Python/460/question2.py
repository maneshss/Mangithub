"""


Objective:
Estimate:
1. Probability of hitting each dartboard section.
2. Long-term average score per dart.

Dartboard Details:
-------------------------------------------------
Circle Color | Radius | Points (if within)
Black         | 0.1    | 5
Purple        | 0.3    | 4
Green         | 0.5    | 3
Blue          | 0.7    | 2
Red           | 1.0    | 1
Outside Red   | >1.0   | 0
-------------------------------------------------
"""

import numpy as np
import pandas as pd

# -------------------------------
# Simulation Parameters
# -------------------------------
N = 20000  # Number of dart throws (adjust between 400–20000)
mu = 0.0  # Mean for both x and y
sigma = 0.3  # Standard deviation for x and y
seed = 2025  # Random seed for reproducibility

# Circle radii and corresponding points
radii = [0.1, 0.3, 0.5, 0.7, 1.0]  # boundaries
points = [5, 4, 3, 2, 1]  # points for each section
labels = [
    "Black (≤0.1)",
    "Purple (0.1, 0.3]",
    "Green (0.3, 0.5]",
    "Blue (0.5, 0.7]",
    "Red (0.7, 1.0]",
    "Outside (>1.0)",
]

# -------------------------------
# Step 1: Generate random dart hits
# -------------------------------
rng = np.random.default_rng(seed)
x = rng.normal(mu, sigma, N)
y = rng.normal(mu, sigma, N)

# Compute distance from center (r)
r = np.sqrt(x**2 + y**2)

# -------------------------------
# Step 2: Determine which section each dart hits
# -------------------------------
# np.digitize returns the index of the radius interval (0..len(radii))
sections = np.digitize(r, radii, right=True)

# Assign points based on section
score_map = {i: (points[i] if i < len(points) else 0) for i in range(len(points) + 1)}
scores = np.vectorize(score_map.get)(sections)

# -------------------------------
# Step 3: Compute probabilities
# -------------------------------
counts = pd.value_counts(sections, sort=False).reindex(
    range(len(points) + 1), fill_value=0
)
probabilities = counts / N

# Create results DataFrame
results = pd.DataFrame(
    {
        "Section": labels,
        "Count": counts.values,
        "Probability": probabilities.values,
        "Points (if landed)": [5, 4, 3, 2, 1, 0],
    }
)

# -------------------------------
# Step 4: Long-term average score
# -------------------------------
average_score = scores.mean()

# -------------------------------
# Step 5: Display Results
# -------------------------------
print("=== Dartboard Monte Carlo Simulation ===")
print(results)
print(f"\nLong-term Average Score per Shot: {average_score:.4f}")

# -------------------------------
# Optional: Save to Excel
# -------------------------------
results.to_excel("dartboard_monte_carlo_results.xlsx", index=False)
print("\nResults saved to 'dartboard_monte_carlo_results.xlsx'")
