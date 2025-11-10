# assignment_heuristic_metaheuristic.py
import math, random, time, csv, os
from collections import defaultdict
from typing import List, Tuple, Dict, Set

try:
    import pandas as pd
    import numpy as np
except ImportError:
    raise SystemExit("Please `pip install pandas numpy` first")

# === CONFIG ===
EXCEL_PATH = "/Users/maneshss/Desktop/Study/Project/Mangithub/Python/460/Problems4.xlsx"
SHEET_NAME = 0  # or the sheet name with the 20x20 matrix
N_EMP, N_TASK = 20, 20
SEED = 42

# Metaheuristic settings
ITERATIONS = 1000  # total GRASP/SA iterations
RCL_ALPHA = 0.25  # GRASP: 0 (pure greedy) .. 1 (pure random)
SA_T0 = 5.0  # initial temperature
SA_TMIN = 1e-3
SA_COOL = 0.995  # cooling rate per move
STALL_LIMIT = 250  # stop early if no new global best for many iterations
TABU_CAP = 5000  # remember this many solutions to avoid trivial repeats

random.seed(SEED)
np_random = np.random.default_rng(SEED)


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def load_cost_matrix(path: str, sheet=0) -> np.ndarray:
    """
    Reads a 20x20 matrix from Excel.
    If your file has headers or extra columns, ensure the first 20x20 block is the matrix.
    """
    df = pd.read_excel(path, sheet_name=sheet, header=None)
    # Find the first 20x20 numeric block.
    # If the sheet is *exactly* 20x20, this just works.
    arr = df.select_dtypes(include=[float, int]).to_numpy()
    if arr.shape[0] >= N_EMP and arr.shape[1] >= N_TASK:
        mat = arr[:N_EMP, :N_TASK].astype(float)
        return mat
    raise ValueError(
        f"Expected at least {N_EMP}x{N_TASK} numeric block; got {arr.shape}"
    )


def cost_of(sol: List[int], C: np.ndarray) -> float:
    # sol[i] = task assigned to employee i
    return float(sum(C[i, sol[i]] for i in range(len(sol))))


def is_feasible(sol: List[int]) -> bool:
    # one task per employee, each task used exactly once
    return len(set(sol)) == len(sol) == N_TASK


def vec_to_key(sol: List[int]) -> Tuple[int, ...]:
    return tuple(sol)


# ------------------------------------------------------------
# Construction Heuristic (Greedy with RCL for GRASP)
# ------------------------------------------------------------
def greedy_construction(C: np.ndarray, alpha: float = 0.0) -> List[int]:
    """
    Assigns one task to each employee.
    alpha=0.0 => pure greedy; alpha>0 => GRASP randomized restricted candidate list.
    """
    n = C.shape[0]
    available_tasks: Set[int] = set(range(n))
    assignment = [-1] * n

    # Process employees in an order (optionally shuffled to diversify)
    employees = list(range(n))
    # A mild randomization to avoid determinism when alpha>0
    random.shuffle(employees)

    for i in employees:
        # build candidate list for employee i among remaining tasks
        costs = [(j, C[i, j]) for j in available_tasks]
        costs.sort(key=lambda x: x[1])
        if alpha <= 1e-12:
            # pure greedy: pick the cheapest
            j = costs[0][0]
        else:
            # GRASP RCL: choose among top-k determined by threshold
            vals = [v for (_, v) in costs]
            cmin, cmax = vals[0], vals[-1]
            threshold = cmin + alpha * (cmax - cmin)
            rcl = [j for (j, v) in costs if v <= threshold]
            j = random.choice(rcl)
        assignment[i] = j
        available_tasks.remove(j)

    # The employee order was shuffled; we return assignment indexed by true employee id
    return assignment


# ------------------------------------------------------------
# Local Search Neighborhood: 2-swap (swap tasks of two employees)
# ------------------------------------------------------------
def best_improving_swap(sol: List[int], C: np.ndarray) -> Tuple[bool, List[int], float]:
    """
    Tries all pairwise swaps (i,k). Returns first improving move (first-improvement).
    """
    n = len(sol)
    current = cost_of(sol, C)
    for i in range(n - 1):
        ti = sol[i]
        ci = C[i, ti]
        for k in range(i + 1, n):
            tk = sol[k]
            if tk == ti:
                continue
            # incremental delta
            delta = (C[i, tk] + C[k, ti]) - (ci + C[k, tk])
            if delta < -1e-12:
                new_sol = sol.copy()
                new_sol[i], new_sol[k] = new_sol[k], new_sol[i]
                return True, new_sol, current + delta
    return False, sol, current


def sa_neighbor(sol: List[int]) -> List[int]:
    """Random 2-swap move"""
    i, k = random.sample(range(len(sol)), 2)
    new_sol = sol.copy()
    new_sol[i], new_sol[k] = new_sol[k], new_sol[i]
    return new_sol


# ------------------------------------------------------------
# Metaheuristic: GRASP + Simulated Annealing + Local Search Polishing
# ------------------------------------------------------------
def grasp_sa(
    C: np.ndarray,
    iterations: int = ITERATIONS,
    alpha: float = RCL_ALPHA,
    t0: float = SA_T0,
    tmin: float = SA_TMIN,
    cool: float = SA_COOL,
    stall_limit: int = STALL_LIMIT,
    tabu_cap: int = TABU_CAP,
):
    """
    Each iteration:
      1) Construction heuristic (GRASP) -> feasible solution
      2) Simulated Annealing walk over 2-swap neighbors
      3) First-improvement local search polish
    Keeps global best.
    """
    n = C.shape[0]
    best_sol = None
    best_cost = math.inf
    seen: Set[Tuple[int, ...]] = set()
    no_improve = 0

    for it in range(1, iterations + 1):
        # 1) Construction (uses heuristic as subroutine)
        s = greedy_construction(C, alpha=alpha)
        # guard (should always be true)
        if not is_feasible(s):
            # as fallback, repair by permuting until unique
            # (rare with our construction)
            s = list(range(n))
            random.shuffle(s)

        # Avoid trivial repeats
        key = vec_to_key(s)
        if key in seen:
            # small random shake
            s = sa_neighbor(s)
        seen.add(vec_to_key(s))
        if len(seen) > tabu_cap:
            # keep memory bounded
            for _ in range(len(seen) - tabu_cap):
                seen.pop()

        # 2) Simulated Annealing on 2-swap neighborhood
        t = t0
        curr = s
        curr_cost = cost_of(curr, C)
        while t > tmin:
            cand = sa_neighbor(curr)
            cand_cost = cost_of(cand, C)
            d = cand_cost - curr_cost
            if d <= 0 or random.random() < math.exp(-d / t):
                curr, curr_cost = cand, cand_cost
            t *= cool

        # 3) First-improvement local search polish
        improved = True
        while improved:
            improved, curr, curr_cost = best_improving_swap(curr, C)

        # Update global best
        if curr_cost + 1e-9 < best_cost:
            best_sol, best_cost = curr, curr_cost
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= stall_limit:
                break

        if it % 50 == 0:
            print(f"[iter {it}] best={best_cost:.3f}")

    return best_sol, best_cost


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    print("Loading cost matrix...")
    C = load_cost_matrix(EXCEL_PATH, SHEET_NAME)
    assert C.shape == (N_EMP, N_TASK), f"Got {C.shape}, expected {(N_EMP, N_TASK)}"

    start = time.time()
    best_sol, best_cost = grasp_sa(C)
    elapsed = time.time() - start

    # Reporting
    print("\n=== BEST SOLUTION FOUND ===")
    print(f"Total cost: {best_cost:.3f}")
    print(f"Time: {elapsed:.2f}s")
    print("Employee -> Task assignments (1-based indexing):")
    for i, t in enumerate(best_sol, 1):
        print(f"Employee {i:2d} -> Task {t + 1:2d} (c_ij={C[i - 1, t]:.2f})")

    # Save CSV
    out_path = os.path.join(os.getcwd(), "assignment_solution.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Employee", "Task", "Cost_ij"])
        for i, t in enumerate(best_sol):
            w.writerow([i + 1, t + 1, C[i, t]])
        w.writerow([])
        w.writerow(["Total Cost", best_cost])
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
