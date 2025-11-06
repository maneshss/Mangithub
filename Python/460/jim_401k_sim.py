#!/usr/bin/env python3
"""
Monte Carlo simulation for Jim's 401(k) balance at age 60.
Assumptions:
- Contributions occur at the start of each year (then compound for that year).
- Annual returns are independent normal draws for each fund.
- Salary raises are independent normal draws; a floor is applied to avoid extreme negatives.
- No fees, taxes, withdrawals, or additional employer matching rules beyond 5% match (total 15% of salary).
"""

import numpy as np


def run_simulation(N=100000, seed=42):
    np.random.seed(seed)

    start_age = 24
    end_age = 60
    years = end_age - start_age
    start_salary = 55000.0
    contribution_rate = 0.15

    # Allocation weights
    wA, wB, wC = 0.50, 0.25, 0.25

    # Annual return distributions (decimal)
    muA, sdA = 0.0587, 0.0925
    muB, sdB = 0.0715, 0.1488
    muC, sdC = 0.0839, 0.1641

    # Salary growth distribution (decimal)
    mu_g, sd_g = 0.0304, 0.0089

    # Practical clamps
    MIN_RETURN = -0.999  # cannot lose more than 100%
    MIN_GROWTH = -0.10  # limit a single-year pay cut to -10%

    def simulate_once():
        salary = start_salary
        balance = 0.0
        for _ in range(years):
            rA = max(np.random.normal(muA, sdA), MIN_RETURN)
            rB = max(np.random.normal(muB, sdB), MIN_RETURN)
            rC = max(np.random.normal(muC, sdC), MIN_RETURN)
            rP = wA * rA + wB * rB + wC * rC

            contrib = contribution_rate * salary
            balance = (balance + contrib) * (1.0 + rP)

            g = max(np.random.normal(mu_g, sd_g), MIN_GROWTH)
            salary *= 1.0 + g
        return balance

    balances = np.fromiter((simulate_once() for _ in range(N)), dtype=float, count=N)
    prob_over_1m = float(np.mean(balances > 1_000_000))
    stats = {
        "N": N,
        "Years": years,
        "Mean": float(np.mean(balances)),
        "Median": float(np.median(balances)),
        "StdDev": float(np.std(balances, ddof=1)),
        "P5": float(np.percentile(balances, 5)),
        "P25": float(np.percentile(balances, 25)),
        "P75": float(np.percentile(balances, 75)),
        "P95": float(np.percentile(balances, 95)),
        "Prob_Over_1M": prob_over_1m,
    }
    return balances, stats


if __name__ == "__main__":
    balances, stats = run_simulation(N=10000, seed=42)
    print("Summary statistics:")
    for k, v in stats.items():
        if k in {"N", "Years"}:
            print(f"{k}: {v}")
        elif k == "Prob_Over_1M":
            print(f"{k}: {v:.4%}")
        else:
            print(f"{k}: ${v:,.2f}")
