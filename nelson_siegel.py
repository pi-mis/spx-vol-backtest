"""
nelson_siegel.py
================
Diebold-Li (2006) Nelson-Siegel model applied to the VIX term structure.

Model:
    IV(τ) = β₁  +  β₂ · L(τ)  +  β₃ · C(τ)

where:
    L(τ) = (1 − e^{−λτ}) / (λτ)        "slope" loading
    C(τ) = L(τ) − e^{−λτ}               "curvature" loading
    τ     = time to maturity in years
    λ     = decay parameter (fixed per day, or optimised)

Factor interpretation:
    β₁  → long-run vol level        (parallel shift of the curve)
    β₂  → short-term slope          (negative = upward sloping curve)
    β₃  → medium-term hump/curvature

Input: today's observed IV at 4 maturities
    τ = [9/365, 30/365, 90/365, 180/365]
    IV = [VIX9D, VIX, VIX3M, VIX6M]

Output: β₁, β₂, β₃ extracted via OLS (model is linear in the βs).
        λ is optimised once per fitting call (Brent's method on RMSE).
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.linalg import lstsq
from typing import NamedTuple


# ── Factor loadings ───────────────────────────────────────────────────────────

def _L(tau: np.ndarray, lam: float) -> np.ndarray:
    """Slope loading L(τ) — handles τ → 0 safely."""
    lt = lam * tau
    return np.where(lt < 1e-9, 1.0, (1.0 - np.exp(-lt)) / lt)


def _C(tau: np.ndarray, lam: float) -> np.ndarray:
    """Curvature loading C(τ) = L(τ) − e^{−λτ}."""
    return _L(tau, lam) - np.exp(-lam * tau)


def _X(tau: np.ndarray, lam: float) -> np.ndarray:
    """Design matrix [1, L(τ), C(τ)], shape (n, 3)."""
    return np.column_stack([np.ones_like(tau), _L(tau, lam), _C(tau, lam)])


# ── Single cross-section fit ──────────────────────────────────────────────────

class NSFit(NamedTuple):
    beta1:     float
    beta2:     float
    beta3:     float
    lam:       float
    r2:        float
    rmse:      float   # vol points
    n_points:  int


def fit_ns(
    tau: np.ndarray,
    iv:  np.ndarray,
    lam: float,
) -> NSFit:
    """OLS fit of Nelson-Siegel for a single cross-section."""
    X = _X(tau, lam)
    betas, _, _, _ = lstsq(X, iv)
    iv_hat = X @ betas
    ss_res = float(np.sum((iv - iv_hat) ** 2))
    ss_tot = float(np.sum((iv - iv.mean()) ** 2))
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan
    rmse   = float(np.sqrt(ss_res / len(iv)))
    return NSFit(
        beta1=float(betas[0]),
        beta2=float(betas[1]),
        beta3=float(betas[2]),
        lam=lam, r2=r2, rmse=rmse, n_points=len(iv),
    )


def fit_ns_optimal(
    tau:    np.ndarray,
    iv:     np.ndarray,
    bounds: tuple[float, float] = (0.5, 50.0),
) -> NSFit:
    """Find λ that minimises RMSE via Brent, then OLS fit."""
    res = minimize_scalar(
        lambda l: fit_ns(tau, iv, l).rmse,
        bounds=bounds, method="bounded",
    )
    return fit_ns(tau, iv, res.x)


# ── Reconstruct IV at any maturity ────────────────────────────────────────────

def predict_iv(tau: float | np.ndarray, ns: NSFit) -> np.ndarray:
    """Reconstruct IV at arbitrary maturity/maturities from a fitted NSFit."""
    tau_arr = np.atleast_1d(tau)
    return (_X(tau_arr, ns.lam) @
            np.array([ns.beta1, ns.beta2, ns.beta3]))


# ── Rolling panel fit ─────────────────────────────────────────────────────────

def fit_ns_panel(
    term_structure: pd.DataFrame,
    maturities:     dict[str, float],
    verbose:        bool = True,
) -> pd.DataFrame:
    """
    Fit Nelson-Siegel on every row of `term_structure`.

    Parameters
    ----------
    term_structure : DataFrame with columns matching keys of `maturities`
                     and daily DatetimeIndex.
                     Rows with fewer than 3 non-NaN values are skipped.
    maturities     : dict mapping column names → τ in years
                     e.g. {"vix9d": 9/365, "vix": 30/365, ...}
    verbose        : print progress

    Returns
    -------
    DataFrame indexed by date with columns:
        beta1, beta2, beta3, lam, r2, rmse, n_points
    """
    cols = list(maturities.keys())
    tau_full = np.array([maturities[c] for c in cols])

    rows = []
    n = len(term_structure)

    for i, (date, row) in enumerate(term_structure.iterrows()):
        # Keep only non-NaN columns
        mask   = row[cols].notna().values
        if mask.sum() < 3:
            continue

        tau_i = tau_full[mask]
        iv_i  = row[cols].values[mask].astype(float)

        try:
            ns = fit_ns_optimal(tau_i, iv_i)
            if np.isnan(ns.r2) or ns.r2 < 0.5:
                continue
            rows.append({"date": date, **ns._asdict()})
        except Exception:
            continue

        if verbose and (i % 200 == 0 or i == n - 1):
            pct = (i + 1) / n * 100
            print(f"  NS fit: {pct:5.1f}%  ({date.date()})  "
                  f"β₁={ns.beta1:.2f}  β₂={ns.beta2:.2f}  "
                  f"β₃={ns.beta3:.2f}  λ={ns.lam:.2f}  R²={ns.r2:.3f}")

    if not rows:
        raise ValueError("No valid NS fits produced.")

    df = pd.DataFrame(rows).set_index("date").sort_index()

    if verbose:
        print(f"\n  ✓ {len(df)} daily NS fits  "
              f"({df.index[0].date()} → {df.index[-1].date()})")

    return df
