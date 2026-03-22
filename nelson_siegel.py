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
    λ     = decay parameter

IMPORTANT — fixed vs optimised λ:
    Diebold-Li (2006) fix λ at a constant chosen to maximise the
    curvature loading at an intermediate maturity. For our 4 VIX
    maturities (9d, 30d, 90d, 180d) the optimal fixed λ is around
    0.5–2.0 (in units of 1/years).

    Using a per-day optimised λ makes β₁, β₂, β₃ incomparable
    across time — the same curve shape can be represented by wildly
    different (β, λ) combinations. ARIMA on unstable betas produces
    garbage forecasts.

    We use FIXED_LAMBDA = 1.5 by default, matching the midpoint of
    the typical range for short-dated option surfaces.

Factor interpretation:
    β₁  → long-run vol level        (parallel shift)
    β₂  → short-term slope          (negative = upward sloping)
    β₃  → medium-term hump
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.linalg import lstsq
from typing import NamedTuple

# Fixed decay parameter.
#
# How to choose λ:
#   The curvature loading C(τ,λ) peaks at τ* ≈ 1.7/λ.
#   We want the peak near the middle of our maturity range (~90 days = 0.247yr).
#   → λ ≈ 1.7 / 0.247 ≈ 6.9, so λ=10 is appropriate.
#
# With λ=1.5 (too small for short maturities):
#   All four loadings are nearly identical → near-singular design matrix
#   → β₃ is unidentifiable → std(β) ~ 60–95 → ARIMA on noise.
#
# With λ=10 (correct for 9d–180d maturities):
#   Loadings spread well: L(9d)≈0.89, L(30d)≈0.67, L(90d)≈0.34, L(180d)≈0.20
#   C peaks near 90d → β₃ is well identified → stable betas → good ARIMA.
FIXED_LAMBDA = 10.0


# ── Factor loadings ───────────────────────────────────────────────────────────

def _L(tau: np.ndarray, lam: float) -> np.ndarray:
    lt = lam * tau
    return np.where(lt < 1e-9, 1.0, (1.0 - np.exp(-lt)) / lt)


def _C(tau: np.ndarray, lam: float) -> np.ndarray:
    return _L(tau, lam) - np.exp(-lam * tau)


def _X(tau: np.ndarray, lam: float) -> np.ndarray:
    return np.column_stack([np.ones_like(tau), _L(tau, lam), _C(tau, lam)])


# ── Single cross-section fit ──────────────────────────────────────────────────

class NSFit(NamedTuple):
    beta1:     float
    beta2:     float
    beta3:     float
    lam:       float
    r2:        float
    rmse:      float
    n_points:  int


def fit_ns(tau: np.ndarray, iv: np.ndarray, lam: float) -> NSFit:
    """OLS fit with fixed λ."""
    X = _X(tau, lam)
    betas, _, _, _ = lstsq(X, iv)
    iv_hat = X @ betas
    ss_res = float(np.sum((iv - iv_hat) ** 2))
    ss_tot = float(np.sum((iv - iv.mean()) ** 2))
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan
    rmse   = float(np.sqrt(ss_res / len(iv)))
    return NSFit(
        beta1=float(betas[0]), beta2=float(betas[1]), beta3=float(betas[2]),
        lam=lam, r2=r2, rmse=rmse, n_points=len(iv),
    )


def fit_ns_optimal(
    tau: np.ndarray, iv: np.ndarray,
    bounds: tuple[float, float] = (0.5, 50.0),
) -> NSFit:
    """Optimised λ via Brent — use only for diagnostics, not for rolling panel."""
    res = minimize_scalar(lambda l: fit_ns(tau, iv, l).rmse,
                          bounds=bounds, method="bounded")
    return fit_ns(tau, iv, res.x)


# ── Reconstruct IV at any maturity ────────────────────────────────────────────

def predict_iv(tau: float | np.ndarray, ns: NSFit) -> np.ndarray:
    tau_arr = np.atleast_1d(tau)
    return _X(tau_arr, ns.lam) @ np.array([ns.beta1, ns.beta2, ns.beta3])


# ── Rolling panel fit ─────────────────────────────────────────────────────────

def fit_ns_panel(
    term_structure: pd.DataFrame,
    maturities:     dict[str, float],
    lam:            float | None = None,
    verbose:        bool = True,
) -> pd.DataFrame:
    """
    Fit Nelson-Siegel on every row of `term_structure`.

    Parameters
    ----------
    lam : fixed λ to use. If None, uses FIXED_LAMBDA (recommended).
          Pass a specific value to override. Set to -1 to use per-day
          optimisation (not recommended — produces unstable betas).
    """
    use_lam   = FIXED_LAMBDA if lam is None else lam
    use_fixed = (lam != -1)

    if verbose:
        if use_fixed:
            print(f"  Using fixed λ = {use_lam}  "
                  f"(stable betas, ARIMA-ready)")
        else:
            print(f"  Using per-day optimised λ  "
                  f"(diagnostic mode only)")

    cols     = list(maturities.keys())
    tau_full = np.array([maturities[c] for c in cols])

    rows = []
    n    = len(term_structure)

    for i, (date, row) in enumerate(term_structure.iterrows()):
        mask = row[cols].notna().values
        if mask.sum() < 3:
            continue

        tau_i = tau_full[mask]
        iv_i  = row[cols].values[mask].astype(float)

        try:
            if use_fixed:
                ns = fit_ns(tau_i, iv_i, use_lam)
            else:
                ns = fit_ns_optimal(tau_i, iv_i)

            if np.isnan(ns.r2) or ns.r2 < 0.5:
                continue

            rows.append({"date": date, **ns._asdict()})
        except Exception:
            continue

        if verbose and (i % 200 == 0 or i == n - 1):
            pct = (i + 1) / n * 100
            print(f"  NS fit: {pct:5.1f}%  ({date.date()})  "
                  f"β₁={ns.beta1:6.2f}  β₂={ns.beta2:6.2f}  "
                  f"β₃={ns.beta3:6.2f}  R²={ns.r2:.3f}")

    if not rows:
        raise ValueError("No valid NS fits produced.")

    df = pd.DataFrame(rows).set_index("date").sort_index()

    if verbose:
        print(f"\n  ✓ {len(df)} daily NS fits  "
              f"({df.index[0].date()} → {df.index[-1].date()})")
        print(f"  β₁: mean={df.beta1.mean():.2f}  std={df.beta1.std():.2f}")
        print(f"  β₂: mean={df.beta2.mean():.2f}  std={df.beta2.std():.2f}")
        print(f"  β₃: mean={df.beta3.mean():.2f}  std={df.beta3.std():.2f}")

    return df
