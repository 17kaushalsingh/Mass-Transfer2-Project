"""
Equilibrium model for ternary liquid-liquid extraction.

Components:
    A = Carrier (e.g., Cottonseed Oil)
    B = Solvent (e.g., Propane)
    C = Solute  (e.g., Oleic Acid)

Ternary coordinates: wt% basis, right-angle triangle (x=A, y=C, B=100-A-C).
Solvent-free basis:
    X = C/(A+C) in raffinate phase
    Y = C/(A+C) in extract phase
    N = B/(A+C) solvent ratio
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
from numpy.polynomial import polynomial as P
from scipy.interpolate import CubicSpline
from scipy.optimize import fsolve


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TieLineData:
    """Raw and solvent-free tie-line data."""
    # Raffinate phase (wt%)
    A_raff: np.ndarray
    C_raff: np.ndarray
    B_raff: np.ndarray
    # Extract phase (wt%)
    A_ext: np.ndarray
    C_ext: np.ndarray
    B_ext: np.ndarray
    # Solvent-free coordinates
    X: np.ndarray          # C/(A+C) raffinate
    Y: np.ndarray          # C/(A+C) extract
    N_raff: np.ndarray     # B/(A+C) raffinate
    N_ext: np.ndarray      # B/(A+C) extract


@dataclass
class EquilibriumModel:
    """Fitted equilibrium model with callable interpolation functions."""
    tie_line_data: TieLineData

    # Phase envelope on ternary triangle — A-axis fits (internal use)
    C_raff_from_A: Callable[[float], float] = field(repr=False)
    C_ext_from_A: Callable[[float], float] = field(repr=False)

    # Phase envelope on ternary triangle — B-axis fits (for correct B-x-axis plot)
    C_raff_from_B: Callable[[float], float] = field(repr=False)
    C_ext_from_B: Callable[[float], float] = field(repr=False)

    # Distribution curve
    Y_from_X: Callable[[float], float] = field(repr=False)

    # Solvent-ratio curves
    N_raff_from_X: Callable[[float], float] = field(repr=False)
    N_ext_from_Y: Callable[[float], float] = field(repr=False)

    # Conjugate line
    X_ext_from_X_raff: Callable[[float], float] = field(repr=False)

    # R-squared values for diagnostics
    r_squared: dict = field(default_factory=dict)

    # Polynomial coefficients (numpy polyfit convention: highest degree first)
    poly_coeffs: dict = field(default_factory=dict)

    # Plait point (where the two phases merge)
    plait_point: Optional[Tuple[float, float, float]] = None

    # Bounds for valid interpolation
    X_range: Tuple[float, float] = (0.0, 1.0)
    Y_range: Tuple[float, float] = (0.0, 1.0)

    def X_from_Y(self, Y_val: float) -> float:
        """Inverse of Y_from_X via numerical root-finding."""
        # Clamp Y to valid range
        Y_val = float(np.clip(Y_val, self.Y_range[0], self.Y_range[1]))
        x0 = Y_val  # initial guess: X ≈ Y
        sol = fsolve(lambda x: self.Y_from_X(float(x)) - Y_val, x0, full_output=True)
        return float(sol[0][0])


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_tie_line_data(filepath: str | Path) -> TieLineData:
    """
    Load tie-line data from JSON and compute solvent-free coordinates.

    Parameters
    ----------
    filepath : path to JSON with keys 'phase_1' (raffinate) and 'phase_2' (extract).
        Each entry has 'cottonseed_oil', 'oleic_acid', 'propane' (or generic A/B/C).

    Returns
    -------
    TieLineData with raw wt% and solvent-free coordinates.
    """
    filepath = Path(filepath)
    with open(filepath, "r") as f:
        raw = json.load(f)

    phase_1 = raw["phase_1"]  # raffinate
    phase_2 = raw["phase_2"]  # extract

    n = len(phase_1)
    assert len(phase_2) == n, "phase_1 and phase_2 must have equal length"

    # Detect key names — support both specific and generic
    sample = phase_1[0]
    if "cottonseed_oil" in sample:
        key_A, key_C, key_B = "cottonseed_oil", "oleic_acid", "propane"
    elif "A" in sample:
        key_A, key_C, key_B = "A", "C", "B"
    else:
        keys = list(sample.keys())
        key_A, key_C, key_B = keys[0], keys[1], keys[2]

    A_raff = np.array([p[key_A] for p in phase_1], dtype=float)
    C_raff = np.array([p[key_C] for p in phase_1], dtype=float)
    B_raff = np.array([p[key_B] for p in phase_1], dtype=float)

    A_ext = np.array([p[key_A] for p in phase_2], dtype=float)
    C_ext = np.array([p[key_C] for p in phase_2], dtype=float)
    B_ext = np.array([p[key_B] for p in phase_2], dtype=float)

    # Solvent-free coordinates
    AC_raff = A_raff + C_raff
    AC_ext = A_ext + C_ext

    X = np.where(AC_raff > 0, C_raff / AC_raff, 0.0)
    Y = np.where(AC_ext > 0, C_ext / AC_ext, 0.0)
    N_raff = np.where(AC_raff > 0, B_raff / AC_raff, 0.0)
    N_ext = np.where(AC_ext > 0, B_ext / AC_ext, 0.0)

    return TieLineData(
        A_raff=A_raff, C_raff=C_raff, B_raff=B_raff,
        A_ext=A_ext, C_ext=C_ext, B_ext=B_ext,
        X=X, Y=Y, N_raff=N_raff, N_ext=N_ext,
    )


# ---------------------------------------------------------------------------
# Fitting helpers
# ---------------------------------------------------------------------------

def _r_squared(y_actual: np.ndarray, y_predicted: np.ndarray) -> float:
    """Compute coefficient of determination R²."""
    ss_res = np.sum((y_actual - y_predicted) ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
    if ss_tot == 0:
        return 1.0
    return 1.0 - ss_res / ss_tot


def _fit_poly(x: np.ndarray, y: np.ndarray, degree: int) -> Tuple[Callable, float, np.ndarray]:
    """
    Fit polynomial y = f(x) using numpy.polyfit.

    Returns (callable, R², coefficients_highest_first).
    """
    coeffs = np.polyfit(x, y, degree)
    poly_func = np.poly1d(coeffs)
    y_pred = poly_func(x)
    r2 = _r_squared(y, y_pred)
    return poly_func, r2, coeffs


def _fit_cubic_spline(x: np.ndarray, y: np.ndarray) -> Tuple[Callable, float]:
    """Fit a cubic spline (for narrow-range extract curve)."""
    # Sort by x
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    cs = CubicSpline(xs, ys, bc_type="natural")
    y_pred = cs(x)
    r2 = _r_squared(y, y_pred)

    def wrapper(val):
        return float(cs(val))

    return wrapper, r2


# ---------------------------------------------------------------------------
# Main fitting function
# ---------------------------------------------------------------------------

def fit_equilibrium_model(data: TieLineData, raff_degree: int = 4,
                          ext_degree: int = 3, dist_degree: int = 3,
                          conj_degree: int = 2) -> EquilibriumModel:
    """
    Fit all equilibrium curves from tie-line data.

    Parameters
    ----------
    data : TieLineData from load_tie_line_data
    raff_degree : polynomial degree for raffinate curve C_raff(A)
    ext_degree : polynomial degree for extract curve C_ext(A)
    dist_degree : polynomial degree for distribution curve Y(X)
    conj_degree : polynomial degree for conjugate line X_ext(X_raff)

    Returns
    -------
    EquilibriumModel with all fitted callables.
    """
    r2 = {}
    coeffs = {}

    # --- Raffinate curve: C_raff = f(A_raff) on ternary triangle ---
    C_raff_func, r2["C_raff_from_A"], coeffs["C_raff_from_A"] = _fit_poly(
        data.A_raff, data.C_raff, raff_degree
    )

    # --- Extract curve: C_ext = f(A_ext) ---
    # Extract phase has a narrow A range (0.2-2.3), use cubic spline for better fit
    C_ext_func_spline, r2_spline = _fit_cubic_spline(data.A_ext, data.C_ext)
    C_ext_func_poly, r2_poly, coeffs_ext = _fit_poly(data.A_ext, data.C_ext, ext_degree)

    if r2_spline > r2_poly:
        C_ext_func = C_ext_func_spline
        r2["C_ext_from_A"] = r2_spline
        coeffs["C_ext_from_A"] = None  # spline, no poly coeffs
    else:
        C_ext_func = C_ext_func_poly
        r2["C_ext_from_A"] = r2_poly
        coeffs["C_ext_from_A"] = coeffs_ext

    # --- Distribution curve: Y = f(X) ---
    Y_func, r2["Y_from_X"], coeffs["Y_from_X"] = _fit_poly(
        data.X, data.Y, dist_degree
    )

    # --- Solvent-ratio curves ---
    N_raff_func, r2["N_raff_from_X"], coeffs["N_raff_from_X"] = _fit_poly(
        data.X, data.N_raff, 3
    )
    N_ext_func, r2["N_ext_from_Y"], coeffs["N_ext_from_Y"] = _fit_poly(
        data.Y, data.N_ext, 3
    )

    # --- Conjugate line: X_ext = f(X_raff) ---
    # X_ext ≈ Y (since X_ext = C_ext/(A_ext+C_ext) = Y in extract phase)
    conj_func, r2["X_ext_from_X_raff"], coeffs["X_ext_from_X_raff"] = _fit_poly(
        data.X, data.Y, conj_degree
    )

    # --- B-axis phase envelope fits: C = f(B) for correct ternary triangle ---
    # Sort by B ascending for well-behaved fitting
    raff_order = np.argsort(data.B_raff)
    C_raff_from_B_func, r2["C_raff_from_B"], coeffs["C_raff_from_B"] = _fit_poly(
        data.B_raff[raff_order], data.C_raff[raff_order], raff_degree
    )

    # Extract phase: use spline (narrow, non-monotone range)
    ext_order = np.argsort(data.B_ext)
    C_ext_from_B_spline, r2_ext_B_spline = _fit_cubic_spline(
        data.B_ext[ext_order], data.C_ext[ext_order]
    )
    C_ext_from_B_poly, r2_ext_B_poly, coeffs_ext_B = _fit_poly(
        data.B_ext[ext_order], data.C_ext[ext_order], ext_degree
    )
    if r2_ext_B_spline > r2_ext_B_poly:
        C_ext_from_B_func = C_ext_from_B_spline
        r2["C_ext_from_B"] = r2_ext_B_spline
        coeffs["C_ext_from_B"] = None
    else:
        C_ext_from_B_func = C_ext_from_B_poly
        r2["C_ext_from_B"] = r2_ext_B_poly
        coeffs["C_ext_from_B"] = coeffs_ext_B

    # Wrap numpy callables to accept scalar floats
    def _wrap(func):
        def wrapped(val):
            return float(func(float(val)))
        return wrapped

    # Determine valid interpolation ranges
    X_range = (float(np.min(data.X)), float(np.max(data.X)))
    Y_range = (float(np.min(data.Y)), float(np.max(data.Y)))

    # Estimate plait point (where raffinate and extract curves meet)
    plait = _estimate_plait_point(data, C_raff_func, C_ext_func)

    model = EquilibriumModel(
        tie_line_data=data,
        C_raff_from_A=_wrap(C_raff_func),
        C_ext_from_A=_wrap(C_ext_func) if callable(C_ext_func) else C_ext_func,
        C_raff_from_B=_wrap(C_raff_from_B_func),
        C_ext_from_B=C_ext_from_B_func if callable(C_ext_from_B_func) else _wrap(C_ext_from_B_func),
        Y_from_X=_wrap(Y_func),
        N_raff_from_X=_wrap(N_raff_func),
        N_ext_from_Y=_wrap(N_ext_func),
        X_ext_from_X_raff=_wrap(conj_func),
        r_squared=r2,
        poly_coeffs=coeffs,
        plait_point=plait,
        X_range=X_range,
        Y_range=Y_range,
    )
    return model


def _estimate_plait_point(data: TieLineData, C_raff_func, C_ext_func
                          ) -> Optional[Tuple[float, float, float]]:
    """Estimate the plait point where raffinate and extract curves meet."""
    try:
        # Find intersection of the two phase boundary curves
        # At the plait point, the two phases become identical
        # Use the last tie-line as an approximation (compositions converge)
        A_range = np.linspace(0.1, max(data.A_raff), 500)
        C_raff_vals = np.array([float(C_raff_func(a)) for a in A_range])
        C_ext_vals = np.array([float(C_ext_func(a)) for a in A_range])

        # Find where the curves cross
        diff = C_raff_vals - C_ext_vals
        sign_changes = np.where(np.diff(np.sign(diff)))[0]

        if len(sign_changes) > 0:
            idx = sign_changes[0]
            A_plait = float(A_range[idx])
            C_plait = float(C_raff_func(A_plait))
            B_plait = 100.0 - A_plait - C_plait
            return (A_plait, C_plait, B_plait)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Convenience functions for getting full ternary point from solvent-free X or Y
# ---------------------------------------------------------------------------

def get_raffinate_point_from_X(eq: EquilibriumModel, X: float) -> Tuple[float, float, float]:
    """
    Given solvent-free X = C/(A+C) in raffinate phase,
    return the full ternary point (A, C, B) in wt%.

    Solves the system:
        C = C_raff_from_A(A)
        C/(A+C) = X
    """
    X = float(X)

    def equations(A_guess):
        A = float(A_guess[0])
        C = eq.C_raff_from_A(A)
        if A + C <= 0:
            return [1.0]
        return [C / (A + C) - X]

    # Initial guess: if X is small, A is large
    A0 = max(0.1, (1.0 - X) * 60.0)
    sol = fsolve(equations, [A0], full_output=True)
    A_sol = float(sol[0][0])
    C_sol = eq.C_raff_from_A(A_sol)
    B_sol = 100.0 - A_sol - C_sol
    return (A_sol, C_sol, B_sol)


def get_extract_point_from_Y(eq: EquilibriumModel, Y: float) -> Tuple[float, float, float]:
    """
    Given solvent-free Y = C/(A+C) in extract phase,
    return the full ternary point (A, C, B) in wt%.
    """
    Y = float(Y)

    def equations(A_guess):
        A = float(A_guess[0])
        C = eq.C_ext_from_A(A)
        if A + C <= 0:
            return [1.0]
        return [C / (A + C) - Y]

    A0 = 1.0  # extract phase has very small A
    sol = fsolve(equations, [A0], full_output=True)
    A_sol = float(sol[0][0])
    C_sol = eq.C_ext_from_A(A_sol)
    B_sol = 100.0 - A_sol - C_sol
    return (A_sol, C_sol, B_sol)


def get_equilibrium_extract_from_raffinate(
    eq: EquilibriumModel, A_raff: float, C_raff: float
) -> Tuple[float, float, float]:
    """
    Given a raffinate composition (A, C in wt%), find the equilibrium
    extract composition via the distribution curve.

    Steps:
        1. Compute X = C/(A+C) for the raffinate
        2. Look up Y = Y_from_X(X)
        3. Get extract point from Y
    """
    AC = A_raff + C_raff
    if AC <= 0:
        return (0.0, 0.0, 100.0)  # pure solvent
    X = C_raff / AC
    Y = eq.Y_from_X(X)
    return get_extract_point_from_Y(eq, Y)


def get_N_from_ternary(A: float, C: float, B: float) -> float:
    """Compute solvent ratio N = B/(A+C) from ternary wt%."""
    AC = A + C
    if AC <= 0:
        return float("inf")
    return B / AC


def get_X_from_ternary(A: float, C: float) -> float:
    """Compute solvent-free composition X = C/(A+C)."""
    AC = A + C
    if AC <= 0:
        return 0.0
    return C / AC
