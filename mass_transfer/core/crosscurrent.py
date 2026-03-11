"""
Crosscurrent (single-section) multistage liquid-liquid extraction solver.

Each stage receives fresh solvent. Stages are solved sequentially:
    Feed → Stage 1 → Stage 2 → ... → Stage N → Final Raffinate
              ↓           ↓                ↓
           Extract 1   Extract 2        Extract N

For each stage i:
    - Feed to stage: raffinate from previous stage (R_{i-1}) + fresh solvent S
    - Products: new raffinate R_i + extract E_i
    - Equilibrium: tie-line relates raffinate and extract compositions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy.optimize import fsolve

from .equilibrium import (
    EquilibriumModel,
    get_X_from_ternary,
    get_N_from_ternary,
    get_raffinate_point_from_X,
    get_extract_point_from_Y,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class StageResult:
    """Results for a single crosscurrent stage."""
    stage_number: int

    # Raffinate leaving this stage (wt%)
    A_raff: float
    C_raff: float
    B_raff: float
    R_flow: float  # total raffinate flow (kg or kg/h)

    # Extract leaving this stage (wt%)
    A_ext: float
    C_ext: float
    B_ext: float
    E_flow: float  # total extract flow

    # Solvent-free values
    X_raff: float   # C/(A+C) in raffinate
    Y_ext: float    # C/(A+C) in extract
    N_raff: float   # B/(A+C) in raffinate
    N_ext: float    # B/(A+C) in extract

    # Performance
    acid_removed_kg: float    # kg of C removed in this stage
    pct_removal_stage: float  # % of feed acid removed in this stage
    pct_removal_cumul: float  # cumulative % removal


@dataclass
class CrosscurrentResult:
    """Complete crosscurrent extraction results."""
    # Feed conditions
    feed_A: float
    feed_C: float
    feed_B: float
    feed_flow: float
    solvent_per_stage: float
    n_stages: int

    stages: List[StageResult] = field(default_factory=list)

    # Mixed extract totals
    mixed_extract_A: float = 0.0
    mixed_extract_C: float = 0.0
    mixed_extract_B: float = 0.0
    mixed_extract_flow: float = 0.0

    # Final products (solvent-free basis)
    final_raff_X: float = 0.0
    final_ext_Y_mixed: float = 0.0
    total_pct_removal: float = 0.0


# ---------------------------------------------------------------------------
# Single stage solver
# ---------------------------------------------------------------------------

def _solve_single_stage(
    R_prev: float,
    xA_prev: float,
    xC_prev: float,
    xB_prev: float,
    S: float,
    eq: EquilibriumModel,
) -> tuple:
    """
    Solve one crosscurrent stage.

    Parameters
    ----------
    R_prev : total raffinate flow entering stage (kg)
    xA_prev, xC_prev, xB_prev : raffinate composition entering (wt fractions, 0-1)
    S : pure solvent added (kg)
    eq : equilibrium model

    Returns
    -------
    (R_out, xA_R, xC_R, xB_R, E_out, xA_E, xC_E, xB_E)
    """
    # Total mass into the stage
    M_in = R_prev + S

    # Mass of each component entering
    mA_in = R_prev * xA_prev
    mC_in = R_prev * xC_prev
    mB_in = R_prev * xB_prev + S  # pure solvent is B

    def equations(unknowns):
        X_raff = unknowns[0]
        R_out = unknowns[1]

        # Clamp X to physical range
        X_raff = max(0.0, min(1.0, X_raff))
        R_out = max(0.1, R_out)

        # Get raffinate ternary composition from equilibrium
        A_R, C_R, B_R = get_raffinate_point_from_X(eq, X_raff)

        # Convert to weight fractions
        wA_R = A_R / 100.0
        wC_R = C_R / 100.0
        wB_R = B_R / 100.0

        # Get equilibrium extract composition
        Y_ext = eq.Y_from_X(X_raff)
        A_E, C_E, B_E = get_extract_point_from_Y(eq, Y_ext)
        wA_E = A_E / 100.0
        wC_E = C_E / 100.0

        # Extract flow from total balance
        E_out = M_in - R_out

        # Mass balances (A and C)
        eq1 = mA_in - R_out * wA_R - E_out * wA_E
        eq2 = mC_in - R_out * wC_R - E_out * wC_E

        return [eq1, eq2]

    # Initial guess
    X_feed = mC_in / (mA_in + mC_in) if (mA_in + mC_in) > 0 else 0.0
    X0 = X_feed * 0.7  # extraction reduces X
    R0 = R_prev * 0.5  # rough guess

    sol = fsolve(equations, [X0, R0], full_output=True)
    X_sol, R_sol = sol[0]
    X_sol = max(0.0, min(1.0, float(X_sol)))
    R_sol = max(0.1, float(R_sol))

    # Reconstruct full compositions
    A_R, C_R, B_R = get_raffinate_point_from_X(eq, X_sol)
    Y_ext = eq.Y_from_X(X_sol)
    A_E, C_E, B_E = get_extract_point_from_Y(eq, Y_ext)

    E_sol = M_in - R_sol

    return (
        R_sol, A_R / 100.0, C_R / 100.0, B_R / 100.0,
        E_sol, A_E / 100.0, C_E / 100.0, B_E / 100.0,
    )


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def solve_crosscurrent(
    feed_A: float,
    feed_C: float,
    feed_flow: float,
    solvent_per_stage: float,
    n_stages: int,
    eq_model: EquilibriumModel,
    feed_B: float = 0.0,
) -> CrosscurrentResult:
    """
    Solve crosscurrent extraction for N stages.

    Parameters
    ----------
    feed_A : wt% of carrier (A) in feed
    feed_C : wt% of solute (C) in feed
    feed_flow : total feed flow rate (kg or kg/h)
    solvent_per_stage : kg of pure solvent added per stage
    n_stages : number of extraction stages
    eq_model : fitted EquilibriumModel
    feed_B : wt% of solvent in feed (default 0 for fresh feed)

    Returns
    -------
    CrosscurrentResult with per-stage details and summary.
    """
    feed_B_actual = feed_B if feed_B > 0 else (100.0 - feed_A - feed_C)
    if abs(feed_A + feed_C + feed_B_actual - 100.0) > 0.5:
        feed_B_actual = 100.0 - feed_A - feed_C

    result = CrosscurrentResult(
        feed_A=feed_A,
        feed_C=feed_C,
        feed_B=feed_B_actual,
        feed_flow=feed_flow,
        solvent_per_stage=solvent_per_stage,
        n_stages=n_stages,
    )

    # Initialize: entering raffinate
    R_prev = feed_flow
    xA_prev = feed_A / 100.0
    xC_prev = feed_C / 100.0
    xB_prev = feed_B_actual / 100.0

    total_acid_in_feed = feed_flow * feed_C / 100.0
    cumul_acid_removed = 0.0

    # Accumulators for mixed extract
    total_ext_A = 0.0
    total_ext_C = 0.0
    total_ext_B = 0.0
    total_ext_flow = 0.0

    for stage_num in range(1, n_stages + 1):
        R_out, wA_R, wC_R, wB_R, E_out, wA_E, wC_E, wB_E = _solve_single_stage(
            R_prev, xA_prev, xC_prev, xB_prev, solvent_per_stage, eq_model
        )

        # Acid removed in this stage
        acid_in = R_prev * xC_prev
        acid_out_raff = R_out * wC_R
        acid_removed = acid_in - acid_out_raff
        cumul_acid_removed += acid_removed

        # Solvent-free coordinates
        AC_R = wA_R + wC_R
        X_raff = wC_R / AC_R if AC_R > 0 else 0.0
        N_raff = wB_R / AC_R if AC_R > 0 else 0.0

        AC_E = wA_E + wC_E
        Y_ext = wC_E / AC_E if AC_E > 0 else 0.0
        N_ext = wB_E / AC_E if AC_E > 0 else 0.0

        stage = StageResult(
            stage_number=stage_num,
            A_raff=wA_R * 100.0, C_raff=wC_R * 100.0, B_raff=wB_R * 100.0,
            R_flow=R_out,
            A_ext=wA_E * 100.0, C_ext=wC_E * 100.0, B_ext=wB_E * 100.0,
            E_flow=E_out,
            X_raff=X_raff, Y_ext=Y_ext,
            N_raff=N_raff, N_ext=N_ext,
            acid_removed_kg=acid_removed,
            pct_removal_stage=(acid_removed / total_acid_in_feed * 100.0
                               if total_acid_in_feed > 0 else 0.0),
            pct_removal_cumul=(cumul_acid_removed / total_acid_in_feed * 100.0
                               if total_acid_in_feed > 0 else 0.0),
        )
        result.stages.append(stage)

        # Accumulate mixed extract
        total_ext_A += E_out * wA_E
        total_ext_C += E_out * wC_E
        total_ext_B += E_out * wB_E
        total_ext_flow += E_out

        # Update for next stage
        R_prev = R_out
        xA_prev = wA_R
        xC_prev = wC_R
        xB_prev = wB_R

    # Mixed extract composition
    if total_ext_flow > 0:
        result.mixed_extract_A = total_ext_A / total_ext_flow * 100.0
        result.mixed_extract_C = total_ext_C / total_ext_flow * 100.0
        result.mixed_extract_B = total_ext_B / total_ext_flow * 100.0
    result.mixed_extract_flow = total_ext_flow

    # Final solvent-free
    last = result.stages[-1]
    result.final_raff_X = last.X_raff
    if total_ext_flow > 0:
        AC_mix = total_ext_A + total_ext_C
        result.final_ext_Y_mixed = total_ext_C / AC_mix if AC_mix > 0 else 0.0
    result.total_pct_removal = last.pct_removal_cumul

    return result
