"""
Countercurrent multistage liquid-liquid extraction solver.

Two modes:
    A) Simple countercurrent (no reflux) — solvent and feed flow in opposite directions
    B) Countercurrent with extract reflux — Ponchon-Savarit numerical analog
       using the difference-point method

Flow diagram (simple):
    Feed → [1] → [2] → ... → [N] → Raffinate
    Solvent ← [1] ← [2] ← ... ← [N] ← Solvent

Flow diagram (with reflux):
    Extract Product ← Separator ← E₁ ← [1] ← [2] ← ... ← [N] → Raffinate
                       ↓ R₀ reflux →  [1]               Feed →  [f]
    The enriching section (1..f) has operating lines through Δ_E
    The stripping section (f+1..N) has operating lines through Δ_S
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import fsolve, brentq

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
class CountercurrentStage:
    """Results for a single countercurrent stage."""
    stage_number: int
    X_raff: float   # solvent-free raffinate composition
    Y_ext: float    # solvent-free extract composition
    N_raff: float   # solvent ratio in raffinate
    N_ext: float    # solvent ratio in extract
    A_raff: float   # wt% A in raffinate
    C_raff: float   # wt% C in raffinate
    B_raff: float   # wt% B in raffinate
    A_ext: float    # wt% A in extract
    C_ext: float    # wt% C in extract
    B_ext: float    # wt% B in extract
    section: str = "enriching"  # "enriching" or "stripping"


@dataclass
class CountercurrentResult:
    """Complete countercurrent extraction results."""
    n_stages: int
    feed_stage: int
    stages: List[CountercurrentStage] = field(default_factory=list)
    reflux_ratio: Optional[float] = None
    min_reflux_ratio: Optional[float] = None
    min_stages: Optional[int] = None

    # Product specifications (solvent-free)
    X_feed: float = 0.0
    X_raff_spec: float = 0.0
    X_ext_spec: float = 0.0

    # Flows (solvent-free basis, kg/h)
    feed_flow_sf: float = 0.0
    extract_product_flow_sf: float = 0.0
    raffinate_product_flow_sf: float = 0.0

    # Difference points
    delta_E: Optional[Tuple[float, float]] = None  # (X_delta_E, N_delta_E)
    delta_S: Optional[Tuple[float, float]] = None  # (X_delta_S, N_delta_S)


# ---------------------------------------------------------------------------
# Simple countercurrent (no reflux)
# ---------------------------------------------------------------------------

def solve_countercurrent_simple(
    feed_A: float,
    feed_C: float,
    feed_flow: float,
    solvent_flow: float,
    n_stages: int,
    eq_model: EquilibriumModel,
) -> CountercurrentResult:
    """
    Solve simple countercurrent extraction (no reflux).

    Stages numbered 1 (extract end) to N (raffinate end).
    Feed enters at stage N, solvent enters at stage 1.

    Parameters
    ----------
    feed_A, feed_C : wt% in feed
    feed_flow : total feed flow (kg/h)
    solvent_flow : total pure solvent flow (kg/h)
    n_stages : number of stages
    eq_model : fitted equilibrium model
    """
    X_feed = feed_C / (feed_A + feed_C) if (feed_A + feed_C) > 0 else 0.0

    # We solve all stages simultaneously
    # Unknowns: X_1, X_2, ..., X_N (solvent-free raffinate at each stage)
    # At each stage, Y_i = Y_from_X(X_i) from equilibrium
    # Operating line from overall balance gives Y_{i-1} from X_i

    # Overall balance (solvent-free basis):
    # F'(X_F) + S'(0) = R'(X_N) + E'(Y_1)  ... but S is pure B, so S' = 0 on sf basis
    # Actually on sf basis: F' = feed_flow * (feed_A + feed_C)/100
    F_sf = feed_flow * (feed_A + feed_C) / 100.0

    def equations(X_vec):
        X = np.array(X_vec, dtype=float)
        X = np.clip(X, 0.001, 0.999)

        Y = np.array([eq_model.Y_from_X(float(xi)) for xi in X])

        resid = np.zeros(n_stages)

        # Operating line for simple countercurrent:
        # Balance around stages i to N:
        #   F'·X_F + L'_{i-1}·Y_{i-1} = R'·X_N + ... wait, let me think more carefully.
        #
        # For simple countercurrent:
        # Balance around stage i (mixing point):
        #   R_{i-1} + E_{i+1} = R_i + E_i
        #   (mass of A): R_{i-1}·xA_{i-1} + E_{i+1}·xA_{i+1} = R_i·xA_i + E_i·xA_i(eq)
        #
        # On solvent-free basis, the operating line is:
        #   Y_{i} = (F'/S_eff) * (X_{i+1} - X_N) + 0
        # This is approximate. Let me use overall balance approach.
        #
        # Balance from stage i+1 to stage N (raffinate end):
        #   F'·X_F + E'_{i+1}·Y_{i+1} = R_N'·X_N + F'·X_F ... no.
        #
        # Simpler: envelope around stages 1..i:
        #   E_1·Y_1 = R_0·X_0 ... but there's no R_0 for simple countercurrent
        #
        # Let me redefine clearly:
        # Stage numbering: Feed enters stage N, solvent enters stage 1.
        # R goes: Feed → N → N-1 → ... → 1 → Raffinate product
        # E goes: Solvent → 1 → 2 → ... → N → Extract product
        #
        # Balance around stages 1 to i (envelope from raffinate end to stage i):
        #   R_in(from feed/stage i+1) + S(pure solvent into stage 1)
        #     = R_out(raffinate from stage 1) + E_out(extract from stage i)
        #
        # Actually let me just use: for each stage, raffinate in equilibrium with extract
        # and operating line relates passing streams.
        #
        # Operating line (sf basis, balance around stages 1..i):
        # S*(Y_0) + R_{i+1}'*X_{i+1} = R_1'*X_1 + E_i'*Y_i ... too complicated.
        #
        # Let me use a direct approach: global A and C balances for each stage.

        # I'll solve it as a system of 2N equations (X_i, R_i for each stage)
        # But to keep it simpler with just N unknowns, I use the operating line on X-Y diagram.
        #
        # Operating line relates Y_{i-1} to X_i (passing streams between stages i-1 and i):
        # Y_{i-1} = (F_sf / E_total_sf) * X_i + (E_total_sf * Y_N - F_sf * X_F) / E_total_sf
        #
        # For simple countercurrent on sf basis:
        # Y_{i-1} = (R'_sf / E'_sf) * X_i + (Y_1 - (R'_sf/E'_sf)*X_1)
        # where R'_sf = F_sf (carrier doesn't change much for dilute), E'_sf ≈ small
        #
        # Actually this gets complicated because R' and E' change at each stage due to
        # different N values. Let me use the N-X/Y diagram approach with mass balances.

        # Per-stage approach with full balances:
        # For stage i, the ternary compositions are determined by X_i (equilibrium).
        # The operating line on N vs X/Y diagram: passing streams R_i and E_{i-1}
        # lie on a line through the difference point.
        #
        # For simple countercurrent without reflux, there's ONE difference point (Δ)
        # for all stages:
        #   Δ = Feed - Raffinate = Extract - Solvent (overall balance)
        #   On N-X/Y: Δ_X = X_Δ, Δ_N = N_Δ

        # Overall balance (sf): F'*X_F = R_N'*X_N + PE'*Y_1
        # where R_N' = raffinate product sf flow, PE' = extract product sf flow
        # X_N = X from last stage raffinate, Y_1 = Y from first stage extract
        # But these are unknowns. Let me just solve the envelope balance.

        # Balance around stages 1..i:
        # R_{i+1} + S = R_1 + E_i   (S enters at stage 1)
        # On entering stage i+1: if i < N, it's R_{i+1}; if i = N, it's Feed
        # E_{N+1} = 0 (no extract enters from outside at stage N end... wait)
        # S enters at stage 1 from outside.
        #
        # Let me use envelope from extract end (stage 1) to stage i:
        # In: feed to stage 1 from outside = pure solvent S (B only, no A or C)
        # In: raffinate from stage i+1 = R_{i+1} with composition (A,C,B)_R of stage i+1
        # Out: raffinate from stage 1 = R_1 (= final raffinate product)
        # Out: extract from stage i = E_i with composition from equilibrium at stage i
        #
        # No wait, for countercurrent, raffinate flows from N to 1, extract flows from 1 to N.
        #
        # Let me re-index. Stage j:
        # R_{j-1} (from stage j-1 or feed if j=N+1... ugh)
        #
        # New approach: just solve the X-Y operating line directly.
        # Balance around whole system (sf basis):
        # F'·X_F = R'·X_1 + E'·Y_N    (R' = raff product sf, E' = ext product sf)
        # where R' + E' = F' (sf mass balance)
        # X_1 is the final raffinate composition
        # Y_N is the final extract composition
        #
        # Operating line on X-Y: between any two adjacent stages,
        # Y_{i-1} = (R'_sf/E'_sf) * X_i + Y_1 - (R'_sf/E'_sf)*X_1
        # But R'_sf and E'_sf are approximately constant on sf basis.
        #
        # This is the Kremser-type approach. Let me use it.

        # Approximate: R' ≈ F' (carrier balance), so E' = F' - R' is the net sf extract
        # Actually more precisely:
        # R' = F_sf (since no carrier leaves with extract in ideal case)
        # In reality some carrier dissolves, but as approximation:

        # Operating line: Y_{i-1} = (F_sf/E_sf_total)*X_i + (something)
        # where E_sf_total comes from overall balance.

        # Actually, let me just use the relationship that:
        # For the operating line between stages on the X-Y diagram,
        # passing streams must satisfy the mass balance.
        #
        # Y_{i+1} = (F_sf / (F_sf + 0)) * X_i + ... nah this is getting circular.

        # Let me take a simpler approach: use the operating line derived from
        # the overall mass balance.
        # With pure solvent (Y_solvent = 0) entering opposite to feed:
        #
        # Operating line: Y = (R'_out/E'_total) * (X - X_raff_product) + 0
        # where R'_out = sf raffinate flow, E'_total = sf extract flow
        # X_raff_product = X_1 (final raffinate leaving stage 1)
        #
        # R'_out = F_sf - E'_total  (sf mass balance)
        # F_sf * X_F = R'_out * X_1 + E'_total * Y_N
        # Y_N is the extract leaving at the feed end, which equals Y from equilibrium at stage N
        #
        # But X_1 and Y_N are unknowns (part of the solution).
        # So I need to solve iteratively.

        # APPROACH: Iterate on X_1 (final raffinate composition).
        # Given X_1, step through stages using equilibrium + operating line.
        # Adjust X_1 until the feed end matches X_F.
        pass

        return resid

    # Better approach: use stage-stepping from the raffinate end
    return _solve_simple_countercurrent_stepping(
        feed_A, feed_C, feed_flow, solvent_flow, n_stages, eq_model
    )


def _solve_simple_countercurrent_stepping(
    feed_A: float,
    feed_C: float,
    feed_flow: float,
    solvent_flow: float,
    n_stages: int,
    eq_model: EquilibriumModel,
) -> CountercurrentResult:
    """
    Solve simple countercurrent using stage-stepping with operating line on X-Y diagram.

    Strategy: guess X_1 (final raffinate), step through stages to stage N,
    then adjust X_1 so that X_{N_feed_in} = X_F.
    """
    X_F = feed_C / (feed_A + feed_C) if (feed_A + feed_C) > 0 else 0.0
    F_sf = feed_flow * (feed_A + feed_C) / 100.0  # solvent-free feed flow

    # The operating line on the X-Y diagram for simple countercurrent:
    # Y = m*(X - X_1) where m = R'_out/E'_total and Y=0 when X=X_1
    # This assumes pure solvent (Y_solvent = 0).

    def step_through(X_1_guess):
        """Given X_1 (raffinate product), step through N stages and return X at feed end."""
        X_1 = float(np.clip(X_1_guess, 0.001, X_F - 0.001))

        # From overall sf balance: F_sf * X_F = R'*X_1 + E'*Y_N
        # R' + E' = F_sf
        # Y_N from equilibrium at stage N = Y_from_X(X_N) = Y_from_X(~X_F)
        # But Y_N is unknown. We need the operating line slope.

        # For the operating line, we need the ratio R'/E'.
        # Use the mixing point approach:
        # Feed (A,C wt%) mixed with solvent gives point M
        # M lies on the line between feed and solvent on the ternary diagram
        # R' and E' are determined by lever rule.

        # On X-Y diagram with pure solvent (Y_s = 0):
        # Operating line passes through (X_1, 0) and (X_F, Y_N)
        # where Y_N = Y_from_X(X_N) and X_N ≈ X_F (feed stage composition)

        # We don't know Y_N yet. Use the overall balance to get the operating line slope.
        # F_sf * X_F + 0 = R'*X_1 + E'*Y_N  ... (1)
        # Also: on sf basis, F_sf enters and splits.
        # The operating line connects (X_1, Y_0=0) to (X_F, Y_N).

        # Step from stage 1:
        # X_1 → Y_1 = Y_from_X(X_1)  (equilibrium)
        # Y_1 → X_2 via operating line: Y_1 = m*(X_2 - X_1)  → X_2 = X_1 + Y_1/m

        # But we need m. Get m from overall balance.
        # We need to estimate Y_N. Use iteration:
        # First estimate: step through with approximate m, then refine.

        # Actually, the slope m depends on solvent-to-feed ratio.
        # For simple countercurrent: the amount of solvent determines the operating line.
        # On the N-X/Y diagram, the difference point Δ has coordinates:
        #   X_Δ = (F'*X_F - R'*X_1) / (F' - R') = Y_N (from balance)
        #   Wait, this is getting circular.

        # Let me use a different approach.
        # The S/F ratio (on actual basis) determines separation.
        # On X-Y diagram: operating line slope = L'/V' where
        # L' = sf raffinate flow ≈ constant through column
        # V' = sf extract flow ≈ related to solvent rate

        # For the actual system:
        # N_raff and N_ext change at each stage, so L' and V' aren't constant.
        # But as approximation for the X-Y operating line:
        # The key relationship is the overall balance.

        # SIMPLER APPROACH: use the mixing point on the ternary diagram.
        # Mix point M: F + S → total
        # Then R₁ (final raffinate) and E_N (final extract) lie on a tie line through M... no.
        # R₁ and E_N are related by the difference point.

        # OK let me just use a direct numerical approach:
        # Unknowns: X_1 through X_N and Y_1 through Y_N
        # Equations:
        #   Y_i = Y_from_X(X_i)  for i=1..N  (equilibrium, N equations)
        #   Operating line for i=1..N-1: relates Y_i to X_{i+1} (passing streams)
        #   Plus: X_N is related to feed, Y_0 = 0 (pure solvent)
        #   That gives N-1 operating line equations + 1 feed condition = N equations
        #   Total: 2N equations for 2N unknowns. But we can substitute Y_i = Y_from_X(X_i).
        #   So N unknowns X_1..X_N and N equations from operating lines.

        # Operating line equation (sf basis, balance around stages 1..i):
        # R'_1*X_1 + E'_i*Y_i = S_sf*0 + R'_{i+1}*X_{i+1}
        # where S_sf = 0 (pure solvent has no sf component)
        # Wait, solvent IS the "free" component. On sf basis, pure solvent contributes 0.

        # On solvent-free basis:
        # Balance around stages 1 to i:
        # (solvent-free flow in raffinate going right) * X_{i+1}
        #   = (solvent-free flow in raffinate leaving left) * X_1
        #   + (solvent-free flow in extract leaving right) * Y_i
        #   - (solvent-free flow of solvent entering) * Y_solvent
        #
        # In sf, the flows are: R' going from left to right, E' going from right to left.
        # At each stage, sf flow changes slightly because A dissolves in extract.
        # Approximate: R'_sf ≈ F_sf for all stages (carrier mostly stays in raffinate).
        # Then: F_sf * X_{i+1} + 0 = F_sf * X_1 + E'_sf * Y_i
        # → Y_i = (F_sf / E'_sf) * (X_{i+1} - X_1)

        # E'_sf from overall balance: F_sf * X_F = F_sf * X_1 + E'_sf * Y_N
        # → E'_sf = F_sf * (X_F - X_1) / Y_N

        # So: Y_i = (Y_N / (X_F - X_1)) * (X_{i+1} - X_1)
        # This is the operating line: slope = Y_N / (X_F - X_1), passing through (X_1, 0).

        # But Y_N = Y_from_X(X_N) and X_N is determined by stepping.
        # So iterate: guess slope, step, check if feed matches.

        # Step from extract end (stage 1) to feed end (stage N):
        X_curr = X_1
        stages_data = []

        for i in range(n_stages):
            # Equilibrium: get Y from X
            Y_eq = eq_model.Y_from_X(X_curr)

            stages_data.append((X_curr, Y_eq))

            if i < n_stages - 1:
                # Operating line: need X_{i+2} from Y_{i+1} = Y_eq
                # Y_eq = slope * (X_next - X_1)  → X_next = X_1 + Y_eq/slope
                # But we don't know slope yet!
                pass

        # I need to iterate on slope. Let me restructure.
        return stages_data

    # Iterative solution: find X_1 such that after N stages, X_N matches X_F
    def objective(X_1_val):
        X_1 = float(np.clip(X_1_val[0], 0.001, X_F * 0.999))

        X_curr = X_1
        for i in range(n_stages):
            Y_eq = eq_model.Y_from_X(X_curr)

            if i < n_stages - 1:
                # Operating line to get X_next
                # Using the approximate operating line with constant sf flows:
                # We need slope. Estimate from overall:
                # At the feed end, X_N should be close to X_F
                # Y_N = Y_eq at stage N
                # Overall: Y = (slope) * (X - X_1)
                # slope = Y_N / (X_N - X_1)  but X_N is what we're solving for...

                # Use the SOLVENT RATIO approach instead.
                # The ratio of sf flows = F_sf / E_sf
                # E_sf on actual basis: the extract carries some A+C.
                # The solvent flow S contributes to E.
                # On actual basis, E_total_actual ≈ S + dissolved A + C
                # E_sf = dissolved A+C in extract ≈ small fraction

                # Simplify: use S/F ratio to determine operating line slope on X-Y.
                # The operating line on X-Y passes through (X_1, 0) (pure solvent end)
                # and (X_F, Y_N_target).
                # slope_op = Y_N / (X_F - X_1)  ... but Y_N is unknown.

                # Alternative: from the mixing point.
                # M = (F + S) mix → then operating line relates to M.
                # On X-Y sf diagram: the operating line slope is:
                # m = (A+C)_raff_sf_flow / (A+C)_ext_sf_flow ≈ F_sf / E_sf

                # E_sf (solvent-free extract flow) can be estimated:
                # From the mixing point: the total mass in is F + S.
                # The ratio E_sf/F_sf ≈ how much A+C goes to extract.
                # For very large S/F, almost all C goes to extract.

                # Let me use a different strategy:
                # Numerically, use the balance equations directly.
                # For stages 1..i, overall A balance (sf, approximately):
                # R'*X_1 + E_i'*Y_eq = R_{i+1}'*X_{i+1}
                # Assume R' ≈ constant = F_sf*(1-X_1)/(1-0) ≈ F_sf (carrier flow)
                # E_i' ≈ total extract sf... this is still approximate.

                # BEST APPROACH: use mixing point on ternary + lever rule.
                # But for the X-Y diagram, the operating line for simple countercurrent
                # (with pure solvent, Y_s=0) ALWAYS passes through (X_1, 0).
                # The other point is (X_F, Y_N) where Y_N = Y_from_X(X_feed_stage).
                # The slope depends on the solvent-to-feed ratio.

                # From mass balance on the mixing point:
                # Mix composition: M_A = (F*A_F)/(F+S), M_C = (F*C_F)/(F+S), M_B = (F*B_F + S)/(F+S)
                # X_M = M_C/(M_A+M_C) = (F*C_F)/(F*A_F+F*C_F) = C_F/(A_F+C_F) = X_F
                # So X_M = X_F (mixing doesn't change X on sf basis).
                # N_M = M_B/(M_A+M_C) = (F*B_F + S*100) / (F*(A_F+C_F))
                # For fresh feed (B=0): N_M = S*100 / (F*(A_F+C_F)/100*100)
                # = S / (F_sf) ... hmm units.

                # N_M = (0 + solvent_flow) / F_sf (since feed has no B, solvent is pure B)
                # Actually: total B = 0 + solvent_flow, total (A+C) = F_sf
                # N_M = solvent_flow / F_sf

                # The operating line on N-X/Y goes through:
                # Raffinate product: (X_1, N_raff_1) and Extract product: (Y_N, N_ext_N)
                # Both lie on a line through the difference point Δ.
                # For simple countercurrent: Δ lies at the intersection.
                # Δ_X = X_F (from above), Δ_N = -F_sf/... actually Δ is derived from:
                # Δ = R_0 - E_1 = R_1 - E_2 = ... (constant for all stages)
                # On sf-N diagram: (X_Δ, N_Δ) is the difference point.

                # This is getting very involved for the general case.
                # Let me use a practical numerical approach:

                # Operating line on X-Y (approximate for dilute systems):
                # slope ≈ L/V where L = F_sf, V = S_effective_sf
                # V_sf = S * (avg fraction of C in extract / C_total)
                # This is too approximate.

                # FINAL APPROACH: solve directly with fsolve on all X values.
                pass

            X_curr = X_curr  # placeholder, will use fsolve below

        return [0.0]  # placeholder

    # ===== DIRECT FSOLVE APPROACH =====
    # Unknowns: X_1, X_2, ..., X_N (sf raffinate compositions at each stage)
    # Constraints: N equations from operating line + equilibrium
    # Operating line (sf basis, approx constant sf flows):
    # Y_i = (F_sf/(F_sf*(X_F-X_1)/Y_from_X(X_N))) * (X_{i+1} - X_1)
    # Simplification: Y_i and X_{i+1} related by the line through (X_1,0) and (X_F, Y_N)

    # Actually, let me just solve it properly with full mass balances.
    # Use envelope balance around stages 1..i:
    # Component C (sf): R'*X_1 + E'_i*Y_i = R'*X_{i+1} + 0   (Y_0 = 0 for pure solvent)
    # Wait no. Balance around stages 1..i means:
    # IN: R_{i+1} (raff from stage i+1, or feed if i=N) + S (pure solvent at stage 1)
    # OUT: R_1 (final raffinate) + E_i (extract from stage i)
    #
    # On sf basis (constant carrier approximation):
    # C balance: F_sf * X_{i+1} = F_sf * X_1 + E_sf_approx * Y_i  (for i < N)
    #   where X_{N+1} = X_F  and  Y_0 = 0
    #   and E_sf_approx is the sf extract flow which we approximate as constant.
    #
    # For the last stage (i=N): X_{N+1} = X_F (the feed)
    # For the solvent end (i=0): Y_0 = 0
    #
    # Operating line: Y_i = (F_sf/E_sf) * (X_{i+1} - X_1)
    # E_sf from overall: E_sf = F_sf * (X_F - X_1) / Y_N  where Y_N = Y_from_X(X_N)
    # Substituting: Y_i = Y_from_X(X_N) * (X_{i+1} - X_1) / (X_F - X_1)

    # So: X_{i+1} = X_1 + (X_F - X_1) * Y_i / Y_N
    # And Y_i = Y_from_X(X_i)
    # Y_N = Y_from_X(X_N)

    # This gives us: X_{i+1} = X_1 + (X_F - X_1) * Y_from_X(X_i) / Y_from_X(X_N)
    # for i = 1, ..., N-1
    # And for i=N: X_{N+1} = X_F (automatically satisfied)

    # So we can step: given X_1, compute X_2, X_3, ..., X_N, then check X_{N+1} = X_F.
    # This is a 1D root-finding problem!

    def compute_X_sequence(X_1_val):
        """Step through stages given X_1, return X values."""
        X_1 = float(X_1_val)
        X = [X_1]

        for i in range(n_stages - 1):
            Y_i = eq_model.Y_from_X(X[i])
            Y_N_est = eq_model.Y_from_X(X[-1]) if i > 0 else eq_model.Y_from_X(X_F * 0.9)
            # Recompute with latest X_N estimate... this is iterative.
            # Let me use a different formulation.
            pass

        return X

    # Even simpler: iterative stepping.
    # Use the fact that the operating line on X-Y is a STRAIGHT LINE
    # through (X_1, 0) and (X_F, Y_N).
    # Given X_1, step: X_1 → Y_1 (eq) → X_2 (op line) → Y_2 (eq) → ... → X_N → check vs X_F
    # The operating line is: Y = [(Y_N)/(X_F - X_1)] * (X - X_1)
    # So: X_next = X_1 + Y * (X_F - X_1) / Y_N

    # But Y_N depends on X_N which we haven't computed yet!
    # TWO approaches:
    # 1) Iterate: guess Y_N, step, update Y_N, repeat
    # 2) Use fsolve on X_1 with nested stepping

    # Use approach: iterate to convergence on the operating line.

    def residual_X1(X_1_val):
        """Given X_1, step through N stages, return error in X_N vs X_F."""
        X_1 = float(np.clip(X_1_val, 0.001, X_F * 0.99))

        # Iterate on Y_N (operating line endpoint)
        Y_N_est = eq_model.Y_from_X(X_F * 0.95)  # initial estimate

        for iteration in range(30):
            X_curr = X_1
            X_all = [X_1]

            for stage in range(n_stages):
                Y_curr = eq_model.Y_from_X(X_curr)

                if stage < n_stages - 1:
                    # Operating line: X_next = X_1 + Y_curr * (X_F - X_1) / Y_N_est
                    if Y_N_est > 1e-10:
                        X_next = X_1 + Y_curr * (X_F - X_1) / Y_N_est
                    else:
                        X_next = X_curr * 1.1
                    X_next = float(np.clip(X_next, X_curr * 1.001, 0.999))
                    X_curr = X_next
                    X_all.append(X_next)

            Y_N_new = eq_model.Y_from_X(X_all[-1])
            if abs(Y_N_new - Y_N_est) < 1e-8:
                break
            Y_N_est = 0.5 * Y_N_est + 0.5 * Y_N_new

        return X_all[-1] - X_F

    # Find X_1 using brentq
    # X_1 must be between 0 and X_F
    try:
        X_1_sol = brentq(residual_X1, 0.001, X_F * 0.99, xtol=1e-6, maxiter=100)
    except ValueError:
        # Fall back to fsolve
        sol = fsolve(lambda x: residual_X1(x[0]), [X_F * 0.3], full_output=True)
        X_1_sol = float(sol[0][0])

    # Now reconstruct the full solution
    X_1 = float(np.clip(X_1_sol, 0.001, X_F * 0.99))

    # Final pass to get all stage data
    Y_N_est = eq_model.Y_from_X(X_F * 0.95)
    for iteration in range(50):
        X_curr = X_1
        X_all = [X_1]
        Y_all = []

        for stage in range(n_stages):
            Y_curr = eq_model.Y_from_X(X_curr)
            Y_all.append(Y_curr)

            if stage < n_stages - 1:
                if Y_N_est > 1e-10:
                    X_next = X_1 + Y_curr * (X_F - X_1) / Y_N_est
                else:
                    X_next = X_curr * 1.1
                X_next = float(np.clip(X_next, X_curr * 1.001, 0.999))
                X_curr = X_next
                X_all.append(X_next)

        Y_N_new = Y_all[-1]
        if abs(Y_N_new - Y_N_est) < 1e-8:
            break
        Y_N_est = 0.5 * Y_N_est + 0.5 * Y_N_new

    # Build result
    stages = []
    for i in range(n_stages):
        X_i = X_all[i]
        Y_i = Y_all[i]

        A_R, C_R, B_R = get_raffinate_point_from_X(eq_model, X_i)
        A_E, C_E, B_E = get_extract_point_from_Y(eq_model, Y_i)

        stages.append(CountercurrentStage(
            stage_number=i + 1,
            X_raff=X_i, Y_ext=Y_i,
            N_raff=get_N_from_ternary(A_R, C_R, B_R),
            N_ext=get_N_from_ternary(A_E, C_E, B_E),
            A_raff=A_R, C_raff=C_R, B_raff=B_R,
            A_ext=A_E, C_ext=C_E, B_ext=B_E,
            section="simple",
        ))

    result = CountercurrentResult(
        n_stages=n_stages,
        feed_stage=n_stages,
        stages=stages,
        X_feed=X_F,
        X_raff_spec=X_all[0],
        X_ext_spec=Y_all[-1],
        feed_flow_sf=F_sf,
    )
    return result


# ---------------------------------------------------------------------------
# Minimum stages (total reflux)
# ---------------------------------------------------------------------------

def find_min_stages(
    X_feed: float,
    X_raff_spec: float,
    X_ext_spec: float,
    eq_model: EquilibriumModel,
    max_stages: int = 100,
) -> int:
    """
    Find minimum theoretical stages at total reflux.

    At total reflux, the operating line is Y = X (on the X-Y diagram).
    Step between the equilibrium curve Y = f(X) and the Y = X line
    from the extract product to the raffinate product.

    Parameters
    ----------
    X_feed : solvent-free feed composition (not used directly, but defines system)
    X_raff_spec : desired raffinate purity (sf basis), e.g. 0.02 for 2%
    X_ext_spec : desired extract purity (sf basis), e.g. 0.90 for 90%
    eq_model : fitted equilibrium model
    max_stages : safety limit

    Returns
    -------
    Minimum number of theoretical stages.
    """
    # Start from extract product: Y = X_ext_spec
    # Step: Y → X via operating line (Y = X at total reflux, so X = Y)
    # Then: X → Y_eq via equilibrium: Y_eq = Y_from_X(X)
    # Continue until X <= X_raff_spec

    n_stages = 0
    Y_curr = X_ext_spec  # start from extract product end

    for _ in range(max_stages):
        # Operating line at total reflux: X = Y
        X_curr = Y_curr

        if X_curr <= X_raff_spec:
            break

        n_stages += 1

        # Equilibrium step: X → Y
        Y_eq = eq_model.Y_from_X(X_curr)

        # Move to next: the operating line gives Y_next = X_next = X_curr
        # Wait, at total reflux, operating line is Y = X.
        # We go: Y_curr (on op line) → X_curr = Y_curr (horizontal to eq curve)
        # Then: on eq curve at X_curr, Y_eq = Y_from_X(X_curr)
        # Then: Y_eq → X_next = Y_eq (drop to op line Y=X)
        # So: X_next = Y_eq ... no wait.

        # Correct stepping procedure:
        # 1. Start at Y = X_ext on the operating line → X = X_ext (since Y=X)
        # 2. Move horizontally to equilibrium curve: at X = X_ext, Y_eq = Y_from_X(X_ext)
        # 3. This doesn't make sense going this direction because Y_eq > X typically means
        #    the equilibrium curve is above Y=X line.

        # Actually for stepping DOWN (from extract to raffinate):
        # Start at X = X_ext on the X axis (or equivalently at Y = X_ext on Y=X line)
        # Equilibrium gives Y at this X: Y = Y_from_X(X_ext)
        # But we want to go from high to low, so:
        # 1. X_curr = X_ext_spec (starting raffinate composition at the extract end)
        #    Y_curr = Y_from_X(X_curr) (equilibrium)
        # 2. Operating line: next raffinate X_next such that Y_curr = X_next (Y=X)
        #    So X_next = Y_curr ... but Y_from_X(X) < X usually means solute prefers raff.
        #    For our system Y_from_X(X) < X (distribution coefficient < 1 typically for high X)
        #    This would make X_next < X_curr, going in the wrong direction!

        # Let me reconsider. In LLE, the distribution curve Y vs X:
        # Y is the extract-phase sf composition, X is raffinate-phase sf composition.
        # For a solute that preferentially goes to the solvent (extract): Y_from_X(X) > X
        # at low X, and potentially Y_from_X(X) < X at high X.

        # The stepping should be from the RAFFINATE end:
        # Start at X = X_raff_spec
        # 1. Equilibrium: Y = Y_from_X(X_raff_spec)
        # 2. Operating line (Y=X): X_next = Y (move up to Y=X line)
        # 3. Repeat until X >= X_ext_spec
        break  # exit the loop, redo below

    # Correct implementation: step from raffinate to extract end
    n_stages = 0
    X_curr = X_raff_spec

    for _ in range(max_stages):
        # Equilibrium: Y = Y_from_X(X)
        Y_eq = eq_model.Y_from_X(X_curr)

        n_stages += 1

        # At total reflux: operating line is Y = X
        # Next X (from the operating line): X_next = Y_eq
        X_next = Y_eq

        if X_next >= X_ext_spec:
            break

        X_curr = X_next

    return n_stages


# ---------------------------------------------------------------------------
# Minimum reflux ratio
# ---------------------------------------------------------------------------

def find_min_reflux_ratio(
    feed_A: float,
    feed_C: float,
    feed_flow: float,
    X_raff_spec: float,
    X_ext_spec: float,
    eq_model: EquilibriumModel,
) -> float:
    """
    Find minimum reflux ratio using pinch point analysis.

    The minimum reflux occurs when the operating line and equilibrium curve
    touch (pinch point), typically at the feed composition.

    Parameters
    ----------
    feed_A, feed_C : wt% in feed
    feed_flow : total feed flow
    X_raff_spec : desired raffinate purity (sf, e.g. 0.02)
    X_ext_spec : desired extract purity (sf, e.g. 0.90)
    eq_model : fitted equilibrium model

    Returns
    -------
    Minimum reflux ratio (r_min).
    """
    X_F = feed_C / (feed_A + feed_C) if (feed_A + feed_C) > 0 else 0.0

    # At the pinch point (typically at feed), the operating line touches the
    # equilibrium curve. On the N-X/Y diagram:
    # The difference point Δ_E for the enriching section lies on the line
    # through the feed point and the pinch point.

    # For minimum reflux, the operating line in the enriching section passes
    # through the equilibrium point at the feed composition.

    # Feed point on N-X/Y: (X_F, N_raff_at_X_F) and (Y_F_eq, N_ext_at_Y_F)
    Y_F = eq_model.Y_from_X(X_F)
    N_raff_F = eq_model.N_raff_from_X(X_F)
    N_ext_F = eq_model.N_ext_from_Y(Y_F)

    # Extract product point: (X_ext_spec, N_ext_at_ext_product)
    # At the extract product end, composition is Y = X_ext_spec
    N_ext_product = eq_model.N_ext_from_Y(X_ext_spec)

    # The difference point Δ_E at minimum reflux passes through:
    # The extract product point (Y = X_ext_spec, N = N_ext_product) and
    # the feed equilibrium point on the extract side (Y_F, N_ext_F)

    # Δ_E is the intersection of:
    # Line through (X_ext_spec, N_ext_product) and (Y_F, N_ext_F)
    # extended above the N curves.

    # The coordinates of Δ_E:
    if abs(X_ext_spec - Y_F) < 1e-10:
        # Degenerate case
        return 0.0

    slope_min = (N_ext_product - N_ext_F) / (X_ext_spec - Y_F)
    N_delta_E_min = N_ext_product + slope_min * (X_ext_spec - X_ext_spec)  # = N_ext_product
    # Actually: Δ_E lies on the line through these two points, extrapolated.
    # Δ_E has coordinates (X_Δ, N_Δ) where X_Δ = X_ext_spec (for extract product)
    # Wait, that's not right either.

    # Let me use the standard formulation:
    # For the enriching section with reflux:
    # Reflux ratio r = R0 / PE  (reflux flow / extract product flow, on sf basis)
    # R0 returns to the column, PE is the extract product withdrawn.
    # E1 = R0 + PE, where E1 is the extract leaving stage 1 (top).
    #
    # On N-X/Y diagram:
    # Δ_E has coordinates: X_Δ = X_ext_spec, N_Δ = -(1+r)/r * (something)
    # Actually the difference point for the enriching section:
    # Δ_E = E1 - R0 = PE (the net flow = extract product)
    # On sf basis: Δ_E_sf = PE_sf, with composition X_Δ = X_ext_spec
    # N_Δ = -(N_PE) ... hmm, the difference point can have negative N.

    # Standard result for extract reflux:
    # The difference point Δ_E is located at:
    # X_Δ = X_ext_spec (= Y of extract product on sf basis)
    # N_Δ = -(1 + 1/r) / (1/N_R0 - 1/N_E1) ... this is getting complicated.

    # SIMPLER: use the direct pinch-point formula.
    # At minimum reflux, the enriching operating line passes through
    # the equilibrium tie-line at the feed stage.
    # The slope of this line on N-Y vs Y plot gives the minimum reflux.

    # For the enriching section, operating lines pass through Δ_E.
    # At minimum reflux, the line through Δ_E and the extract product
    # also passes through the feed equilibrium point.

    # Δ_E coordinates: (X_Δ_E, N_Δ_E)
    # Line through (Y_F, N_ext_F) and (X_ext_spec, N_ext_prod):
    # This line extended gives Δ_E at minimum reflux.

    # Actually the line through the feed tie-line endpoints on the N-X/Y diagram
    # (the raffinate feed point and extract feed point) intersected with
    # the extract product composition vertical gives Δ_E at minimum reflux.

    # Point 1: Feed raffinate (X_F, N_raff_F)
    # Point 2: Feed extract (Y_F, N_ext_F)
    # Extend line to X = X_ext_spec:
    if abs(Y_F - X_F) < 1e-10:
        return 0.0

    slope_tie = (N_ext_F - N_raff_F) / (Y_F - X_F)
    N_delta_E_min = N_raff_F + slope_tie * (X_ext_spec - X_F)

    # The reflux ratio relates to N_Δ_E:
    # For the extract product: the point on the N curve is (X_ext_spec, N_ext_product)
    # Δ_E is at (X_ext_spec, N_Δ_E)
    # The reflux ratio: r = R0/PE
    # From the lever rule: r = (N_Δ_E - N_ext_product) / (N_ext_product - N_R0)
    # where N_R0 is the N value of the reflux stream.
    # Assuming reflux is extract product with solvent removed: N_R0 ≈ 0
    # Then: r_min = N_delta_E_min / N_ext_product - 1 ... approximately.

    # More precisely, for extract reflux, the reflux R0 has the same composition
    # as the extract product after solvent removal.
    # If extract product has Y = X_ext_spec and we remove solvent:
    # The reflux stream on the N-Y diagram is at (X_ext_spec, 0) ideally.
    # But in practice, some solvent remains.

    # Using the standard formula:
    # r_min = (N_delta_E_min - N_ext_product) / (N_ext_product - 0)
    # where 0 is the N value of the reflux (solvent-free reflux).

    # Actually, the difference point Δ_E for the enriching section:
    # Δ_E = PE (extract product, net outflow)
    # On N-X/Y: if PE is essentially solvent-free, then N_PE = 0.
    # But PE after solvent removal... hmm.

    # Let me use a more practical formula:
    # r_min = (N_Δ_E - N_E1) / (N_E1 - N_R0)
    # At the extract product end: E1 has composition (X_ext_spec, N_ext_product)
    # R0 (reflux) has composition (X_ext_spec, N_R0) where N_R0 depends on how
    # the reflux is prepared (e.g., by solvent evaporation).

    # Standard assumption: reflux is produced by total solvent removal from extract.
    # So R0 is a liquid at (X_ext_spec, 0) on the N-X/Y diagram.
    # But this point may not lie on the raffinate curve. The actual R0 point
    # lies on the ternary diagram with B=0 (or minimal B).

    # For this system, after removing propane from the extract:
    # R0 has composition: A_R0 = (100-X_ext_spec*100), C_R0 = X_ext_spec*100, B_R0 = 0
    # Wait, on wt% basis: if Y = C/(A+C) = X_ext_spec, then C = X_ext_spec*(A+C)
    # With B=0: A + C = 100, so C = 100*X_ext_spec, A = 100*(1-X_ext_spec)
    # N_R0 = 0/100 = 0

    N_R0 = 0.0  # solvent-free reflux

    # r_min from the pinch-point analysis:
    # Δ_E is at (X_ext_spec, N_Δ_E_min) on the N-Y diagram
    # The extract leaving stage 1: (X_ext_spec, N_ext_product)
    # The reflux R0: (X_ext_spec, N_R0 = 0)
    # Lever rule: Δ_E_flow / E1_flow = (N_ext_product - N_R0) / (N_Δ_E_min - N_R0)
    # And: r = R0/PE = R0/(E1 - R0)
    # E1 = R0 + PE → PE = E1 - R0
    # Δ_E = PE (on sf basis)
    # R0 / PE = (E1 - PE) / PE = E1/PE - 1

    # From the N coordinates:
    # PE is at (X_ext_spec, 0) if product is solvent-free... no, PE = Δ_E
    # This is getting circular. Let me use the simpler relationship.

    # The key relationship is:
    # On the N-Y diagram, the enriching section operating lines pass through Δ_E.
    # The position of Δ_E determines r.

    # For the reflux stream (after total solvent removal): it enters at the top
    # as a liquid with composition (X_ext_spec, 0) on the N-X diagram.
    # The extract leaving stage 1 is at (Y_1 ≈ X_ext_spec, N_ext_1 = N_ext_product).

    # Balance: E1 = R0 + PE
    # On N diagram: E1 at N_ext_product, R0 at 0, PE is the product.
    # By lever rule: R0/PE = |N_ext_product - N_Δ_E| / |N_Δ_E - 0|
    # If Δ_E is below 0 (negative N): r = R0/PE = (N_ext_product - N_Δ_E) / (-N_Δ_E)

    # Actually for extract reflux: Δ_E lies BELOW the N=0 line (negative N region)
    # because the net flow (extract product) has negative N contribution.

    # Let me just use the formula: r = R0/PE
    # From the geometry:
    # N_Δ_E = -PE_sf / (something) ... hmm.

    # OK, let me derive this properly.
    # Let's work in sf basis:
    # E1_sf = R0_sf + PE_sf  ... (1) (sf = A+C content)
    # E1_sf * N_E1 = R0_sf * N_R0 + PE_sf * N_PE  ... (2) (B balance)
    # For total solvent removal: N_R0 = 0, N_PE = 0 (both are solvent-free after removal)
    # Wait, but E1 HAS solvent. When we strip solvent from E1:
    # E1 → separate into: liquid (R0 + PE mixture) + recovered solvent
    # R0 and PE are both solvent-free: N_R0 = N_PE = 0
    # From (2): E1_sf * N_E1 = 0 → this can't be right since E1 has solvent.

    # The issue is that R0 and PE have the SAME composition (just different amounts).
    # R0 is refluxed back, PE is withdrawn as product.
    # Both have the same composition: that of the extract product after solvent removal.
    # N_R0 = N_PE = 0 (all solvent removed).

    # Difference point for enriching section:
    # Δ_E represents the NET flow out of the enriching section top.
    # Δ_E = E1 - R0 = PE (on total basis)
    # But in terms of the streams: PE has N_PE = 0 (solvent-free).
    # On the N-Y diagram: PE is at (Y = X_ext_spec, N = 0).

    # Now: r = R0_sf / PE_sf
    # E1_sf = R0_sf + PE_sf = PE_sf * (1 + r)
    # B balance: E1_sf * N_E1 = R0_sf * 0 + PE_sf * 0 = 0 ?!
    # That says N_E1 = 0, which contradicts that E1 has solvent.

    # The problem is that after we remove solvent from E1, the reflux R0 is solvent-free.
    # But E1 (the actual extract leaving stage 1) DOES have solvent.
    # The solvent removal step: E1 → R0 + PE + recovered solvent S_rec
    # The recovered solvent is recycled.
    # So R0 is a liquid at Y = X_ext_spec but with B = 0 (on the ternary triangle).
    # On the N-Y diagram: R0 is at (X_ext_spec, 0).

    # E1 is at (X_ext_spec, N_ext_at_Y=X_ext_spec) = (X_ext_spec, N_ext_product).

    # Δ_E on the N-Y diagram:
    # Using the definition Δ_E = E1 - R0 (net upward flow):
    # Δ_E_sf = E1_sf - R0_sf = PE_sf
    # Δ_E_N = (E1_sf * N_E1 - R0_sf * N_R0) / PE_sf
    #        = (E1_sf * N_E1 - 0) / PE_sf
    #        = ((1+r)*PE_sf * N_E1) / PE_sf
    #        = (1+r) * N_E1  where N_E1 = N_ext_product

    # Wait, that gives N_Δ_E > N_E1, meaning Δ_E is ABOVE E1 on the N diagram.
    # Hmm, but the standard diagrams show Δ_E below.

    # I think the sign convention matters. For EXTRACT reflux:
    # The net flow at the top is: going OUT = PE (extract product), going IN = R0 (reflux)
    # From the column's perspective: net OUT at top = PE = E1 - R0
    # Δ_E = PE = net flow OUT at top

    # Let me redefine: Δ_E represents the net flow.
    # For each stage i in the enriching section:
    # E_i - R_{i-1} = Δ_E = constant (net flow)
    # Where R_0 = reflux.

    # On N-Y/X coordinates:
    # The difference point Δ_E: all operating lines (connecting R_{i-1} and E_i)
    # pass through the point Δ_E on the N-Y diagram.

    # Δ_E composition:
    # X_Δ = (E_i*Y_i - R_{i-1}*X_{i-1}) / (E_i - R_{i-1}) on sf basis
    # N_Δ = (E_i*N_E_i - R_{i-1}*N_R_{i-1}) / (E_i - R_{i-1}) on B basis

    # At stage 1:
    # X_Δ = (E1_sf*X_ext_spec - R0_sf*X_ext_spec) / (E1_sf - R0_sf)
    #      = X_ext_spec * (E1_sf - R0_sf) / (E1_sf - R0_sf) = X_ext_spec ✓

    # N_Δ = (E1_sf*N_ext_product - R0_sf*0) / (E1_sf - R0_sf)
    #      = E1_sf * N_ext_product / PE_sf
    #      = (1+r) * N_ext_product

    # So Δ_E is at (X_ext_spec, (1+r)*N_ext_product) — ABOVE E1 on N diagram.

    # At minimum reflux, the operating line through Δ_E passes through
    # the feed tie-line endpoints.
    # The Δ_E at min reflux: the line through (X_F, N_raff_F) and Δ_E,
    # extended, passes through (Y_F, N_ext_F).
    # Or equivalently: the line through the feed tie-line endpoints
    # (raffinate point and extract point on N-X/Y) extended to X = X_ext_spec
    # gives N_Δ_E_min.

    # Slope of feed tie-line on N-X/Y diagram:
    # From (X_F, N_raff_F) to (Y_F, N_ext_F):
    # slope = (N_ext_F - N_raff_F) / (Y_F - X_F)

    # Extend to X = X_ext_spec:
    # N_Δ_E_min = N_raff_F + slope * (X_ext_spec - X_F)

    # Already computed above:
    # N_delta_E_min = N_raff_F + slope_tie * (X_ext_spec - X_F)

    # Then: r_min = N_delta_E_min / N_ext_product - 1
    r_min = N_delta_E_min / N_ext_product - 1.0

    return max(0.0, r_min)


# ---------------------------------------------------------------------------
# Countercurrent with extract reflux
# ---------------------------------------------------------------------------

def solve_countercurrent_reflux(
    feed_A: float,
    feed_C: float,
    feed_flow: float,
    reflux_ratio: float,
    X_raff_spec: float,
    X_ext_spec: float,
    eq_model: EquilibriumModel,
    max_stages: int = 100,
) -> CountercurrentResult:
    """
    Solve countercurrent extraction with extract reflux.

    Uses the difference-point method (numerical Ponchon-Savarit).

    Parameters
    ----------
    feed_A, feed_C : wt% in feed
    feed_flow : total feed flow (kg/h)
    reflux_ratio : r = R0/PE (reflux / extract product, sf basis)
    X_raff_spec : desired raffinate product composition (sf basis)
    X_ext_spec : desired extract product composition (sf basis)
    eq_model : fitted equilibrium model
    max_stages : safety limit

    Returns
    -------
    CountercurrentResult with stage-by-stage data.
    """
    X_F = feed_C / (feed_A + feed_C) if (feed_A + feed_C) > 0 else 0.0
    F_sf = feed_flow * (feed_A + feed_C) / 100.0  # solvent-free feed flow

    # Product flows (solvent-free basis)
    PE_sf = F_sf * (X_F - X_raff_spec) / (X_ext_spec - X_raff_spec)
    RN_sf = F_sf - PE_sf

    # N values on the equilibrium curves
    N_ext_product = eq_model.N_ext_from_Y(X_ext_spec)
    N_raff_product = eq_model.N_raff_from_X(X_raff_spec)

    # Difference point for enriching section
    # Δ_E at (X_ext_spec, N_Δ_E) where N_Δ_E = (1+r)*N_ext_product
    N_delta_E = (1.0 + reflux_ratio) * N_ext_product
    X_delta_E = X_ext_spec

    # Difference point for stripping section
    # From overall balance: Δ_S = Δ_E - F (on the N-X/Y diagram)
    # Δ_S coordinates: overall balance around the whole column
    # Δ_E = PE (net out at top), Δ_S = RN (net out at bottom, in the other direction)
    # Actually: Δ_E - Δ_S = F (feed)
    # Δ_S has coordinates:
    # X_Δ_S = (PE_sf * X_ext_spec - F_sf * X_F) / (PE_sf - F_sf)
    #        = (PE_sf * X_ext_spec - F_sf * X_F) / (-RN_sf)
    # N_Δ_S = (PE_sf * N_Δ_E - F_sf * N_F) / (PE_sf - F_sf)

    # Feed N value: for fresh feed with no solvent, N_F = 0
    feed_B = 100.0 - feed_A - feed_C
    N_F = feed_B / (feed_A + feed_C) if (feed_A + feed_C) > 0 else 0.0

    X_delta_S = (PE_sf * X_ext_spec - F_sf * X_F) / (PE_sf - F_sf)
    N_delta_S = (PE_sf * N_delta_E - F_sf * N_F) / (PE_sf - F_sf)

    # Stage stepping from the extract product end
    stages = []
    X_curr = X_ext_spec  # start from extract end
    section = "enriching"
    feed_stage = 0

    for stage_num in range(1, max_stages + 1):
        # Current point is on the extract curve (Y = X_curr in sf basis)
        Y_curr = X_curr  # on the operating line, this is the extract composition

        # Get the equilibrium raffinate composition
        # Y → X via equilibrium (inverse): X_eq = X such that Y_from_X(X) = Y_curr
        X_eq = eq_model.X_from_Y(Y_curr)

        # Get full ternary points
        A_R, C_R, B_R = get_raffinate_point_from_X(eq_model, X_eq)
        A_E, C_E, B_E = get_extract_point_from_Y(eq_model, Y_curr)
        N_raff = get_N_from_ternary(A_R, C_R, B_R)
        N_ext = get_N_from_ternary(A_E, C_E, B_E)

        stages.append(CountercurrentStage(
            stage_number=stage_num,
            X_raff=X_eq, Y_ext=Y_curr,
            N_raff=N_raff, N_ext=N_ext,
            A_raff=A_R, C_raff=C_R, B_raff=B_R,
            A_ext=A_E, C_ext=C_E, B_ext=B_E,
            section=section,
        ))

        # Check if we've reached the raffinate specification
        if X_eq <= X_raff_spec:
            break

        # Check for feed stage transition (enriching → stripping)
        if section == "enriching" and X_eq <= X_F:
            section = "stripping"
            feed_stage = stage_num

        # Operating line: find the next extract composition Y_next
        # The line through (X_eq, N_raff) and the appropriate difference point
        # intersected with the N_ext(Y) curve gives Y_next.

        if section == "enriching":
            Y_next = _find_next_Y_from_delta(
                X_eq, N_raff, X_delta_E, N_delta_E, eq_model
            )
        else:
            Y_next = _find_next_Y_from_delta(
                X_eq, N_raff, X_delta_S, N_delta_S, eq_model
            )

        if Y_next is None or Y_next <= 0 or np.isnan(Y_next):
            break

        X_curr = Y_next  # the extract composition for the next stage

    if feed_stage == 0:
        feed_stage = len(stages)

    result = CountercurrentResult(
        n_stages=len(stages),
        feed_stage=feed_stage,
        stages=stages,
        reflux_ratio=reflux_ratio,
        X_feed=X_F,
        X_raff_spec=X_raff_spec,
        X_ext_spec=X_ext_spec,
        feed_flow_sf=F_sf,
        extract_product_flow_sf=PE_sf,
        raffinate_product_flow_sf=RN_sf,
        delta_E=(X_delta_E, N_delta_E),
        delta_S=(X_delta_S, N_delta_S),
    )

    # Compute min stages and min reflux for reference
    result.min_stages = find_min_stages(X_F, X_raff_spec, X_ext_spec, eq_model)
    result.min_reflux_ratio = find_min_reflux_ratio(
        feed_A, feed_C, feed_flow, X_raff_spec, X_ext_spec, eq_model
    )

    return result


def _find_next_Y_from_delta(
    X_raff: float,
    N_raff: float,
    X_delta: float,
    N_delta: float,
    eq_model: EquilibriumModel,
) -> Optional[float]:
    """
    Find the next extract composition using the operating line through the difference point.

    On the N-X/Y diagram, draw a line through (X_raff, N_raff) and (X_delta, N_delta).
    Find where this line intersects the N_ext(Y) curve.

    Returns Y_next (sf extract composition at the next stage).
    """
    # Line: N = N_raff + slope * (Y - X_raff)
    # where slope = (N_delta - N_raff) / (X_delta - X_raff)

    if abs(X_delta - X_raff) < 1e-10:
        return None

    slope = (N_delta - N_raff) / (X_delta - X_raff)

    def equation(Y):
        Y = float(Y)
        N_line = N_raff + slope * (Y - X_raff)
        N_curve = eq_model.N_ext_from_Y(Y)
        return N_line - N_curve

    # Search for intersection in valid Y range
    Y_min, Y_max = eq_model.Y_range
    Y_min = max(Y_min, 0.001)

    try:
        # Try brentq first for robustness
        # Check sign change
        f_min = equation(Y_min)
        f_max = equation(Y_max)

        if f_min * f_max < 0:
            Y_sol = brentq(equation, Y_min, Y_max, xtol=1e-6)
            return float(Y_sol)
        else:
            # Try fsolve with initial guess between current X and delta
            Y_guess = (X_raff + X_delta) / 2.0
            Y_guess = np.clip(Y_guess, Y_min, Y_max)
            sol = fsolve(equation, Y_guess, full_output=True)
            Y_sol = float(sol[0][0])
            if Y_min <= Y_sol <= Y_max:
                return Y_sol

            # Try multiple initial guesses
            for Y_try in np.linspace(Y_min, Y_max, 20):
                sol = fsolve(equation, float(Y_try), full_output=True)
                Y_sol = float(sol[0][0])
                if Y_min <= Y_sol <= Y_max and abs(equation(Y_sol)) < 1e-4:
                    return Y_sol

    except (ValueError, RuntimeError):
        pass

    return None


def find_max_extract_purity(eq_model: EquilibriumModel) -> float:
    """
    Find the maximum achievable extract product purity (solvent-free basis).

    This is limited by the non-monotonicity of the Y(X) distribution curve
    at high X values. The max Y on the distribution curve is the limit.

    Returns
    -------
    Maximum Y value (solvent-free extract purity).
    """
    X_vals = np.linspace(0.0, eq_model.X_range[1], 500)
    Y_vals = np.array([eq_model.Y_from_X(float(x)) for x in X_vals])
    return float(np.max(Y_vals))
