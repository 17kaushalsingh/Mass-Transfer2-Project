# Countercurrent Extraction Formulation

This note explains how the countercurrent models in [`countercurrent.py`](/Users/17kaushalsingh/GitHub/Courses/Mass-Transfer2-Project/mass_transfer/core/countercurrent.py) are formulated and solved. The file contains two related problems:

- Simple countercurrent extraction without reflux
- Countercurrent extraction with extract reflux using a difference-point construction

The focus here is the mathematical setup and stage construction.

## Common Coordinate System

The countercurrent routines use the same solvent-free variables as the equilibrium model:

- `X = C / (A + C)` for the raffinate phase
- `Y = C / (A + C)` for the extract phase
- `N = B / (A + C)` for the solvent ratio

These variables are useful because:

- they separate the solvent from the carrier-solute subsystem
- tie-lines become relations between a raffinate `X` and an extract `Y`
- stage stepping can be done on `X-Y` and `N-(X/Y)` diagrams

The equilibrium model supplies:

- `Y = f(X)`
- its inverse `X = f^{-1}(Y)` numerically
- raffinate and extract `N` curves

So the countercurrent problem is posed as repeated alternation between:

- an equilibrium step
- an operating-line step

## 1. Simple Countercurrent Without Reflux

### Physical Layout

The simple column is arranged so that:

- feed enters at the raffinate end
- fresh solvent enters at the extract end
- the two phases move in opposite directions

The stage numbering in the solver runs from the extract end to the raffinate end.

## Formulation Idea

For a true countercurrent cascade, every stage is coupled to its neighbors in both directions:

- raffinate flows one way
- extract flows the opposite way

That makes the full problem harder than crosscurrent because the unknown stage compositions are globally coupled.

The implemented method avoids solving a large simultaneous nonlinear system. Instead, it uses an approximate stage-stepping construction on the `X-Y` diagram.

## Solvent-Free Feed Composition

The feed is reduced to:

`X_F = C_F / (A_F + C_F)`

and the solvent-free feed flow is:

`F' = F (A_F + C_F) / 100`

where `F'` is the amount of non-solvent material carried by the feed.

## Operating-Line Construction

The simple countercurrent method assumes that on the solvent-free `X-Y` diagram:

- the operating line is a straight line
- it passes through `(X_1, 0)`

where:

- `X_1` is the final raffinate composition at the solvent-entry end
- `Y = 0` represents the fresh-solvent end on a solvent-free basis

The second endpoint of the line is linked to the feed end through `X_F` and the terminal extract composition `Y_N`.

Because `Y_N` is not known a priori, the operating line is not available in closed form at the start. The code therefore treats `X_1` as the main unknown and iterates on the terminal extract composition while stepping through the stages.

## Stage-Stepping Logic

Given a trial `X_1`:

1. Apply equilibrium at stage 1:
   `Y_1 = f(X_1)`
2. Use the current operating-line estimate to move to the next raffinate composition:
   `X_2`
3. Apply equilibrium again:
   `Y_2 = f(X_2)`
4. Continue until stage `N`

The algorithm adjusts the operating-line endpoint estimate until the terminal extract composition becomes self-consistent.

Then it evaluates the residual:

- computed terminal raffinate composition versus feed-end target `X_F`

The correct `X_1` is the root of that residual.

## Root-Finding Strategy

The simple countercurrent routine uses:

- `brentq` when a bracket is available
- `fsolve` as fallback

Once `X_1` is found, the stage sequence is reconstructed in a final stepping pass.

## Interpretation

This method is a numerical McCabe-Thiele-like construction adapted to liquid-liquid extraction in solvent-free variables:

- equilibrium gives the curved relation `Y = f(X)`
- the operating line gives the interstage balance relation
- repeated stepping counts and locates the theoretical stages

## Important Approximation

This routine is explicitly approximate. The code comments already reflect that the exact countercurrent balance in variable-`N` liquid-liquid extraction is more involved.

The main simplifying assumptions are:

- the operating line is treated as straight in `X-Y` space
- the solvent-free flow structure is treated in an approximate way
- stage coupling is handled by stepping rather than a full simultaneous flow-composition solve

So this solver is best understood as a practical equilibrium-stage construction, not a rigorous MESH-equation column model.

## 2. Minimum Stages at Total Reflux

At total reflux, the operating line is taken as:

- `Y = X`

This is the analog of the total-reflux limit in distillation design: internal contacting exists, but no net product withdrawal distorts the stepping line.

The minimum-stage calculation proceeds by stepping between:

- the equilibrium curve `Y = f(X)`
- the line `Y = X`

The implementation starts from the raffinate specification `X_raff_spec` and repeats:

1. go to equilibrium: `Y_eq = f(X_curr)`
2. move to the total-reflux operating line: `X_next = Y_eq`

until the target extract specification is reached or exceeded.

This gives the minimum number of theoretical stages for the specified end purities under the model assumptions.

## 3. Minimum Reflux Ratio

The reflux design uses a pinch-point argument on the `N-(X/Y)` diagram.

The logic is:

- at minimum reflux, the enriching-section operating line just touches the equilibrium structure
- that limiting operating line is tied to the feed tie-line geometry

The implementation computes:

- feed solvent-free composition `X_F`
- equilibrium extract composition at the feed, `Y_F = f(X_F)`
- solvent ratios at the feed tie-line endpoints:
  `N_raff,F` and `N_ext,F`

The feed tie-line on the `N-(X/Y)` diagram is the line joining:

- `(X_F, N_raff,F)`
- `(Y_F, N_ext,F)`

That line is extrapolated to the specified extract product composition `X_ext_spec`, giving the enriching-section difference point at minimum reflux:

- `Delta_E,min = (X_ext_spec, N_Delta_E,min)`

The reflux ratio is then estimated from the position of this point relative to the extract-end `N` value:

`r_min = N_Delta_E,min / N_ext,product - 1`

The implementation clips the result at zero if the geometry predicts an extremely favorable separation.

## 4. Countercurrent Extraction With Reflux

### Physical Picture

In the reflux configuration, solvent-bearing extract leaving the top section is partially stripped of solvent and part of the concentrated liquid is returned as reflux.

This divides the column into two sections:

- Enriching section
- Stripping section

The feed stage is the transition point between them.

## Overall Solvent-Free Product Flows

The model first computes solvent-free product rates from the overall balance:

- feed solvent-free flow: `F'`
- extract product solvent-free flow: `PE'`
- raffinate product solvent-free flow: `RN'`

Using the specified end compositions:

`PE' = F' (X_F - X_raff_spec) / (X_ext_spec - X_raff_spec)`

`RN' = F' - PE'`

This is the overall solute balance written on a solvent-free basis.

## Difference Points

The reflux model is built around two difference points:

- `Delta_E` for the enriching section
- `Delta_S` for the stripping section

These are the liquid-liquid extraction analog of Ponchon-Savarit difference points.

### Enriching-Section Difference Point

At the extract-product end, the difference point is placed at:

- `X_Delta_E = X_ext_spec`
- `N_Delta_E = (1 + r) N_ext,product`

where `r` is the reflux ratio and `N_ext,product` is the solvent ratio of the extract equilibrium point at the product composition.

This comes from the section balance:

- the net top withdrawal is the extract product
- the actual stream leaving stage 1 is split into product plus reflux

So the difference point location depends directly on the reflux ratio.

### Stripping-Section Difference Point

The stripping-section difference point is obtained from the overall column balance:

- `Delta_E - Delta_S = Feed`

On the solvent-free diagram this gives explicit coordinates for `Delta_S` once `Delta_E`, `F'`, and `X_F` are known.

The code computes:

- `X_Delta_S`
- `N_Delta_S`

from the overall solvent-free and solvent-ratio balances.

## Stage Construction With Reflux

The reflux solver starts from the extract product end:

- `Y_curr = X_ext_spec`

For each stage:

1. Invert the equilibrium relation to find the conjugate raffinate:
   `X_eq = f^{-1}(Y_curr)`
2. Reconstruct the full raffinate and extract points and their solvent ratios.
3. Decide which section the stage belongs to:
   enriching until the feed composition is crossed, then stripping.
4. Draw the operating line through:
   - the current raffinate point `(X_eq, N_raff)`
   - the relevant difference point (`Delta_E` or `Delta_S`)
5. Intersect that line with the extract `N` curve to obtain the next extract composition `Y_next`

This is repeated until:

- the raffinate specification is reached
- or no further valid intersection is found

So the reflux column is solved by alternating between:

- equilibrium transfer between phases
- sectionwise operating-line transfer through a fixed difference point

## Geometric Meaning of `_find_next_Y_from_delta`

The helper routine `_find_next_Y_from_delta(...)` performs the key operating-line step.

Given:

- the current raffinate point `(X_raff, N_raff)`
- a section difference point `(X_Delta, N_Delta)`

it defines the straight line joining those two points. The next stage extract composition is the value of `Y` where this line intersects the extract `N_ext(Y)` curve.

Numerically, this is a scalar root-finding problem:

- line `N_line(Y)`
- curve `N_ext(Y)`
- solve `N_line(Y) - N_ext(Y) = 0`

The implementation tries:

- `brentq` if a sign change is found
- `fsolve` otherwise, with several starting guesses

## What the Reflux Solver Produces

The reflux routine returns:

- total number of stages
- feed-stage location
- per-stage `X`, `Y`, `N` values
- full ternary compositions of both phases
- enriching and stripping difference points
- reference values for minimum stages and minimum reflux

## Modeling Assumptions

Both countercurrent modes are equilibrium-stage models and therefore assume:

- ideal stagewise phase equilibrium
- no mass-transfer-rate limitation
- no holdup or hydraulics
- no heat effects
- validity of the fitted equilibrium correlations throughout the stepped region

The simple countercurrent routine adds an additional approximation:

- it uses a simplified straight operating line in `X-Y` space

The reflux routine is more geometric and better aligned with a Ponchon-Savarit-style construction because it works through explicit difference points on the `N-(X/Y)` diagram.

## Relation to the Implementation

The main routines map to the formulation as follows:

- `solve_countercurrent_simple(...)`
  approximate stage stepping on the `X-Y` diagram
- `find_min_stages(...)`
  total-reflux stepping between `Y = f(X)` and `Y = X`
- `find_min_reflux_ratio(...)`
  pinch-based estimate from feed tie-line geometry on the `N-(X/Y)` diagram
- `solve_countercurrent_reflux(...)`
  sectionwise difference-point stepping
- `_find_next_Y_from_delta(...)`
  numerical intersection of an operating line with the extract `N` curve

The important engineering point is that the countercurrent model is built as a stage-construction problem in solvent-free coordinates, using equilibrium relations plus operating-line geometry rather than a large general-purpose flowsheet solve.
