# AI Assistant Instructions

## General Rules

- **Language:** Python only. No MATLAB.
- **Environment:** Always use the project venv at `./venv/`. Activate before any `pip` or `python` commands.
- **Incremental Steps:** Do not generate large amounts of code without user approval. Propose module stubs, confirm the math, then iterate.
- **No Graphical Methods:** All solvers must use `scipy.optimize.fsolve` (or equivalent numerical root-finding). Graphical stage-stepping is for visualization only, not computation.

## Architecture

- All source code lives under `src/` with subpackages `core/`, `viz/`, `ml/`, `gui/`.
- Follow the project structure defined in README.md exactly.
- Every module must be importable independently (`__init__.py` files).
- Use relative imports within the `src` package.

## Implementation Plan (follow this order)

1. **Phase 0 — Equilibrium Model** (`src/core/equilibrium.py`)
   - Load data.json, convert to solvent-free basis (X, Y, N).
   - Fit polynomial curves: raffinate curve, extract curve, distribution curve (Y vs X), N vs X, N vs Y, conjugate line.
   - Provide callable interpolation functions.
   - Plot: right-angle triangle, N vs X/Y, X vs Y distribution diagram.

2. **Phase 1 — Crosscurrent Solver** (`src/core/crosscurrent.py`)
   - Formulate mass balance + equilibrium as system of nonlinear equations.
   - Solve with `fsolve` for N stages simultaneously.
   - Validate with Problem Part (ii): 100 kg feed, 25% acid, 2 stages, 1000 kg propane each.

3. **Phase 2 — Countercurrent Solver** (`src/core/countercurrent.py`)
   - Simple countercurrent (no reflux) first.
   - Then full reflux implementation: difference-point method, min stages, min reflux ratio, design for given reflux.
   - Validate with Problem Part (iii): 1000 kg/h, 25% acid, products at 2% and 90% (solvent-free), reflux ratio 4.5.

4. **Phase 3 — Visualization** (`src/viz/`)
   - `ternary_plots.py`: Right-angle triangle diagrams, N-X-Y plots, X-Y distribution.
   - `heatmaps.py`: Stage-wise composition, flow rate, and % removal heatmaps (Seaborn + Plotly).
   - `surfaces.py`: 3D response surface and contour plots (Plotly).

5. **Phase 4 — Surrogate Model** (`src/ml/`)
   - `data_generator.py`: Sweep solver over (n_stages, solvent_amount, feed_composition) grid → DataFrame.
   - `neural_net.py`: PyTorch ANN (Input→64→32→1), train/val/test split, Adam+MSE, save checkpoint.
   - `optimization.py`: Use trained model for dense response surfaces, find optimal operating point.

6. **Phase 5 — GUI** (`src/gui/`)
   - PyQt6 desktop application with 4 tabs: Data Input, Simulation, Heatmaps, Surrogate Model.
   - Embed Matplotlib/Plotly plots inside Qt widgets.
   - Professional styling, progress bars for long operations.

7. **Phase 6 — Testing & Validation** (`tests/`)
   - Unit tests for each solver against known problem answers.
   - End-to-end validation for the cottonseed oil system.

## Mathematical Conventions

- **Components:** A = carrier (cottonseed oil), B = solvent (propane), C = solute (oleic acid).
- **Ternary coordinates:** wt% basis, right-angle triangle (x = A, y = C, B = 100 - A - C).
- **Solvent-free basis:** X = C/(A+C) in raffinate, Y = C/(A+C) in extract, N = B/(A+C).
- **Tie-lines:** `data.json` row index i in phase_1 connects to row index i in phase_2.

## Code Quality

- Type hints on all public functions.
- Docstrings explaining mathematical meaning of parameters.
- No magic numbers — define constants (e.g., component names, default values).
- Return structured results (dataclasses or named tuples), not raw tuples.

## Visualization Style

- Use Plotly for interactive plots (embedded in GUI), Matplotlib/Seaborn for static/export.
- Heatmaps: annotated cells, clear axis labels, colorbar.
- 3D surfaces: axis labels with units, hover info.
- All plots must have titles, axis labels, and legends where applicable.

## GUI Rules

- Framework: PyQt6.
- Generalized: user can load any ternary equilibrium data JSON.
- Default: load cottonseed oil data from `data.json` on startup.
- Thread long computations (solver, NN training) to keep GUI responsive.

## Group Members

- 3 members. Names and exact role mapping to be specified by user.
- Member 1: Core solvers. Member 2: Visualization & data generation. Member 3: ML & GUI.
