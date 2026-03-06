"""
Simulation tab: run crosscurrent and countercurrent solvers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QHeaderView,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from ..core.equilibrium import EquilibriumModel


class SolverWorker(QThread):
    """Background thread for running solvers."""
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, solver_func, kwargs):
        super().__init__()
        self.solver_func = solver_func
        self.kwargs = kwargs

    def run(self):
        try:
            result = self.solver_func(**self.kwargs)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class SimulationTab(QWidget):
    """Tab for running extraction simulations."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.eq_model: Optional[EquilibriumModel] = None
        self.worker: Optional[SolverWorker] = None
        self.last_result = None
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)

        # Left: inputs
        left = QVBoxLayout()

        # Mode selector
        mode_group = QGroupBox("Extraction Mode")
        mode_layout = QVBoxLayout(mode_group)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            "Crosscurrent",
            "Countercurrent (Simple)",
            "Countercurrent (Reflux)",
        ])
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_layout.addWidget(self.mode_combo)
        left.addWidget(mode_group)

        # Feed parameters
        feed_group = QGroupBox("Feed Conditions")
        feed_layout = QFormLayout(feed_group)

        self.feed_A_spin = QDoubleSpinBox()
        self.feed_A_spin.setRange(0, 100); self.feed_A_spin.setValue(75.0)
        self.feed_A_spin.setSuffix(" wt%")
        feed_layout.addRow("Carrier (A):", self.feed_A_spin)

        self.feed_C_spin = QDoubleSpinBox()
        self.feed_C_spin.setRange(0, 100); self.feed_C_spin.setValue(25.0)
        self.feed_C_spin.setSuffix(" wt%")
        feed_layout.addRow("Solute (C):", self.feed_C_spin)

        self.feed_flow_spin = QDoubleSpinBox()
        self.feed_flow_spin.setRange(1, 100000); self.feed_flow_spin.setValue(100.0)
        self.feed_flow_spin.setSuffix(" kg")
        feed_layout.addRow("Feed flow:", self.feed_flow_spin)

        left.addWidget(feed_group)

        # Operation parameters
        op_group = QGroupBox("Operating Parameters")
        self.op_layout = QFormLayout(op_group)

        self.n_stages_spin = QSpinBox()
        self.n_stages_spin.setRange(1, 50); self.n_stages_spin.setValue(2)
        self.op_layout.addRow("Stages:", self.n_stages_spin)

        self.solvent_spin = QDoubleSpinBox()
        self.solvent_spin.setRange(1, 100000); self.solvent_spin.setValue(1000.0)
        self.solvent_spin.setSuffix(" kg")
        self.op_layout.addRow("Solvent/stage:", self.solvent_spin)

        # Reflux-specific inputs (hidden initially)
        self.reflux_spin = QDoubleSpinBox()
        self.reflux_spin.setRange(0.1, 100); self.reflux_spin.setValue(4.5)
        self.reflux_label = QLabel("Reflux ratio:")

        self.x_raff_spin = QDoubleSpinBox()
        self.x_raff_spin.setRange(0.001, 0.99); self.x_raff_spin.setValue(0.02)
        self.x_raff_spin.setDecimals(3)
        self.x_raff_label = QLabel("X_raff spec:")

        self.x_ext_spin = QDoubleSpinBox()
        self.x_ext_spin.setRange(0.01, 0.99); self.x_ext_spin.setValue(0.90)
        self.x_ext_spin.setDecimals(3)
        self.x_ext_label = QLabel("X_ext spec:")

        self._reflux_widgets = [
            (self.reflux_label, self.reflux_spin),
            (self.x_raff_label, self.x_raff_spin),
            (self.x_ext_label, self.x_ext_spin),
        ]
        for label, widget in self._reflux_widgets:
            self.op_layout.addRow(label, widget)
            label.hide()
            widget.hide()

        left.addWidget(op_group)

        # Run button
        self.run_btn = QPushButton("Run Simulation")
        self.run_btn.setMinimumHeight(40)
        self.run_btn.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.run_btn.clicked.connect(self._run_solver)
        left.addWidget(self.run_btn)

        self.status_label = QLabel("")
        left.addWidget(self.status_label)
        left.addStretch()

        main_layout.addLayout(left, stretch=1)

        # Right: results
        right = QVBoxLayout()

        # Results table
        table_group = QGroupBox("Results")
        table_layout = QVBoxLayout(table_group)
        self.results_table = QTableWidget()
        self.results_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        table_layout.addWidget(self.results_table)
        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        table_layout.addWidget(self.summary_label)
        right.addWidget(table_group, stretch=1)

        # Plot
        plot_group = QGroupBox("Stage Diagram")
        plot_layout = QVBoxLayout(plot_group)
        self.canvas = FigureCanvas(Figure(figsize=(8, 6)))
        plot_layout.addWidget(self.canvas)
        right.addWidget(plot_group, stretch=2)

        main_layout.addLayout(right, stretch=2)

    def set_model(self, eq_model: EquilibriumModel):
        self.eq_model = eq_model

    def _on_mode_changed(self, index):
        is_reflux = index == 2
        is_simple_cc = index == 1

        # Show/hide reflux-specific widgets
        for label, widget in self._reflux_widgets:
            if is_reflux:
                label.show(); widget.show()
            else:
                label.hide(); widget.hide()

        # Show/hide solvent spinner for countercurrent simple
        if is_reflux:
            self.solvent_spin.hide()
            self.n_stages_spin.hide()
        elif is_simple_cc:
            self.solvent_spin.show()
            self.n_stages_spin.show()
            self.feed_flow_spin.setValue(1000.0)
            self.solvent_spin.setSuffix(" kg/h")
            self.feed_flow_spin.setSuffix(" kg/h")
        else:
            self.solvent_spin.show()
            self.n_stages_spin.show()
            self.feed_flow_spin.setValue(100.0)
            self.solvent_spin.setSuffix(" kg")
            self.feed_flow_spin.setSuffix(" kg")

    def _run_solver(self):
        if self.eq_model is None:
            QMessageBox.warning(self, "No Model", "Load equilibrium data first.")
            return

        mode = self.mode_combo.currentIndex()
        self.run_btn.setEnabled(False)
        self.status_label.setText("Solving...")

        if mode == 0:  # Crosscurrent
            from ..core.crosscurrent import solve_crosscurrent
            kwargs = dict(
                feed_A=self.feed_A_spin.value(),
                feed_C=self.feed_C_spin.value(),
                feed_flow=self.feed_flow_spin.value(),
                solvent_per_stage=self.solvent_spin.value(),
                n_stages=self.n_stages_spin.value(),
                eq_model=self.eq_model,
            )
            self.worker = SolverWorker(solve_crosscurrent, kwargs)

        elif mode == 1:  # Countercurrent simple
            from ..core.countercurrent import solve_countercurrent_simple
            kwargs = dict(
                feed_A=self.feed_A_spin.value(),
                feed_C=self.feed_C_spin.value(),
                feed_flow=self.feed_flow_spin.value(),
                solvent_flow=self.solvent_spin.value(),
                n_stages=self.n_stages_spin.value(),
                eq_model=self.eq_model,
            )
            self.worker = SolverWorker(solve_countercurrent_simple, kwargs)

        else:  # Countercurrent with reflux
            from ..core.countercurrent import solve_countercurrent_reflux
            kwargs = dict(
                feed_A=self.feed_A_spin.value(),
                feed_C=self.feed_C_spin.value(),
                feed_flow=self.feed_flow_spin.value(),
                reflux_ratio=self.reflux_spin.value(),
                X_raff_spec=self.x_raff_spin.value(),
                X_ext_spec=self.x_ext_spin.value(),
                eq_model=self.eq_model,
            )
            self.worker = SolverWorker(solve_countercurrent_reflux, kwargs)

        self.worker.finished.connect(self._on_solver_done)
        self.worker.error.connect(self._on_solver_error)
        self.worker.start()

    def _on_solver_done(self, result):
        self.run_btn.setEnabled(True)
        self.last_result = result
        self.status_label.setText("Solver finished.")

        mode = self.mode_combo.currentIndex()

        if mode == 0:
            self._display_crosscurrent_results(result)
        elif mode == 1:
            self._display_countercurrent_results(result)
        else:
            self._display_reflux_results(result)

        # Notify heatmap tab and animation tab
        main = self.window()
        if hasattr(main, "heatmap_tab"):
            main.heatmap_tab.set_result(result)
        if hasattr(main, "animation_tab"):
            main.animation_tab.set_result(result)

    def _on_solver_error(self, msg):
        self.run_btn.setEnabled(True)
        self.status_label.setText("Solver error!")
        QMessageBox.critical(self, "Solver Error", msg)

    def _display_crosscurrent_results(self, result):
        stages = result.stages
        cols = ["Stage", "A_raff%", "C_raff%", "B_raff%", "R(kg)",
                "A_ext%", "C_ext%", "B_ext%", "E(kg)", "X", "Y", "Removal%"]
        self.results_table.setColumnCount(len(cols))
        self.results_table.setHorizontalHeaderLabels(cols)
        self.results_table.setRowCount(len(stages))

        for i, s in enumerate(stages):
            vals = [
                str(s.stage_number), f"{s.A_raff:.2f}", f"{s.C_raff:.2f}", f"{s.B_raff:.2f}",
                f"{s.R_flow:.1f}", f"{s.A_ext:.2f}", f"{s.C_ext:.2f}", f"{s.B_ext:.2f}",
                f"{s.E_flow:.1f}", f"{s.X_raff:.4f}", f"{s.Y_ext:.4f}", f"{s.pct_removal_cumul:.2f}",
            ]
            for j, v in enumerate(vals):
                item = QTableWidgetItem(v)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.results_table.setItem(i, j, item)

        self.summary_label.setText(
            f"<b>Summary:</b> {result.n_stages} stages, "
            f"Total removal: {result.total_pct_removal:.2f}%, "
            f"Final X_raff: {result.final_raff_X:.4f}, "
            f"Mixed extract flow: {result.mixed_extract_flow:.1f} kg"
        )
        self._plot_crosscurrent(result)

    def _display_countercurrent_results(self, result):
        stages = result.stages
        cols = ["Stage", "X_raff", "Y_ext", "A_raff%", "C_raff%", "B_raff%",
                "A_ext%", "C_ext%", "B_ext%"]
        self.results_table.setColumnCount(len(cols))
        self.results_table.setHorizontalHeaderLabels(cols)
        self.results_table.setRowCount(len(stages))

        for i, s in enumerate(stages):
            vals = [
                str(s.stage_number), f"{s.X_raff:.4f}", f"{s.Y_ext:.4f}",
                f"{s.A_raff:.2f}", f"{s.C_raff:.2f}", f"{s.B_raff:.2f}",
                f"{s.A_ext:.2f}", f"{s.C_ext:.2f}", f"{s.B_ext:.2f}",
            ]
            for j, v in enumerate(vals):
                item = QTableWidgetItem(v)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.results_table.setItem(i, j, item)

        self.summary_label.setText(
            f"<b>Summary:</b> {result.n_stages} stages, "
            f"Raffinate X: {stages[0].X_raff:.4f}, "
            f"Extract Y: {stages[-1].Y_ext:.4f}"
        )
        self._plot_countercurrent(result)

    def _display_reflux_results(self, result):
        self._display_countercurrent_results(result)

        extra = (
            f"<br><b>Reflux:</b> r = {result.reflux_ratio:.2f}, "
            f"Min stages: {result.min_stages}, "
            f"Min reflux: {result.min_reflux_ratio:.3f}, "
            f"Feed stage: {result.feed_stage}"
        )
        self.summary_label.setText(self.summary_label.text() + extra)

    def _plot_crosscurrent(self, result):
        import numpy as np
        self.canvas.figure.clear()
        ax = self.canvas.figure.add_subplot(111)

        data = self.eq_model.tie_line_data
        X_d = np.linspace(0, max(data.X)*1.05, 200)
        Y_d = [self.eq_model.Y_from_X(x) for x in X_d]

        ax.plot(X_d, Y_d, "b-", lw=2, label="Equilibrium")
        mv = max(max(data.X), max(data.Y))*1.1
        ax.plot([0, mv], [0, mv], "k--", lw=1, alpha=0.5, label="Y=X")

        # Stage stepping
        X_f = result.feed_C / (result.feed_A + result.feed_C)
        ax.plot(X_f, 0, "gs", markersize=10, label="Feed")

        for s in result.stages:
            # Vertical: X → Y (equilibrium)
            ax.plot([s.X_raff, s.X_raff], [s.X_raff, s.Y_ext], "r-", lw=1.5)
            # Horizontal: Y → next X
            ax.plot([s.X_raff], [s.Y_ext], "ro", markersize=5)

        ax.set_xlabel("X (raffinate)")
        ax.set_ylabel("Y (extract)")
        ax.set_title("Crosscurrent Stage Diagram")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        self.canvas.figure.tight_layout()
        self.canvas.draw()

    def _plot_countercurrent(self, result):
        import numpy as np
        self.canvas.figure.clear()
        ax = self.canvas.figure.add_subplot(111)

        data = self.eq_model.tie_line_data
        X_d = np.linspace(0, max(data.X)*1.05, 200)
        Y_d = [self.eq_model.Y_from_X(x) for x in X_d]

        ax.plot(X_d, Y_d, "b-", lw=2, label="Equilibrium")
        mv = max(max(data.X), max(data.Y))*1.1
        ax.plot([0, mv], [0, mv], "k--", lw=1, alpha=0.5, label="Y=X")

        # Plot stage points
        for s in result.stages:
            ax.plot(s.X_raff, s.Y_ext, "ro", markersize=6)
            ax.annotate(f" {s.stage_number}", (s.X_raff, s.Y_ext), fontsize=8)

        # Connect stages with lines
        X_vals = [s.X_raff for s in result.stages]
        Y_vals = [s.Y_ext for s in result.stages]
        ax.plot(X_vals, Y_vals, "r-", lw=1, alpha=0.7)

        ax.set_xlabel("X (raffinate)")
        ax.set_ylabel("Y (extract)")
        ax.set_title("Countercurrent Stage Diagram")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        self.canvas.figure.tight_layout()
        self.canvas.draw()
