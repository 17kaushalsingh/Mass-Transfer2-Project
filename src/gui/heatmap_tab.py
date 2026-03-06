"""
Heatmap tab: interactive heatmap visualizations of extraction results.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from PyQt6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from ..core.equilibrium import EquilibriumModel


class HeatmapTab(QWidget):
    """Tab for displaying extraction heatmaps."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.eq_model: Optional[EquilibriumModel] = None
        self.result = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Controls
        ctrl_layout = QHBoxLayout()

        self.comp_btn = QPushButton("Composition")
        self.comp_btn.setCheckable(True)
        self.comp_btn.setChecked(True)
        self.comp_btn.clicked.connect(lambda: self._show_heatmap("composition"))

        self.flow_btn = QPushButton("Flow Rates")
        self.flow_btn.setCheckable(True)
        self.flow_btn.clicked.connect(lambda: self._show_heatmap("flowrate"))

        self.removal_btn = QPushButton("% Removal")
        self.removal_btn.setCheckable(True)
        self.removal_btn.clicked.connect(lambda: self._show_heatmap("removal"))

        self.combined_btn = QPushButton("Combined View")
        self.combined_btn.setCheckable(True)
        self.combined_btn.clicked.connect(lambda: self._show_heatmap("combined"))

        self.export_btn = QPushButton("Export as PNG")
        self.export_btn.clicked.connect(self._export)

        for btn in [self.comp_btn, self.flow_btn, self.removal_btn, self.combined_btn]:
            ctrl_layout.addWidget(btn)

        # Additional visualizations
        ctrl_layout.addWidget(QLabel("  │"))  # visual separator

        self.profile_btn = QPushButton("📈 Profiles")
        self.profile_btn.setCheckable(True)
        self.profile_btn.setToolTip("Concentration & flow rate line charts vs stage")
        self.profile_btn.clicked.connect(lambda: self._show_profiles())
        ctrl_layout.addWidget(self.profile_btn)

        self.raff_ext_btn = QPushButton("⚖ Raff vs Ext")
        self.raff_ext_btn.setCheckable(True)
        self.raff_ext_btn.setToolTip("Raffinate vs Extract compositions at each stage")
        self.raff_ext_btn.clicked.connect(lambda: self._show_raff_vs_ext())
        ctrl_layout.addWidget(self.raff_ext_btn)

        self.removal_curve_btn = QPushButton("📉 Removal Curve")
        self.removal_curve_btn.setCheckable(True)
        self.removal_curve_btn.setToolTip("Per-stage and cumulative % removal line chart")
        self.removal_curve_btn.clicked.connect(lambda: self._show_removal_curve())
        ctrl_layout.addWidget(self.removal_curve_btn)

        ctrl_layout.addStretch()
        ctrl_layout.addWidget(self.export_btn)

        layout.addLayout(ctrl_layout)

        # Canvas
        self.canvas = FigureCanvas(Figure(figsize=(14, 8)))
        layout.addWidget(self.canvas)

        self.info_label = QLabel("Run a simulation first to view heatmaps.")
        layout.addWidget(self.info_label)

    def set_model(self, eq_model: EquilibriumModel):
        self.eq_model = eq_model

    def set_result(self, result):
        self.result = result
        self.info_label.setText("Click a heatmap type to visualize.")
        self._show_heatmap("combined")

    def _show_heatmap(self, hmap_type: str):
        if self.result is None:
            QMessageBox.information(self, "No Data", "Run a simulation first.")
            return

        # Update button states
        for btn in [self.comp_btn, self.flow_btn, self.removal_btn, self.combined_btn,
                    self.profile_btn, self.raff_ext_btn, self.removal_curve_btn]:
            btn.setChecked(False)

        if hmap_type == "composition":
            self.comp_btn.setChecked(True)
        elif hmap_type == "flowrate":
            self.flow_btn.setChecked(True)
        elif hmap_type == "removal":
            self.removal_btn.setChecked(True)
        elif hmap_type == "combined":
            self.combined_btn.setChecked(True)

        # Check if result is crosscurrent (has R_flow attribute on stages)
        from ..core.crosscurrent import CrosscurrentResult
        if not isinstance(self.result, CrosscurrentResult):
            self._show_countercurrent_heatmap()
            return

        from ..viz.heatmaps import (
            composition_heatmap,
            flowrate_heatmap,
            removal_heatmap,
            combined_heatmap,
        )

        self.canvas.figure.clear()

        if hmap_type == "composition":
            fig = composition_heatmap(self.result)
        elif hmap_type == "flowrate":
            fig = flowrate_heatmap(self.result)
        elif hmap_type == "removal":
            fig = removal_heatmap(self.result)
        else:
            fig = combined_heatmap(self.result)

        # Copy figure content to canvas
        self.canvas.figure = fig
        self.canvas.draw()

    def _show_countercurrent_heatmap(self):
        """Show a simple heatmap for countercurrent results."""
        import numpy as np
        import pandas as pd
        import seaborn as sns

        self.canvas.figure.clear()
        stages = self.result.stages
        labels = [f"S{s.stage_number}" for s in stages]

        ax = self.canvas.figure.add_subplot(111)
        data = pd.DataFrame({
            "X_raff": [s.X_raff for s in stages],
            "Y_ext": [s.Y_ext for s in stages],
            "N_raff": [s.N_raff for s in stages],
            "N_ext": [s.N_ext for s in stages],
        }, index=labels).T

        sns.heatmap(data, annot=True, fmt=".3f", cmap="YlOrRd",
                    linewidths=0.5, ax=ax)
        ax.set_title("Countercurrent Stage Compositions")
        self.canvas.figure.tight_layout()
        self.canvas.draw()

    def _export(self):
        if self.canvas.figure is None:
            return
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Heatmap", "heatmap.png",
            "PNG Files (*.png);;SVG Files (*.svg);;All Files (*)"
        )
        if filepath:
            self.canvas.figure.savefig(filepath, dpi=200, bbox_inches="tight")
            QMessageBox.information(self, "Exported", f"Saved to {filepath}")

    # ------------------------------------------------------------------
    # Additional visualizations
    # ------------------------------------------------------------------

    def _uncheck_all(self):
        for btn in [self.comp_btn, self.flow_btn, self.removal_btn, self.combined_btn,
                    self.profile_btn, self.raff_ext_btn, self.removal_curve_btn]:
            btn.setChecked(False)

    def _show_profiles(self):
        """Line charts: concentration and flow rate profiles vs stage."""
        if self.result is None:
            QMessageBox.information(self, "No Data", "Run a simulation first.")
            return

        from ..core.crosscurrent import CrosscurrentResult
        if not isinstance(self.result, CrosscurrentResult):
            QMessageBox.information(self, "Info",
                "Profile charts are currently available for crosscurrent results.")
            return

        self._uncheck_all()
        self.profile_btn.setChecked(True)

        import numpy as np
        stages = self.result.stages
        x = [s.stage_number for s in stages]

        self.canvas.figure.clear()
        fig = self.canvas.figure
        ax1, ax2 = fig.subplots(1, 2)

        # Left: concentration profiles
        ax1.plot(x, [s.A_raff for s in stages], 'o-', lw=2, ms=6, color='#4e79a7', label='A (Carrier)')
        ax1.plot(x, [s.C_raff for s in stages], 's-', lw=2, ms=6, color='#e15759', label='C (Solute)')
        ax1.plot(x, [s.B_raff for s in stages], '^-', lw=2, ms=6, color='#76b7b2', label='B (Solvent)')
        ax1.set_xlabel('Stage Number', fontsize=12)
        ax1.set_ylabel('Raffinate Composition (wt%)', fontsize=12)
        ax1.set_title('Concentration Profile — Raffinate', fontsize=13)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(x)

        # Right: flow rate profiles
        ax2.plot(x, [s.R_flow for s in stages], 'o-', lw=2, ms=6, color='#4e79a7', label='Raffinate R')
        ax2.plot(x, [s.E_flow for s in stages], 's-', lw=2, ms=6, color='#e15759', label='Extract E')
        ax2.set_xlabel('Stage Number', fontsize=12)
        ax2.set_ylabel('Flow Rate (kg)', fontsize=12)
        ax2.set_title('Flow Rate Profile', fontsize=13)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(x)

        fig.suptitle('Concentration & Flow Rate Profiles', fontsize=14, y=1.01)
        fig.tight_layout()
        self.canvas.draw()

    def _show_raff_vs_ext(self):
        """Grouped bar chart: raffinate vs extract compositions at each stage."""
        if self.result is None:
            QMessageBox.information(self, "No Data", "Run a simulation first.")
            return

        from ..core.crosscurrent import CrosscurrentResult
        if not isinstance(self.result, CrosscurrentResult):
            QMessageBox.information(self, "Info",
                "Raffinate vs Extract chart is currently available for crosscurrent results.")
            return

        self._uncheck_all()
        self.raff_ext_btn.setChecked(True)

        import numpy as np
        stages = self.result.stages
        n = len(stages)
        x = np.arange(n)
        w = 0.13  # bar width

        self.canvas.figure.clear()
        fig = self.canvas.figure
        ax = fig.add_subplot(111)

        # Raffinate bars (solid)
        ax.bar(x - 2.5*w, [s.A_raff for s in stages], w, color='#4e79a7', label='A raff')
        ax.bar(x - 1.5*w, [s.C_raff for s in stages], w, color='#e15759', label='C raff')
        ax.bar(x - 0.5*w, [s.B_raff for s in stages], w, color='#76b7b2', label='B raff')

        # Extract bars (hatched)
        ax.bar(x + 0.5*w, [s.A_ext for s in stages], w, color='#4e79a7', alpha=0.5,
               hatch='//', label='A ext', edgecolor='#4e79a7')
        ax.bar(x + 1.5*w, [s.C_ext for s in stages], w, color='#e15759', alpha=0.5,
               hatch='//', label='C ext', edgecolor='#e15759')
        ax.bar(x + 2.5*w, [s.B_ext for s in stages], w, color='#76b7b2', alpha=0.5,
               hatch='//', label='B ext', edgecolor='#76b7b2')

        ax.set_xlabel('Stage Number', fontsize=12)
        ax.set_ylabel('Composition (wt%)', fontsize=12)
        ax.set_title('Raffinate vs Extract Compositions at Each Stage', fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels([f'S{s.stage_number}' for s in stages])
        ax.legend(fontsize=9, ncol=2, loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')

        fig.tight_layout()
        self.canvas.draw()

    def _show_removal_curve(self):
        """Per-stage and cumulative % removal line chart."""
        if self.result is None:
            QMessageBox.information(self, "No Data", "Run a simulation first.")
            return

        from ..core.crosscurrent import CrosscurrentResult
        if not isinstance(self.result, CrosscurrentResult):
            QMessageBox.information(self, "Info",
                "Removal curve is currently available for crosscurrent results.")
            return

        self._uncheck_all()
        self.removal_curve_btn.setChecked(True)

        stages = self.result.stages
        x = [s.stage_number for s in stages]
        per_stage = [s.pct_removal_stage for s in stages]
        cumulative = [s.pct_removal_cumul for s in stages]

        self.canvas.figure.clear()
        fig = self.canvas.figure
        ax1 = fig.add_subplot(111)

        # Per-stage removal as bars
        bars = ax1.bar(x, per_stage, color='#59a14f', alpha=0.7, label='Per-stage removal',
                       edgecolor='white', zorder=2)
        for bar, val in zip(bars, per_stage):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     f'{val:.1f}%', ha='center', va='bottom', fontsize=9, color='#59a14f')

        # Cumulative on secondary axis
        ax2 = ax1.twinx()
        ax2.plot(x, cumulative, 'o-', lw=2.5, ms=8, color='#e15759',
                 label='Cumulative removal', zorder=3)
        for xi, yi in zip(x, cumulative):
            ax2.annotate(f'{yi:.1f}%', (xi, yi), textcoords='offset points',
                         xytext=(0, 10), ha='center', fontsize=9, color='#e15759',
                         fontweight='bold')

        ax1.set_xlabel('Stage Number', fontsize=12)
        ax1.set_ylabel('Per-Stage Removal (%)', fontsize=12, color='#59a14f')
        ax2.set_ylabel('Cumulative Removal (%)', fontsize=12, color='#e15759')
        ax1.set_title('Solute Removal Efficiency Curve', fontsize=13)
        ax1.set_xticks(x)
        ax2.set_ylim(0, 105)
        ax1.grid(True, alpha=0.3)

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='center right')

        fig.tight_layout()
        self.canvas.draw()
