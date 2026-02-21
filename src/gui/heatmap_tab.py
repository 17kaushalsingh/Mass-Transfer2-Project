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
        for btn in [self.comp_btn, self.flow_btn, self.removal_btn, self.combined_btn]:
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
